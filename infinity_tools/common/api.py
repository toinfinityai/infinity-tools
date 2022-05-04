import os
import time
import urllib
import uuid
import zipfile
from tqdm.notebook import tqdm
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from serde import serde, serialize, deserialize, field
from serde.json import from_json, to_json


@serde
@dataclass(frozen=True)
class FailedJobRequest:
    status_code: int
    params: Dict


@serde
@dataclass(frozen=True)
class SuccessfulJobRequest:
    job_id: str
    params: Dict


@dataclass(frozen=True)
class CompletedJob:
    job_id: str
    status_code: int
    params: Dict
    result_url: Optional[str] = None


@serialize
@deserialize
@dataclass(frozen=True)
class Batch:
    uid: str
    timestamp: str
    batch_folder_suffix: Optional[str]
    jobs: List[SuccessfulJobRequest]
    failed_requests: List[FailedJobRequest]
    generator: str
    server: str
    query_endpoint: str
    output_dir: str
    token: str = field(default="", metadata={"serde_skip": True})

    @property
    def batch_dir(self) -> str:
        if self.batch_folder_suffix:
            return f"{self.timestamp}_{self.batch_folder_suffix[0:30]}"
        else:
            return f"{self.timestamp}"

    @property
    def job_ids(self) -> List[str]:
        return [j.job_id for j in self.jobs]

    @property
    def num_successfully_submitted_jobs(self) -> int:
        return len(self.job_ids)

    def to_json(self) -> str:
        return to_json(self, indent=4)

    @classmethod
    def from_batch_file(cls, json_file_path: str, token: str) -> "Batch":
        with open(json_file_path, "r") as f:
            json_str = f.read()
        return cls.from_json(json_str=json_str, token=token)

    @classmethod
    def from_batch_folder(cls, batch_folder_path: str, token: str) -> "Batch":
        json_file_path = os.path.join(batch_folder_path, "batch.json")
        return cls.from_batch_file(json_file_path=json_file_path, token=token)

    @classmethod
    def from_json(cls, json_str: str, token: str) -> "Batch":
        deserialized_batch = from_json(cls, json_str)
        return replace(deserialized_batch, token=token)

    def get_completed_jobs(self) -> List[CompletedJob]:
        """Returns list of completed jobs in the batch."""
        job_ids = [j.job_id for j in self.jobs]
        completed_jobs = []
        for jid in tqdm(job_ids, desc="Polling jobs"):
            r = requests.get(
                f"{self.server}{self.query_endpoint}{jid}/",
                headers={"Authorization": f"Token {self.token}"},
            )
            json_payload = r.json()
            if json_payload["in_progress"]:
                continue
            job_url = json_payload["result_url"] if r.status_code == 200 else None
            completed_jobs.append(
                CompletedJob(
                    job_id=jid,
                    status_code=r.status_code,
                    params=json_payload["param_values"],
                    result_url=job_url,
                )
            )
        return completed_jobs

    def get_completed_jobs_valid_and_invalid(
        self,
    ) -> Tuple[List[CompletedJob], List[CompletedJob]]:
        """Returns tuple of valid and invalid `CompletedJob` objects."""
        completed_valid_jobs = []
        completed_invalid_jobs = []
        completed_jobs = self.get_completed_jobs()
        for cj in completed_jobs:
            if cj.result_url:
                completed_valid_jobs.append(cj)
            else:
                completed_invalid_jobs.append(cj)
        return completed_valid_jobs, completed_invalid_jobs

    def await_jobs(
        self, timeout: int = 3 * 60 * 60, polling_interval: int = 10
    ) -> List[CompletedJob]:
        """Waits for all jobs in batch to complete.

        Args:
            timeout: Maximum allowable time to wait for jobs in batch to complete (in seconds).
            polling_interval: Time interval to sleep (in seconds) between consecutive iterations
                of polling.

        Returns:
            List of all `CompletedJobs` in batch.
        """
        num_jobs = len(self.jobs)
        if num_jobs == 0:
            return []
        job_ids = [j.job_id for j in self.jobs]
        jobs_dict = {j.job_id: j.params for j in self.jobs}
        remaining_job_ids = set(job_ids)
        start_time = datetime.now()
        jobs_in_progress = True
        completed_jobs = 0

        while jobs_in_progress:
            current_iteration_ids = list(remaining_job_ids)
            elapsed_time = int((datetime.now() - start_time).seconds)
            print(
                f"{len(remaining_job_ids)} remaining jobs [{elapsed_time:d} s]...\t\t\t",
                end="\r",
            )
            for jid in current_iteration_ids:
                r = requests.get(
                    f"{self.server}{self.query_endpoint}{jid}/",
                    headers={"Authorization": f"Token {self.token}"},
                )
                if not r.json()["in_progress"]:
                    remaining_job_ids.remove(jid)
                    completed_jobs += 1
                time.sleep(0.01)
            if completed_jobs == num_jobs:
                jobs_in_progress = False
            if (datetime.now() - start_time).seconds > timeout:
                raise TimeoutError()
            time.sleep(polling_interval)

        duration = datetime.now() - start_time
        print(f"Duration for all jobs: {duration.seconds} [s]")

        completed_jobs = []
        for jid in job_ids:
            r = requests.get(
                f"{self.server}{self.query_endpoint}{jid}/",
                headers={"Authorization": f"Token {self.token}"},
            )
            job_result_url = r.json()["result_url"] if r.status_code == 200 else None
            completed_jobs.append(
                CompletedJob(
                    job_id=jid,
                    status_code=r.status_code,
                    params=jobs_dict[jid],
                    result_url=job_result_url,
                )
            )

        return completed_jobs


def job_ids_from_completed_jobs(completed_jobs: List[CompletedJob]) -> List[str]:
    """Returns job ids from a list of completed jobs."""
    return [cj.job_id for cj in completed_jobs]


def filter_for_valid_ids(completed_jobs: List[CompletedJob]) -> List[str]:
    """Returns valid job ids (result URL) from a list of completed jobs."""
    return [cj.job_id for cj in completed_jobs if cj.result_url]


def fetch_params_by_id(
    server: str,
    endpoint: str,
    token: str,
    job_id: str,
) -> Dict:
    """Returns parameters corresponding to specific job id."""

    r = requests.get(
        f"{server}{endpoint}{job_id}/",
        headers={"Authorization": f"Token {token}"},
    )
    return r.json()["param_values"]


def submit_batch_to_api(
    batch_folder_suffix: Optional[str],
    generator: str,
    server: str,
    run_endpoint: str,
    query_endpoint: str,
    token: str,
    output_dir: str,
    job_params: List[Dict],
    write_submission_status_to_file: bool = True,
) -> Tuple[Batch, Optional[str]]:
    """Submits a batch of jobs to the API.

    Returns:
        Tuple of corresponding `Batch` object and a path to its metadata on disk.
    """

    batch_uid = uuid.uuid4().hex
    batch_time = datetime.now()
    batch_timestamp = batch_time.strftime("%Y%m%d_T%H%M%S%f")

    successful_requests = []
    failed_requests = []

    print("Submitting jobs to API...")
    for params in tqdm(job_params):
        r = requests.post(
            f"{server}{run_endpoint}",
            json={
                "name": generator,
                "param_values": params,
            },
            headers={
                "Authorization": f"Token {token}",
                "Content-Type": "application/json",
            },
        )
        if r.status_code == 201:
            job_id = r.json()["id"]
            successful_requests.append(
                SuccessfulJobRequest(job_id=job_id, params=params)
            )
        else:
            failed_requests.append(
                FailedJobRequest(status_code=r.status_code, params=params)
            )
        time.sleep(0.05)

    batch = Batch(
        uid=batch_uid,
        timestamp=batch_timestamp,
        batch_folder_suffix=batch_folder_suffix,
        jobs=successful_requests,
        failed_requests=failed_requests,
        generator=generator,
        server=server,
        query_endpoint=query_endpoint,
        token=token,
        output_dir=output_dir,
    )

    if write_submission_status_to_file:
        batch_path = os.path.join(output_dir, f"{batch.batch_dir}")
        os.makedirs(batch_path, exist_ok=False)
        serialized_batch_file = os.path.join(batch_path, "batch.json")
        with open(serialized_batch_file, "w") as f:
            f.write(batch.to_json())
    else:
        batch_path = None

    return batch, batch_path


def download_completed_jobs(
    completed_jobs: List[CompletedJob], output_dir: str
) -> List[str]:
    """Downloads completed jobs to output directory.

    Returns:
        List of folders corresponding to each completed job.
    """
    os.makedirs(output_dir, exist_ok=True)
    job_folders = []
    print("Downloading completed jobs...")
    for job in tqdm(completed_jobs):
        zip_file = os.path.join(output_dir, job.job_id + ".zip")
        if job.result_url is None:
            continue
        urllib.request.urlretrieve(job.result_url, zip_file)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_file)
        job_folders.append(os.path.join(output_dir, job.job_id))

    return job_folders
