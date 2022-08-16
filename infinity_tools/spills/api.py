import os
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import requests

import infinity_tools.common.api as ca
from infinity_tools.common.api import Batch

SERVER_URL = "https://api.toinfinity.ai"
GENERATOR = "spills-v0.1.0"
RUN_ENDPOINT = "/api/jobs/run/"
RUN_QUERY_ENDPOINT = "/api/job_runs/"
PREVIEW_ENDPOINT = "/api/jobs/preview/"
PREVIEW_QUERY_ENDPOINT = "/api/job_previews/"


ALL_PARAM_KEYS = []
SCENE_LST = []
COLOR_LST = []
FRAME_RATE_LST = []
NAME_TO_OPTIONS = {
    "scene": SCENE_LST,
    "color": COLOR_LST,
    "frame_rate": FRAME_RATE_LST,
}


def _get_all_params(token: str) -> List[Any]:

    if not ALL_PARAM_KEYS:
        fetch_parameter_options(token)

    return ALL_PARAM_KEYS


def _get_param_options(param: str, token: str) -> List[Any]:

    if not NAME_TO_OPTIONS[param]:
        fetch_parameter_options(token)

    return NAME_TO_OPTIONS[param]


def fetch_parameter_options(token: str, server_url: Optional[str] = None):

    if server_url is None:
        _server_url = SERVER_URL
    else:
        _server_url = server_url

    r = requests.get(
        f"{_server_url}/api/jobs/{GENERATOR}/",
        headers={"Authorization": f"Token {token}"},
    )
    params = r.json()["params"]
    params = {e["name"]: e for e in params}

    for name, lst in NAME_TO_OPTIONS.items():
        if not lst:
            lst.extend(params[name]["options"]["choices"])

    if not ALL_PARAM_KEYS:
        ALL_PARAM_KEYS.extend([k for k in params.keys()])


class JobType(Enum):
    PREVIEW = auto()
    VIDEO = auto()


def sample_input(
    token: str,
    scene: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
    profile_irregularity: Optional[float] = None,
    depth: Optional[float] = None,
    frame_rate: Optional[int] = None,
    video_duration: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Dict:

    if scene is None:
        scene = str(np.random.choice(_get_param_options("scene", token)))
    else:
        if scene not in _get_param_options("scene", token):
            raise ValueError(f"`scene` ({scene}) not in supported scene list ({_get_param_options('scene', token)})")

    if color is None:
        color = str(np.random.choice(_get_param_options("color", token)))
    else:
        if color not in _get_param_options("color", token):
            raise ValueError(f"`color` ({color}) not in supported color list ({_get_param_options('color', token)})")

    if size is None:
        size = float(np.random.uniform(1.0, 100.0))
    else:
        if not (1.0 <= size <= 100.0):
            raise ValueError(f"`size` ({size}) must be in range [1.0, 100.0]")

    if profile_irregularity is None:
        profile_irregularity = float(np.random.uniform(0.0, 1.0))
    else:
        if not (0.0 <= profile_irregularity <= 1.0):
            raise ValueError(f"`profile_irregularity` ({profile_irregularity}) must be in range [0.0, 1.0]")

    if aspect_ratio is None:
        aspect_ratio = float(np.random.uniform(1.0, 3.0))
    else:
        if not (1.0 <= aspect_ratio <= 3.0):
            raise ValueError(f"`aspect_ratio` ({aspect_ratio}) must be in range [1.0, 3.0]")

    if depth is None:
        depth = float(np.random.uniform(1.0, 4.0))
    else:
        if not (1.0 <= depth <= 4.0):
            raise ValueError(f"`depth` ({depth}) must be in range [1.0, 8.0]")

    if frame_rate is None:
        frame_rate = int(np.random.choice(_get_param_options("frame_rate", token)))
    else:
        if frame_rate not in _get_param_options("frame_rate", token):
            raise ValueError(
                f"`frame_rate` ({frame_rate}) not in supported frame rate list ({_get_param_options('frame_rate', token)})"
            )

    if video_duration is None:
        video_duration = float(np.random.uniform(1.0, 5.0))
    else:
        if not (1.0 <= video_duration <= 120.0):
            raise ValueError(f"`video_duration` ({video_duration}) must be in range [1.0, 120.0]")

    if random_seed is None:
        random_seed = int(np.random.randint(low=0, high=2**31))
    else:
        if not (0 <= random_seed <= (2**31 - 1)):
            raise ValueError(f"`random_seed` ({random_seed}) must be in range [0, 2^31 - 1]")

    params_dict = {
        "scene": scene,
        "color": color,
        "size": size,
        "aspect_ratio": aspect_ratio,
        "profile_irregularity": profile_irregularity,
        "depth": depth,
        "frame_rate": frame_rate,
        "video_duration": video_duration,
        "random_seed": random_seed,
    }

    return params_dict


def submit_preview_batch_to_api(
    token: str,
    preview_params: List[Dict],
    output_dir: str,
    batch_folder_suffix: Optional[str] = None,
) -> Tuple[Batch, Optional[str]]:

    return ca.submit_batch_to_api(
        batch_folder_suffix=batch_folder_suffix,
        generator=GENERATOR,
        server=SERVER_URL,
        run_endpoint=PREVIEW_ENDPOINT,
        query_endpoint=PREVIEW_QUERY_ENDPOINT,
        token=token,
        output_dir=output_dir,
        job_params=preview_params,
        write_submission_status_to_file=True,
    )


def submit_video_batch_to_api(
    token: str,
    job_params: List[Dict],
    output_dir: str,
    batch_folder_suffix: Optional[str] = None,
) -> Tuple[Batch, Optional[str]]:

    return ca.submit_batch_to_api(
        batch_folder_suffix=batch_folder_suffix,
        generator=GENERATOR,
        server=SERVER_URL,
        run_endpoint=RUN_ENDPOINT,
        query_endpoint=RUN_QUERY_ENDPOINT,
        token=token,
        output_dir=output_dir,
        job_params=job_params,
        write_submission_status_to_file=True,
    )


def download_completed_previews(completed_previews: List[ca.CompletedJob], output_dir: str) -> List[str]:
    return ca.download_completed_jobs(completed_jobs=completed_previews, output_dir=output_dir)


def download_completed_videos(completed_jobs: List[ca.CompletedJob], output_dir: str) -> List[str]:
    return ca.download_completed_jobs(completed_jobs=completed_jobs, output_dir=output_dir)


def get_all_preview_images(folders: List[str]) -> List[str]:
    return [os.path.join(f, "video_preview.png") for f in folders]


def fetch_preview_params_by_id(
    token: str,
    preview_id: str,
) -> Dict:
    return ca.fetch_params_by_id(
        server=SERVER_URL,
        endpoint=PREVIEW_QUERY_ENDPOINT,
        token=token,
        job_id=preview_id,
    )


def fetch_job_params_by_id(
    token: str,
    job_id: str,
) -> Dict:
    return ca.fetch_params_by_id(
        server=SERVER_URL,
        endpoint=RUN_QUERY_ENDPOINT,
        token=token,
        job_id=job_id,
    )


def _expand_overrides_across_each_base(
    seed_ty: JobType,
    token: str,
    base_state_ids: List[str],
    override_params: List[Dict],
) -> List[Dict]:
    if seed_ty not in {JobType.PREVIEW, JobType.VIDEO}:
        raise ValueError(f"`seed_ty` ({seed_ty}) is not supported")

    for override_dict in override_params:
        if not all([k in _get_all_params(token=token) for k in override_dict.keys()]):
            raise ValueError("Not all override parameters are supported")

    params_with_overrides = []
    for seed in base_state_ids:
        if seed_ty == JobType.PREVIEW:
            original_params = fetch_preview_params_by_id(token=token, preview_id=seed)
        else:
            original_params = fetch_job_params_by_id(token=token, job_id=seed)

        params_with_overrides.extend([{**original_params, **op} for op in override_params])

    return params_with_overrides


def expand_overrides_across_each_preview_state(
    token: str,
    base_state_ids: List[str],
    override_params: List[Dict],
) -> List[Dict]:
    return _expand_overrides_across_each_base(
        seed_ty=JobType.PREVIEW,
        token=token,
        base_state_ids=base_state_ids,
        override_params=override_params,
    )


def expand_overrides_across_each_video_state(
    token: str,
    base_state_ids: List[str],
    override_params: List[Dict],
) -> List[Dict]:
    return _expand_overrides_across_each_base(
        seed_ty=JobType.VIDEO,
        token=token,
        base_state_ids=base_state_ids,
        override_params=override_params,
    )


def _submit_rerun_batch_with_overrides(
    job_ty: JobType,
    seed_ty: JobType,
    token: str,
    base_state_id: str,
    override_params: List[Dict],
    output_folder: str,
    batch_folder_suffix: Optional[str] = None,
) -> Tuple[Batch, Optional[str]]:
    if job_ty not in {JobType.PREVIEW, JobType.VIDEO}:
        raise ValueError(f"`job_ty` ({job_ty}) is not supported")
    if seed_ty not in {JobType.PREVIEW, JobType.VIDEO}:
        raise ValueError(f"`seed_ty` ({seed_ty}) is not supported")

    for override_dict in override_params:
        if not all([k in _get_all_params(token=token) for k in override_dict.keys()]):
            raise ValueError("Not all override parameters are supported")

    if seed_ty == JobType.PREVIEW:
        original_params = fetch_preview_params_by_id(token=token, preview_id=base_state_id)
    else:
        original_params = fetch_job_params_by_id(token=token, job_id=base_state_id)

    params_with_overrides = [{**original_params, **op} for op in override_params]

    if job_ty == JobType.PREVIEW:
        return submit_preview_batch_to_api(
            token=token,
            preview_params=params_with_overrides,
            output_dir=output_folder,
            batch_folder_suffix=batch_folder_suffix,
        )
    else:
        return submit_video_batch_to_api(
            token=token,
            job_params=params_with_overrides,
            output_dir=output_folder,
            batch_folder_suffix=batch_folder_suffix,
        )


def _resolve_seed_type(token: str, seed_id: str) -> JobType:
    r_preview = requests.get(
        f"{SERVER_URL}{PREVIEW_QUERY_ENDPOINT}{seed_id}/",
        headers={"Authorization": f"Token {token}"},
    )
    if r_preview.status_code == 200:
        return JobType.PREVIEW
    r_job = requests.get(
        f"{SERVER_URL}{RUN_QUERY_ENDPOINT}{seed_id}/",
        headers={"Authorization": f"Token {token}"},
    )
    if r_job.status_code == 200:
        return JobType.VIDEO
    else:
        raise ValueError(f"{seed_id} is not a valid previous preview or job state")


def submit_rerun_batch_with_overrides_previews(
    token: str,
    base_state_id: str,
    override_params: List[Dict],
    output_folder: str,
    batch_folder_suffix: Optional[str] = None,
) -> Tuple[Batch, Optional[str]]:
    return _submit_rerun_batch_with_overrides(
        job_ty=JobType.PREVIEW,
        seed_ty=_resolve_seed_type(token=token, seed_id=base_state_id),
        token=token,
        base_state_id=base_state_id,
        override_params=override_params,
        output_folder=output_folder,
        batch_folder_suffix=batch_folder_suffix,
    )


def submit_rerun_batch_with_overrides_videos(
    token: str,
    base_state_id: str,
    override_params: List[Dict],
    output_folder: str,
    batch_folder_suffix: Optional[str] = None,
) -> Tuple[Batch, Optional[str]]:
    return _submit_rerun_batch_with_overrides(
        job_ty=JobType.VIDEO,
        seed_ty=_resolve_seed_type(token=token, seed_id=base_state_id),
        token=token,
        base_state_id=base_state_id,
        override_params=override_params,
        output_folder=output_folder,
        batch_folder_suffix=batch_folder_suffix,
    )
