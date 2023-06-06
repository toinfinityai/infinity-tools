import os
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import requests

import infinity_tools.common.api as ca
from infinity_tools.common.api import Batch

SERVER_URL = "https://api.toinfinity.ai"
GENERATOR = "visionfit-v0.3.1"
RUN_ENDPOINT = "/api/jobs/run/"
RUN_QUERY_ENDPOINT = "/api/job_runs/"
PREVIEW_ENDPOINT = "/api/jobs/preview/"
PREVIEW_QUERY_ENDPOINT = "/api/job_previews/"

ALL_PARAM_KEYS = []
SCENE_LST = []
EXERCISE_LST = []
GENDER_LST = []
FRAME_RATE_LST = []
NAME_TO_OPTIONS = {
    "scene": SCENE_LST,
    "exercise": EXERCISE_LST,
    "gender": GENDER_LST,
    "frame_rate": FRAME_RATE_LST,
}

IMAGE_DIM_LST = [256, 512]
NUM_IDENTITIES = 25


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
    exercise: Optional[str] = None,
    gender: Optional[str] = None,
    num_reps: Optional[int] = None,
    rel_baseline_speed: Optional[float] = None,
    max_rel_speed_change: Optional[float] = None,
    trim_start_frac: Optional[float] = None,
    trim_end_frac: Optional[float] = None,
    kinematic_noise_factor: Optional[float] = None,
    camera_distance: Optional[float] = None,
    camera_height: Optional[float] = None,
    avatar_identity: Optional[int] = None,
    relative_height: Optional[float] = None,
    relative_weight: Optional[float] = None,
    relative_camera_yaw_deg: Optional[float] = None,
    relative_camera_pitch_deg: Optional[float] = None,
    lighting_power: Optional[float] = None,
    relative_avatar_angle_deg: Optional[float] = None,
    frame_rate: Optional[int] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    state: Optional[str] = None,
) -> Dict:
    if scene is None:
        scene = str(np.random.choice(_get_param_options("scene", token)))
    else:
        if scene not in _get_param_options("scene", token):
            raise ValueError(f"`scene` ({scene}) not in supported scene list ({_get_param_options('scene', token)})")

    if exercise is None:
        exercise = str(np.random.choice(_get_param_options("exercise", token)))
    else:
        if exercise not in _get_param_options("exercise", token):
            raise ValueError(
                f"`exercise` ({exercise}) not in supported exercise list ({_get_param_options('exercise', token)})"
            )

    if gender is None:
        gender = str(np.random.choice(_get_param_options("gender", token)))
    else:
        if gender not in _get_param_options("gender", token):
            raise ValueError(
                f"`gender` ({gender}) not in supported gender list ({_get_param_options('gender', token)})"
            )

    if num_reps is None:
        num_reps = int(np.random.randint(1, 11))
    else:
        if not (1 <= num_reps <= 20):
            raise ValueError(f"`num_reps` ({num_reps}) must be in range [1, 20]")

    if rel_baseline_speed is None:
        rel_baseline_speed = float(np.random.uniform(0.5, 2.0))
    else:
        if not (0.33 <= rel_baseline_speed <= 3.0):
            raise ValueError(f"`rel_baseline_speed` ({rel_baseline_speed}) must be in range [0.33, 3.0]")

    if max_rel_speed_change is None:
        max_rel_speed_change = float(np.random.uniform(0.0, 0.8))
    else:
        if not (0.0 <= max_rel_speed_change <= 1.0):
            raise ValueError(f"`max_rel_speed_change` ({max_rel_speed_change}) must be in range [0.0, 1.0]")

    if trim_start_frac is None:
        trim_start_frac = 0.0
    else:
        if not (0.0 <= trim_start_frac <= 0.9):
            raise ValueError(f"`trim_start_frac` ({trim_start_frac}) must be in range [0.0, 0.9]")

    if trim_end_frac is None:
        trim_end_frac = 0.0
    else:
        if not (0.0 <= trim_end_frac <= 0.9):
            raise ValueError(f"`trim_end_frac` ({trim_end_frac}) must be in range [0.0, 0.9]")

    if trim_start_frac + trim_end_frac > 0.9:
        raise ValueError(
            f"The sum of `trim_start_frac` and `trim_end_frac` ({trim_start_frac + trim_end_frac}) must be <= 0.9"
        )

    if kinematic_noise_factor is None:
        kinematic_noise_factor = float(np.random.uniform(0.0, 1.0))
    else:
        if not (0.0 <= kinematic_noise_factor <= 2.0):
            raise ValueError(f"`kinematic_noise_factor` ({kinematic_noise_factor}) must be in range [0.0, 2.0]")

    if camera_distance is None:
        camera_distance = float(np.random.uniform(2.5, 5.0))
    else:
        if not (1.0 <= camera_distance <= 5.25):
            raise ValueError(f"`camera_distance` ({camera_distance}) must be in range [1.0, 5.25]")

    if camera_height is None:
        camera_height = float(np.random.uniform(0.1, 2.75))
    else:
        if not (0.1 <= camera_height <= 2.75):
            raise ValueError(f"`camera_height` ({camera_height}) must be in range [0.1, 2.75]")

    if avatar_identity is None:
        avatar_identity = int(np.random.randint(NUM_IDENTITIES))
    else:
        if not (0 <= avatar_identity <= (NUM_IDENTITIES - 1)) or not isinstance(avatar_identity, int):
            raise ValueError(f"`avatar_identity` ({avatar_identity}) must be integer in range [0, {NUM_IDENTITIES-1}]")

    if relative_height is None:
        relative_height = float(np.clip(np.random.normal(), -4.0, 4.0))
    else:
        if not (-4.0 <= relative_height <= 4.0):
            raise ValueError(f"`relative_height` ({relative_height}) must be in range [-4.0, 4.0]")

    if relative_weight is None:
        relative_weight = float(np.clip(np.random.normal(), -4.0, 4.0))
    else:
        if not (-4.0 <= relative_weight <= 4.0):
            raise ValueError(f"`relative_weight` ({relative_weight}) must be in range [-4.0, 4.0]")

    if relative_camera_yaw_deg is None:
        relative_camera_yaw_deg = float(np.random.uniform(-15.0, 15.0))
    else:
        if not (-45.0 <= relative_camera_yaw_deg <= 45.0):
            raise ValueError(f"`relative_camera_yaw_deg` ({relative_camera_yaw_deg}) must be in range [-45.0, 45.0]")

    if relative_camera_pitch_deg is None:
        relative_camera_pitch_deg = float(np.random.uniform(-10.0, 10.0))
    else:
        if not (-45.0 <= relative_camera_pitch_deg <= 45.0):
            raise ValueError(
                f"`relative_camera_pitch_deg` ({relative_camera_pitch_deg}) must be in range [-45.0, 45.0]"
            )

    if lighting_power is None:
        lighting_power = float(np.random.uniform(10.0, 1000.0))
    else:
        if not (0.0 <= lighting_power <= 2_000.0):
            raise ValueError(f"`lighting_power` ({lighting_power}) must be in range [0.0, 2000.0]")

    if relative_avatar_angle_deg is None:
        relative_avatar_angle_deg = float(np.random.uniform(-180.0, 180.0))
    else:
        if not (-180.0 <= relative_avatar_angle_deg <= 180.0):
            raise ValueError(
                f"`relative_avatar_angle_deg` ({relative_avatar_angle_deg}) must be in range [-180.0, 180.0]"
            )

    if frame_rate is None:
        frame_rate = int(np.random.choice(_get_param_options("frame_rate", token)))
    else:
        if frame_rate not in _get_param_options("frame_rate", token):
            raise ValueError(
                f"`frame_rate` ({frame_rate}) not in supported frame rate list ({_get_param_options('frame_rate', token)})"
            )

    if image_width is None and image_height is None:
        square_size = int(np.random.choice(IMAGE_DIM_LST))
        image_width = square_size
        image_height = square_size
    elif image_width is None:
        if not (128 <= image_height <= 512):
            raise ValueError(f"`image_height` ({image_height}) must be in range [128, 512]")
        image_width = image_height
    elif image_height is None:
        if not (128 <= image_width <= 512):
            raise ValueError(f"`image_width` ({image_width}) must be in range [128, 512]")
        image_height = image_width
    else:
        if not (128 <= image_height <= 512):
            raise ValueError(f"`image_height` ({image_height}) must be in range [128, 512]")
        if not (128 <= image_width <= 512):
            raise ValueError(f"`image_width` ({image_width}) must be in range [128, 512]")

    params_dict = {
        "scene": scene,
        "exercise": exercise,
        "gender": gender,
        "num_reps": num_reps,
        "rel_baseline_speed": rel_baseline_speed,
        "max_rel_speed_change": max_rel_speed_change,
        "trim_start_frac": trim_start_frac,
        "trim_end_frac": trim_end_frac,
        "kinematic_noise_factor": kinematic_noise_factor,
        "camera_distance": camera_distance,
        "camera_height": camera_height,
        "avatar_identity": avatar_identity,
        "relative_height": relative_height,
        "relative_weight": relative_weight,
        "relative_camera_yaw_deg": relative_camera_yaw_deg,
        "relative_camera_pitch_deg": relative_camera_pitch_deg,
        "lighting_power": lighting_power,
        "relative_avatar_angle_deg": relative_avatar_angle_deg,
        "frame_rate": frame_rate,
        "image_width": image_width,
        "image_height": image_height,
    }

    if state is not None:
        params_dict["state"] = state

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

        params_with_overrides.extend([{**original_params, **op, **{"state": seed}} for op in override_params])

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
    for pdict in params_with_overrides:
        pdict["state"] = base_state_id

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
