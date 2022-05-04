import os
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

import infinity_tools.common.api as ca
from infinity_tools.common.api import Batch

SERVER_URL = "https://api.toinfinity.ai"
GENERATOR = "visionfit"
RUN_ENDPOINT = "/api/jobs/run/"
RUN_QUERY_ENDPOINT = "/api/job_runs/"
PREVIEW_ENDPOINT = "/api/jobs/preview/"
PREVIEW_QUERY_ENDPOINT = "/api/job_previews/"

SCENE_LST = [
    "GYM_1",
    "BEDROOM_2",
    "BEDROOM_4",
    "BEDROOM_5",
]
EXERCISE_LST = [
    "LEG_RAISE",
    "SUPERMAN",
    "ARM_RAISE",
    "BACK_SQUAT",
    "BICEP_CURL",
    "BENT_OVER_TRICEP_KICKBACK_RIGHT",
    "BURPEE",
    "CRUNCH",
    "DEADLIFT",
    "EXPLOSIVE_PUSH_UP",
    "PUSH_PRESS_LEFT",
    "PUSH_UP",
    "PRESS_LEFT",
    "SPLIT_SQUAT_RIGHT",
    "SIT_UP",
    "UPPERCUT_LEFT",
    "V_UP",
]
GENDER_LST = ["MALE", "FEMALE"]
FRAME_RATE_LST = [24, 12, 8, 6]
IMAGE_DIM_LST = [256, 512]
PARAMS_WITH_DESCR = [
    (
        "scene",
        "Background environment/scene",
        "None",
        "['GYM_1', 'BEDROOM_2', 'BEDROOM_4', 'BEDROOM_5']",
    ),
    (
        "exercise",
        "Exercise animation",
        "None",
        "['LEG_RAISE', 'SUPERMAN', 'ARM_RAISE', 'BACK_SQUAT', 'BICEP_CURL', 'BENT_OVER_TRICEP_KICKBACK_RIGHT', 'BURPEE', 'CRUNCH', 'DEADLIFT', 'EXPLOSIVE_PUSH_UP', 'PUSH_PRESS_LEFT', 'PUSH_UP', 'PRESS_LEFT', 'SPLIT_SQUAT_RIGHT', 'SIT_UP', 'UPPERCUT_LEFT', 'V_UP']",
    ),
    ("gender", "Avatar gender", "MALE", "['MALE', 'FEMALE']"),
    ("num_reps", "Number of base exercise animation repetitions", "1", "1 to 10"),
    ("seconds_per_rep", "Length of baseline rep in seconds", "1.0", "1.0 to 3.0"),
    (
        "max_rel_speed_change",
        "Maximum speed change in reps, relative to the baseline speed; expressed as a fraction between 0 and 1",
        "0.0",
        "0.0 to 1.0",
    ),
    (
        "trim_start_frac",
        "Fraction of seed animation (from start to midpoint) to truncate at the start",
        "0.0",
        "0.0 to 1.0",
    ),
    (
        "trim_end_frac",
        "Fraction of seed animation (from start to midpoint) to truncate at the end",
        "0.0",
        "0.0 to 1.0",
    ),
    (
        "kinematic_noise_factor",
        "Scalar factor used to change the default kinematic noise added in generated animations",
        "1.0",
        "0.0 to 2.0",
    ),
    ("camera_height", "Height of viewing camera", "0.75", "0.1 to 2.75"),
    (
        "relative_camera_yaw_deg",
        "Camera yaw in degrees where 0 is directly facing the avatar",
        "0.0",
        "-45.0 to 45.0",
    ),
    (
        "relative_camera_pitch_deg",
        "Camera pitch in degrees where 0 is directly facing the avatar",
        "0.0",
        "-45.0 to 45.0",
    ),
    ("lighting_power", "Luminosity of the scene", "100.0", "0.0 to 2000.0"),
    (
        "relative_avatar_angle_deg",
        "Avatar rotation in the global XY plane, in degrees, where 0 is directly facing the camera",
        "0.0",
        "-180.0 to 180.0",
    ),
    ("frame_rate", "Output video frame rate", "24", "['24', '12', '8', '6']"),
    ("image_width", "Output image/video width in pixels", "256", "128 to 512"),
    ("image_height", "Output image/video height in pixels", "256", "128 to 512"),
]


class JobType(Enum):
    PREVIEW = auto()
    VIDEO = auto()


def sample_input(
    scene: Optional[str] = None,
    exercise: Optional[str] = None,
    gender: Optional[str] = None,
    num_reps: Optional[int] = None,
    seconds_per_rep: Optional[float] = None,
    max_rel_speed_change: Optional[float] = None,
    trim_start_frac: Optional[float] = None,
    trim_end_frac: Optional[float] = None,
    kinematic_noise_factor: Optional[float] = None,
    camera_height: Optional[float] = None,
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
        scene = str(np.random.choice(SCENE_LST))
    else:
        assert scene in SCENE_LST
    if exercise is None:
        exercise = str(np.random.choice(EXERCISE_LST))
    else:
        assert exercise in EXERCISE_LST
    if gender is None:
        gender = str(np.random.choice(GENDER_LST))
    else:
        assert gender in GENDER_LST
    if num_reps is None:
        num_reps = int(np.random.randint(1, 10))
    else:
        assert num_reps >= 1 and num_reps <= 10
    if seconds_per_rep is None:
        seconds_per_rep = float(np.random.uniform(1.0, 3.0))
    else:
        assert seconds_per_rep >= 1.0 and seconds_per_rep <= 3.0
    if max_rel_speed_change is None:
        max_rel_speed_change = float(np.random.uniform(0.0, 0.3))
    else:
        assert max_rel_speed_change >= 0.0 and max_rel_speed_change <= 1.0
    if trim_start_frac is None:
        trim_start_frac = float(np.random.uniform(0.0, 0.1))
    else:
        assert trim_start_frac >= 0.0 and trim_start_frac <= 1.0
    if trim_end_frac is None:
        trim_end_frac = float(np.random.uniform(0.0, 0.1))
    else:
        assert trim_start_frac >= 0.0 and trim_start_frac <= 1.0
    if kinematic_noise_factor is None:
        kinematic_noise_factor = float(np.random.uniform(0.0, 1.0))
    else:
        assert kinematic_noise_factor >= 0.0 and kinematic_noise_factor <= 2.0
    if camera_height is None:
        camera_height = float(np.random.uniform(0.1, 2.75))
    else:
        assert camera_height >= 0.1 and camera_height <= 2.75
    if relative_camera_yaw_deg is None:
        relative_camera_yaw_deg = float(np.random.uniform(-15.0, 15.0))
    else:
        assert relative_camera_yaw_deg >= -45.0 and relative_camera_yaw_deg <= 45.0
    if relative_camera_pitch_deg is None:
        relative_camera_pitch_deg = float(np.random.uniform(-10.0, 10.0))
    else:
        assert relative_camera_pitch_deg >= -45.0 and relative_camera_pitch_deg <= 45.0
    if lighting_power is None:
        lighting_power = float(np.random.uniform(10.0, 1000.0))
    else:
        assert lighting_power >= 0.0 and lighting_power <= 2_000.0
    if relative_avatar_angle_deg is None:
        relative_avatar_angle_deg = float(np.random.uniform(-180.0, 180.0))
    else:
        assert (
            relative_avatar_angle_deg >= -180.0 and relative_avatar_angle_deg <= 180.0
        )
    if frame_rate is None:
        frame_rate = int(np.random.choice(FRAME_RATE_LST))
    else:
        assert frame_rate in FRAME_RATE_LST
    if image_width is None and image_height is None:
        square_size = int(np.random.choice(IMAGE_DIM_LST))
        image_width = square_size
        image_height = square_size
    elif image_width is None:
        assert image_height >= 128 and image_height <= 512
        image_width = image_height
    elif image_height is None:
        assert image_width >= 128 and image_width <= 512
        image_height = image_width
    else:
        assert image_height >= 128 and image_height <= 512
        assert image_width >= 128 and image_width <= 512

    params_dict = {
        "scene": scene,
        "exercise": exercise,
        "gender": gender,
        "num_reps": num_reps,
        "seconds_per_rep": seconds_per_rep,
        "max_rel_speed_change": max_rel_speed_change,
        "trim_start_frac": trim_start_frac,
        "trim_end_frac": trim_end_frac,
        "kinematic_noise_factor": kinematic_noise_factor,
        "camera_height": camera_height,
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


def download_completed_previews(
    completed_previews: List[ca.CompletedJob], output_dir: str
) -> List[str]:
    return ca.download_completed_jobs(
        completed_jobs=completed_previews, output_dir=output_dir
    )


def download_completed_videos(
    completed_jobs: List[ca.CompletedJob], output_dir: str
) -> List[str]:
    return ca.download_completed_jobs(
        completed_jobs=completed_jobs, output_dir=output_dir
    )


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
    assert seed_ty in {JobType.PREVIEW, JobType.VIDEO}

    ALL_PARAM_KEYS = [T[0] for T in PARAMS_WITH_DESCR]
    for override_dict in override_params:
        assert all([k in ALL_PARAM_KEYS for k in override_dict.keys()])

    params_with_overrides = []
    for seed in base_state_ids:
        if seed_ty == JobType.PREVIEW:
            original_params = fetch_preview_params_by_id(token=token, preview_id=seed)
        else:
            original_params = fetch_job_params_by_id(token=token, job_id=seed)

        params_with_overrides.extend(
            [{**original_params, **op, **{"state": seed}} for op in override_params]
        )

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
    assert job_ty in {JobType.PREVIEW, JobType.VIDEO}
    assert seed_ty in {JobType.PREVIEW, JobType.VIDEO}

    ALL_PARAM_KEYS = [T[0] for T in PARAMS_WITH_DESCR]
    for override_dict in override_params:
        assert all([k in ALL_PARAM_KEYS for k in override_dict.keys()])

    if seed_ty == JobType.PREVIEW:
        original_params = fetch_preview_params_by_id(
            token=token, preview_id=base_state_id
        )
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
