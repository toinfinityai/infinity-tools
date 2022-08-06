import requests
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import infinity_tools.common.api as ca
from infinity_tools.sensefit import vis

SERVER_URL = "https://api.toinfinity.ai"
GENERATOR = "sensefit-v0.2.0"
RUN_ENDPOINT = "/api/jobs/run/"
RUN_QUERY_ENDPOINT = "/api/job_runs/"

ALL_PARAM_KEYS = []
EXERCISE_LST = []
WATCH_LOCATION_LST = []
CROWN_ORIENTATION_LST = []
FPS_LST = []
NAME_TO_OPTIONS = {
    "exercise": EXERCISE_LST,
    "watch_location": WATCH_LOCATION_LST,
    "crown_orientation": CROWN_ORIENTATION_LST,
    "frames_per_second": FPS_LST,
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


def sample_non_rep_params(
    token: str,
    watch_location: Optional[str] = None,
    crown_orientation: Optional[str] = None,
    ref_xy_rotation: Optional[float] = None,
    wrist_offset_deg: Optional[float] = None,
    num_random_frames: Optional[int] = None,
    randomize_body_shape: Optional[bool] = None,
    frames_per_second: Optional[int] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Dict:

    if watch_location is None:
        watch_location = str(np.random.choice(_get_param_options("watch_location", token)))
    else:
        if watch_location not in _get_param_options("watch_location", token):
            raise ValueError(
                f"`watch_location` ({watch_location}) not in supported wrist location list ({_get_param_options('watch_location', token)})"
            )

    if crown_orientation is None:
        crown_orientation = str(np.random.choice(_get_param_options("crown_orientation", token)))
    else:
        if crown_orientation not in _get_param_options("crown_orientation", token):
            raise ValueError(
                f"`crown_orientation` ({crown_orientation}) not in supported crown orientation list ({_get_param_options('crown_orientation', token)})"
            )

    if ref_xy_rotation is None:
        ref_xy_rotation = float(np.random.uniform(0.0, 2 * np.pi))
    else:
        if not (0.0 <= ref_xy_rotation <= 2 * np.pi):
            raise ValueError(f"`ref_xy_rotation` ({ref_xy_rotation}) must be in range [0, 2pi]")

    if wrist_offset_deg is None:
        wrist_offset_deg = float(np.random.uniform(-20.0, 20.0))
    else:
        if not (-90.0 <= wrist_offset_deg <= 90.0):
            raise ValueError(f"`wrist_offset_deg` ({wrist_offset_deg}) must be in range [-90.0, 90.0]")

    if num_random_frames is None:
        num_random_frames = int(np.random.randint(10, 501))
    else:
        if not (10 <= num_random_frames <= 500):
            raise ValueError(f"`num_random_frames` ({num_random_frames}) must be in range [10, 500]")

    if randomize_body_shape is None:
        randomize_body_shape = True
    else:
        if not isinstance(randomize_body_shape, bool):
            raise TypeError(f"`randomize_body_shape` ({randomize_body_shape}) must be of type `bool`")

    if frames_per_second is None:
        frames_per_second = int(np.random.choice(_get_param_options("frames_per_second", token)))
    else:
        if frames_per_second not in _get_param_options("frames_per_second", token):
            raise ValueError(
                f"`frames_per_second` ({frames_per_second}) not in supported frames per second list ({_get_param_options('frames_per_second', token)})"
            )

    if image_width is None and image_height is None:
        image_width = 480
        image_height = 480
    elif image_width is None:
        if not (224 <= image_height <= 1024):
            raise ValueError(f"`image_height` ({image_height}) must be in range [224, 1024]")
        image_width = image_height
    elif image_height is None:
        if not (224 <= image_width <= 1024):
            raise ValueError(f"`image_width` ({image_width}) must be in range [224, 1024]")
        image_height = image_width
    else:
        if not (224 <= image_height <= 1024):
            raise ValueError(f"`image_height` ({image_height}) must be in range [224, 1024]")
        if not (224 <= image_width <= 1024):
            raise ValueError(f"`image_width` ({image_width}) must be in range [224, 1024]")

    if random_seed is None:
        random_seed = int(np.random.randint(low=0, high=2**31))
    else:
        if not (0 <= random_seed <= (2**31 - 1)):
            raise ValueError(f"`random_seed` ({random_seed}) must be in range [0, 2^31 - 1]")

    params_dict = {
        "watch_location": watch_location,
        "crown_orientation": crown_orientation,
        "ref_xy_rotation": ref_xy_rotation,
        "wrist_offset_deg": wrist_offset_deg,
        "use_random_motion": True,
        "num_random_frames": num_random_frames,
        "randomize_body_shape": randomize_body_shape,
        "frames_per_second": frames_per_second,
        "image_width": image_width,
        "image_height": image_height,
        "random_seed": random_seed,
    }

    return params_dict


def sample_rep_params(
    token: str,
    exercise: Optional[str] = None,
    num_reps: Optional[int] = None,
    watch_location: Optional[str] = None,
    crown_orientation: Optional[str] = None,
    ref_xy_rotation: Optional[float] = None,
    rel_baseline_speed: Optional[float] = None,
    max_rel_speed_change: Optional[float] = None,
    trim_start_frac: Optional[float] = None,
    trim_end_frac: Optional[float] = None,
    kinematic_noise_factor: Optional[float] = None,
    wrist_offset_deg: Optional[float] = None,
    randomize_body_shape: Optional[bool] = None,
    frames_per_second: Optional[int] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Dict:

    if exercise is None:
        exercise = str(np.random.choice(_get_param_options("exercise", token)))
    else:
        if exercise not in _get_param_options("exercise", token):
            raise ValueError(
                f"`exercise` ({exercise}) not in supported exercise list ({_get_param_options('exercise', token)})"
            )

    if num_reps is None:
        num_reps = int(np.random.randint(1, 11))
    else:
        if not (1 <= num_reps <= 20):
            raise ValueError(f"`num_reps` ({num_reps}) must in range [1, 20]")

    if watch_location is None:
        watch_location = str(np.random.choice(_get_param_options("watch_location", token)))
    else:
        if watch_location not in _get_param_options("watch_location", token):
            raise ValueError(
                f"`watch_location` ({watch_location}) not in supported wrist location list ({_get_param_options('watch_location', token)})"
            )

    if crown_orientation is None:
        crown_orientation = str(np.random.choice(_get_param_options("crown_orientation", token)))
    else:
        if crown_orientation not in _get_param_options("crown_orientation", token):
            raise ValueError(
                f"`crown_orientation` ({crown_orientation}) not in supported crown orientation list ({_get_param_options('crown_orientation', token)})"
            )

    if ref_xy_rotation is None:
        ref_xy_rotation = float(np.random.uniform(0.0, 2 * np.pi))
    else:
        if not (0.0 <= ref_xy_rotation <= 2 * np.pi):
            raise ValueError(f"`ref_xy_rotation` ({ref_xy_rotation}) must be in range [0, 2pi]")

    if rel_baseline_speed is None:
        rel_baseline_speed = float(np.random.uniform(0.5, 2.0))
    else:
        if not (0.33 <= rel_baseline_speed <= 3.0):
            raise ValueError(f"`rel_baseline_speed` ({rel_baseline_speed}) must be in range [0.33, 3.0]")

    if max_rel_speed_change is None:
        max_rel_speed_change = float(np.random.uniform(0.0, 0.5))
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

    if wrist_offset_deg is None:
        wrist_offset_deg = float(np.random.uniform(-20.0, 20.0))
    else:
        if not (-90.0 <= wrist_offset_deg <= 90.0):
            raise ValueError(f"`wrist_offset_deg` ({wrist_offset_deg}) must be in range [-90.0, 90.0]")

    if randomize_body_shape is None:
        randomize_body_shape = True
    else:
        if not isinstance(randomize_body_shape, bool):
            raise TypeError(f"`randomize_body_shape` ({randomize_body_shape}) must be of type `bool`")

    if frames_per_second is None:
        frames_per_second = int(np.random.choice(_get_param_options("frames_per_second", token)))
    else:
        if frames_per_second not in _get_param_options("frames_per_second", token):
            raise ValueError(
                f"`frames_per_second` ({frames_per_second}) not in supported frames per second list ({_get_param_options('frames_per_second', token)})"
            )

    if image_width is None and image_height is None:
        image_width = 480
        image_height = 480
    elif image_width is None:
        if not (224 <= image_height <= 1024):
            raise ValueError(f"`image_height` ({image_height}) must be in range [224, 1024]")
        image_width = image_height
    elif image_height is None:
        if not (224 <= image_width <= 1024):
            raise ValueError(f"`image_width` ({image_width}) must be in range [224, 1024]")
        image_height = image_width
    else:
        if not (224 <= image_height <= 1024):
            raise ValueError(f"`image_height` ({image_height}) must be in range [224, 1024]")
        if not (224 <= image_width <= 1024):
            raise ValueError(f"`image_width` ({image_width}) must be in range [224, 1024]")

    if random_seed is None:
        random_seed = int(np.random.randint(low=0, high=2**31))
    else:
        if not (0 <= random_seed <= (2**31 - 1)):
            raise ValueError(f"`random_seed` ({random_seed}) must be in range [0, 2^31 - 1]")

    params_dict = {
        "exercise": exercise,
        "num_reps": num_reps,
        "watch_location": watch_location,
        "crown_orientation": crown_orientation,
        "ref_xy_rotation": ref_xy_rotation,
        "rel_baseline_speed": rel_baseline_speed,
        "max_rel_speed_change": max_rel_speed_change,
        "trim_start_frac": trim_start_frac,
        "trim_end_frac": trim_end_frac,
        "kinematic_noise_factor": kinematic_noise_factor,
        "wrist_offset_deg": wrist_offset_deg,
        "use_random_motion": False,
        "randomize_body_shape": randomize_body_shape,
        "frames_per_second": frames_per_second,
        "image_width": image_width,
        "image_height": image_height,
        "random_seed": random_seed,
    }

    return params_dict


def submit_batch_to_api(
    token: str,
    job_params: List[Dict],
    output_dir: str,
    batch_folder_suffix: Optional[str] = None,
) -> Tuple[ca.Batch, Optional[str]]:

    return ca.submit_batch_to_api(
        generator=GENERATOR,
        server=SERVER_URL,
        run_endpoint=RUN_ENDPOINT,
        query_endpoint=RUN_QUERY_ENDPOINT,
        token=token,
        output_dir=output_dir,
        job_params=job_params,
        write_submission_status_to_file=True,
        batch_folder_suffix=batch_folder_suffix,
    )


def download_completed_jobs(completed_jobs: List[ca.CompletedJob], output_dir: str) -> List[str]:
    return ca.download_completed_jobs(completed_jobs=completed_jobs, output_dir=output_dir)


def submit_batch_and_visualize(
    token: str,
    job_params: List[Dict],
    output_dir: str,
    batch_folder_suffix: Optional[str] = None,
):

    batch, batch_folder = submit_batch_to_api(
        job_params=job_params,
        token=token,
        output_dir=output_dir,
        batch_folder_suffix=batch_folder_suffix,
    )
    completed_jobs = batch.await_jobs()
    job_folders = download_completed_jobs(completed_jobs, output_dir=batch_folder)
    vis.visualize_job(job_folders[0])
