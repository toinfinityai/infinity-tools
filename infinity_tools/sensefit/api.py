import numpy as np
from typing import Dict, List, Optional, Tuple
import infinity_tools.common.api as ca
from infinity_tools.sensefit import vis

SERVER_URL = "https://api.toinfinity.ai"
GENERATOR = "sensefit"
RUN_ENDPOINT = "/api/jobs/run/"
RUN_QUERY_ENDPOINT = "/api/job_runs/"

EXERCISE_LST = [
    "BICEP_CURL",
    "HAMMER_CURL",
    "ARM_RAISE",
    "OVERHEAD_PRESS",
    "BARBELL_BACK_SQUAT",
    "BARBELL_BICEP_CURL",
    "BENT_OVER_SINGLE_ARM_DUMBBELL_TRICEP_KICKBACK_RIGHT",
    "BURPEE",
    "CRUNCH",
    "EXPLOSIVE_PUSH_UP",
    "ONE_ARM_DUMBBELL_PUSH_PRESS_LEFT",
    "PUSH_UP",
    "SINGLE_ARM_DUMBBELL_PRESS_LEFT",
    "SINGLE_ARM_DUMBBELL_SPLIT_SQUAT_RIGHT",
    "SIT_UP",
    "UPPERCUT_LEFT",
    "V_UP",
]
WRIST_LOCATION_LST = ["LEFT", "RIGHT"]
CROWN_ORIENTATION_LST = ["LEFT", "RIGHT"]
FPS_LST = [20, 30, 40]


PARAMS_WITH_DESCR = [
    (
        "exercise",
        "Exercise animation",
        "None",
        "[" + ", ".join([f"'{e}'" for e in sorted(EXERCISE_LST)]) + "]",
    ),
    ("num_reps", "Number of base exercise animation repetitions", "1", "1 to 10"),
    (
        "watch_location",
        "Wrist side where device will be placed",
        "None",
        "['LEFT', 'RIGHT']",
    ),
    (
        "crown_orientation",
        "Which side the watch crown should point (from first-person perspective)",
        "None",
        "['LEFT', 'RIGHT']",
    ),
    (
        "ref_xy_rotation",
        "Rotation (in XY plane) of reference orientation, in radians.",
        "None",
        "0 to 6.2831",
    ),
    ("seconds_per_rep", "Duration of baseline rep in seconds", "None", "1.0 to 3.0"),
    (
        "max_rel_speed_change",
        "Maximum speed change in reps, relative to the baseline speed; expressed as a fraction between 0 and 1",
        "0",
        "0 to 1",
    ),
    (
        "trim_start_frac",
        "Fraction of seed animation (from start to midpoint) to truncate at the start",
        "0",
        "0 to 1",
    ),
    (
        "trim_end_frac",
        "Fraction of seed animation (from start to midpoint) to truncate at the end",
        "0",
        "0 to 1",
    ),
    (
        "kinematic_noise_factor",
        "Scalar factor used to change the default kinematic noise added in generated animations",
        "1",
        "0 to 1",
    ),
    (
        "randomize_body_shape",
        "If True, SMPLX body shape will be randomized",
        "False",
        "[True, False]",
    ),
    (
        "use_random_motion",
        "If True, random motion will be used for animation, rather than exercise reps",
        "False",
        "[True, False]",
    ),
    (
        "num_random_frames",
        "Number of random frames to export if using random (non-exercise) motion",
        "100",
        "10 to 500",
    ),
    (
        "frames_per_second",
        "Sampling rate of exported time series and video",
        "20",
        "[20, 30, 40]",
    ),
    (
        "image_width",
        "Width dimension of rendered viewport video, in pixels",
        "480",
        "224 to 1024",
    ),
    (
        "image_height",
        "Height dimension of rendered viewport video, in pixels",
        "480",
        "224 to 1024",
    ),
    ("random_seed", "Random seed for reproducibility", "None", "0 to 2147483647"),
]


def sample_non_rep_params(
    watch_location: Optional[str] = None,
    crown_orientation: Optional[str] = None,
    ref_xy_rotation: Optional[float] = None,
    num_random_frames: Optional[int] = None,
    randomize_body_shape: Optional[bool] = None,
    frames_per_second: Optional[int] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Dict:

    if watch_location is None:
        watch_location = str(np.random.choice(WRIST_LOCATION_LST))
    else:
        if watch_location not in WRIST_LOCATION_LST:
            raise ValueError(
                f"`watch_location` ({watch_location}) not in supported wrist location list ({WRIST_LOCATION_LST})"
            )

    if crown_orientation is None:
        crown_orientation = str(np.random.choice(CROWN_ORIENTATION_LST))
    else:
        if crown_orientation not in CROWN_ORIENTATION_LST:
            raise ValueError(
                f"`crown_orientation` ({crown_orientation}) not in supported crown orientation list ({CROWN_ORIENTATION_LST})"
            )

    if ref_xy_rotation is None:
        ref_xy_rotation = float(np.random.uniform(0.0, 2 * np.pi))
    else:
        if not (0.0 <= ref_xy_rotation <= 2 * np.pi):
            raise ValueError(f"`ref_xy_rotation` ({ref_xy_rotation}) must be in range [0, 2pi]")

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
        frames_per_second = int(np.random.choice(FPS_LST))
    else:
        if frames_per_second not in FPS_LST:
            raise ValueError(
                f"`frames_per_second` ({frames_per_second}) not in supported frames per second list ({FPS_LST})"
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
    exercise: Optional[str] = None,
    num_reps: Optional[int] = None,
    watch_location: Optional[str] = None,
    crown_orientation: Optional[str] = None,
    ref_xy_rotation: Optional[float] = None,
    seconds_per_rep: Optional[float] = None,
    max_rel_speed_change: Optional[float] = None,
    trim_start_frac: Optional[float] = None,
    trim_end_frac: Optional[float] = None,
    kinematic_noise_factor: Optional[float] = None,
    randomize_body_shape: Optional[bool] = None,
    frames_per_second: Optional[int] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Dict:

    if exercise is None:
        exercise = str(np.random.choice(EXERCISE_LST))
    else:
        if exercise not in EXERCISE_LST:
            raise ValueError(f"`exercise` ({exercise}) not in supported exercise list ({EXERCISE_LST})")

    if num_reps is None:
        num_reps = int(np.random.randint(1, 11))
    else:
        if not (1 <= num_reps <= 10):
            raise ValueError(f"`num_reps` ({num_reps}) must in range [1, 10]")

    if watch_location is None:
        watch_location = str(np.random.choice(WRIST_LOCATION_LST))
    else:
        if watch_location not in WRIST_LOCATION_LST:
            raise ValueError(
                f"`watch_location` ({watch_location}) not in supported wrist location list ({WRIST_LOCATION_LST})"
            )

    if crown_orientation is None:
        crown_orientation = str(np.random.choice(CROWN_ORIENTATION_LST))
    else:
        if crown_orientation not in CROWN_ORIENTATION_LST:
            raise ValueError(
                f"`crown_orientation` ({crown_orientation}) not in supported crown orientation list ({CROWN_ORIENTATION_LST})"
            )

    if ref_xy_rotation is None:
        ref_xy_rotation = float(np.random.uniform(0.0, 2 * np.pi))
    else:
        if not (0.0 <= ref_xy_rotation <= 2 * np.pi):
            raise ValueError(f"`ref_xy_rotation` ({ref_xy_rotation}) must be in range [0, 2pi]")

    if seconds_per_rep is None:
        seconds_per_rep = float(np.random.uniform(1.0, 3.0))
    else:
        if not (1.0 <= seconds_per_rep <= 3.0):
            raise ValueError(f"`seconds_per_rep` ({seconds_per_rep}) must be in range [1.0, 3.0]")

    if max_rel_speed_change is None:
        max_rel_speed_change = float(np.random.uniform(0.0, 0.5))
    else:
        if not (0.0 <= max_rel_speed_change <= 1.0):
            raise ValueError(f"`max_rel_speed_change` ({max_rel_speed_change}) must be in range [0.0, 1.0]")

    if trim_start_frac is None:
        trim_start_frac = 0.0
    else:
        if not (0.0 <= trim_start_frac <= 1.0):
            raise ValueError(f"`trim_start_frac` ({trim_start_frac}) must be in range [0.0, 1.0]")

    if trim_end_frac is None:
        trim_end_frac = 0.0
    else:
        if not (0.0 <= trim_end_frac <= 1.0):
            raise ValueError(f"`trim_end_frac` ({trim_end_frac}) must be in range [0.0, 1.0]")
    if trim_start_frac + trim_end_frac >= 0.9:
        raise ValueError(
            f"The sum of `trim_start_frac` and `trim_end_frac` ({trim_start_frac + trim_end_frac}) must be < 0.9"
        )

    if kinematic_noise_factor is None:
        kinematic_noise_factor = float(np.random.uniform(0.0, 1.0))
    else:
        if not (0.0 <= kinematic_noise_factor <= 1.0):
            raise ValueError(f"`kinematic_noise_factor` ({kinematic_noise_factor}) must be in range [0.0, 1.0]")

    if randomize_body_shape is None:
        randomize_body_shape = True
    else:
        if not isinstance(randomize_body_shape, bool):
            raise TypeError(f"`randomize_body_shape` ({randomize_body_shape}) must be of type `bool`")

    if frames_per_second is None:
        frames_per_second = int(np.random.choice(FPS_LST))
    else:
        if frames_per_second not in FPS_LST:
            raise ValueError(
                f"`frames_per_second` ({frames_per_second}) not in supported frames per second list ({FPS_LST})"
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
        "seconds_per_rep": seconds_per_rep,
        "max_rel_speed_change": max_rel_speed_change,
        "trim_start_frac": trim_start_frac,
        "trim_end_frac": trim_end_frac,
        "kinematic_noise_factor": kinematic_noise_factor,
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
