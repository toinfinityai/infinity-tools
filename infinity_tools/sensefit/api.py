import numpy as np
from typing import Dict, List, Optional, Tuple
import infinity_tools.common.api as ca
from infinity_tools.sensefit import vis

SERVER_URL = "https://api.toinfinity.ai"
GENERATOR = "sensefit-v0.2.0"
RUN_ENDPOINT = "/api/jobs/run/"
RUN_QUERY_ENDPOINT = "/api/job_runs/"

EXERCISE_LST = [
    "BICEP_CURL-DUMBBELL",  # legacy
    "HAMMER_CURL-DUMBBELL",  # legacy
    "ARM_RAISE-DUMBBELL",  # legacy
    "OVERHEAD_PRESS-DUMBBELL",  # legacy
    "BEAR_CRAWL-HOLDS",
    "BICEP_CURL-ALTERNATING-DUMBBELL",
    "BICEP_CURL-BARBELL",
    "BIRD_DOG",
    "BRIDGE",
    "BURPEE",
    "CLAMSHELL-LEFT",
    "CLAMSHELL-RIGHT",
    "CRUNCHES",
    "DEADLIFT-DUMBBELL",
    "DONKEY_KICK-LEFT",
    "DONKEY_KICK-RIGHT",
    "DOWNWARD_DOG",
    "LUNGE-CROSSBACK",
    "PRESS-SINGLE_ARM-DUMBBELL-LEFT",
    "PRESS-SINGLE_ARM-DUMBBELL-RIGHT",
    "PUSHUP",
    "PUSHUP-CLOSE_GRIP",
    "PUSHUP-EXPLOSIVE",
    "PUSH_PRESS-SINGLE_ARM-DUMBBELL-LEFT",
    "PUSH_PRESS-SINGLE_ARM-DUMBBELL-RIGHT",
    "SITUP",
    "SPLIT_SQUAT-SINGLE_ARM-DUMBBELL-LEFT",
    "SPLIT_SQUAT-SINGLE_ARM-DUMBBELL-RIGHT",
    "SQUAT-BACK-BARBELL",
    "SQUAT-BODYWEIGHT",
    "SQUAT-GOBLET+SUMO-DUMBBELL",
    "TRICEP_KICKBACK-BENT_OVER+SINGLE_ARM-DUMBBELL-LEFT",
    "TRICEP_KICKBACK-BENT_OVER+SINGLE_ARM-DUMBBELL-RIGHT",
    "UPPERCUT-LEFT",
    "UPPERCUT-RIGHT",
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
        "[" + ", ".join(sorted(EXERCISE_LST)) + "]",
    ),
    ("num_reps", "Number of base exercise animation repetitions", "1", "1 to 20"),
    (
        "watch_location",
        "Wrist side where device will be placed",
        "LEFT",
        "[LEFT, RIGHT]",
    ),
    (
        "crown_orientation",
        "Which side the watch crown should point (from first-person perspective)",
        "RIGHT",
        "[LEFT, RIGHT]",
    ),
    (
        "ref_xy_rotation",
        "Rotation (in XY plane) of reference orientation, in radians.",
        "0.0",
        "0.0 to 6.2831",
    ),
    ("rel_baseline_speed", "Baseline speed of animation, relative to default (natural) speed", "1.0", "0.33 to 3.0"),
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
        "0.0 to 0.9",
    ),
    (
        "trim_end_frac",
        "Fraction of seed animation (from start to midpoint) to truncate at the end",
        "0.0",
        "0.0 to 0.9",
    ),
    (
        "kinematic_noise_factor",
        "Scaling factor used to change the default kinematic noise added in generated animations",
        "1.0",
        "0.0 to 2.0",
    ),
    (
        "wrist_offset_deg",
        "Fixed rotation offset applied to supination/pronation axis of wrists, in degrees. Negative values correspond to supination.",
        "0.0",
        "-90.0 to 90.0",
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
    wrist_offset_deg: Optional[float] = None,
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
        exercise = str(np.random.choice(EXERCISE_LST))
    else:
        if exercise not in EXERCISE_LST:
            raise ValueError(f"`exercise` ({exercise}) not in supported exercise list ({EXERCISE_LST})")

    if num_reps is None:
        num_reps = int(np.random.randint(1, 11))
    else:
        if not (1 <= num_reps <= 20):
            raise ValueError(f"`num_reps` ({num_reps}) must in range [1, 20]")

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
