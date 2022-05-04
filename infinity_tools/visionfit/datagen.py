import glob
import itertools
import json
import os
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_hub as hub
from collections import OrderedDict
from typing import Dict, List, Tuple, TypeVar
from infinity_tools.common.ml.datagen import BaseGenerator
from infinity_tools.common.vis.videos import parse_video_frames

movenet = None

MOVENET_KEYPOINT_DICT = OrderedDict(
    {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }
)


class VisionFitGenerator(BaseGenerator):
    def __init__(
        self,
        sequences_X: List[npt.NDArray],
        sequences_y: List[npt.NDArray],
        window_len: int,
        batch_size: int,
        class_weights: Dict[int, float],
    ):
        super().__init__(
            sequences_X, sequences_y, window_len, batch_size, class_weights
        )

    @staticmethod
    def featurize_data(sequences_X: List[npt.NDArray]) -> List[npt.NDArray]:
        """Featurizes data for model training/inference."""
        return preprocess_keypoints(sequences_X)

    @staticmethod
    def load_data(
        video_path: str, json_path: str = None
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Loads X,y data from a single example."""
        sequence_X = apply_movenet_to_video(video_path)
        if json_path:
            json_file = json.load(open(json_path))
            sequence_y = np.array(
                [img["rep_count_from_start"] for img in json_file["images"]]
            ).flatten()
        else:
            sequence_y = None
        return sequence_X, sequence_y

    @staticmethod
    def load_data_from_folders(
        folders: List[str],
    ) -> Tuple[List[npt.NDArray], List[npt.NDArray]]:
        """Loads all VisionFit data from a list of folders."""

        video_paths = []
        for folder in folders:
            video_paths.extend(glob.glob(os.path.join(folder, "*rgb.mp4")))

        json_paths = []
        for folder in folders:
            json_paths.extend(glob.glob(os.path.join(folder, "*.rgb.json")))

        sequences_y = []
        sequences_X = []

        for video_path, json_path in list(zip(video_paths, json_paths)):
            X, y = VisionFitGenerator.load_data(video_path, json_path)
            sequences_X.append(X)
            sequences_y.append(y)

        return sequences_X, sequences_y


def ensure_movenet_is_loaded():
    """Ensures that MoveNet is loaded into a global variable."""
    global movenet
    if movenet is None:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        movenet = module.signatures["serving_default"]


def preprocess_frame(frame: npt.NDArray, input_size: int = 192) -> tf.Tensor:
    """Processes single frame for MoveNet inference."""
    frame = tf.expand_dims(frame, axis=0)
    frame = tf.image.resize_with_pad(frame, input_size, input_size)
    frame = tf.cast(frame, tf.int32)
    return frame


def apply_movenet_to_video(video_path: str) -> npt.NDArray:
    """Performs keypoint estimation on video with MoveNet."""
    ensure_movenet_is_loaded()
    video_frames = parse_video_frames(video_path)
    video_results = []
    for frame in video_frames:
        processed_frame = preprocess_frame(frame)
        outputs = movenet(processed_frame)
        keypoints_with_scores = outputs["output_0"].numpy()
        frame_results = keypoints_with_scores[0, 0, :, :]
        video_results.append(frame_results)
    return np.array(video_results)


def angle_from_three_points(
    a: npt.NDArray, b: npt.NDArray, c: npt.NDArray
) -> npt.NDArray:
    """Calculates angles between three points in 2D space.

    Args:
        a: (Nx2) numpy array of (x,y) coordinates for set of first point
        b: (Nx2) numpy array of (x,y) coordinates for set of second point
        c: (Nx2) numpy array of (x,y) coordinates for set of third point

    Returns:
        Numpy array of angles between input points, in radians.
    """
    ba = a - b
    bc = c - b
    dot_prod = np.sum(ba * bc, axis=1, keepdims=True)
    cosine_angle = dot_prod / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


def preprocess_keypoints(X_input: List[npt.NDArray]) -> List[npt.NDArray]:
    """Featurizes raw keypoint coordinates into angles.

    Featurization is currently implemented by taking a finite subset of landmarks
    and calculating the 2D angles between all possible sets of three keypoints.

    Args:
        X_input: List of (x,y) keypoint arrays of shape (T_i x num_keypoints x 2),
            where T_i is the number of frames for a given sequence.

    Returns:
        List of featurized data arrays.
    """
    left_landmarks = [
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_hip",
        "left_knee",
    ]
    right_landmarks = [
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_hip",
        "right_knee",
    ]
    landmark_order = list(MOVENET_KEYPOINT_DICT.keys())
    landmarks_of_interest = left_landmarks + right_landmarks
    filter_indices = [landmark_order.index(e) for e in landmarks_of_interest]
    left_angle_combs = list(itertools.permutations(left_landmarks, 3))
    right_angle_combs = list(itertools.permutations(right_landmarks, 3))
    angle_comb_indices = [
        [landmarks_of_interest.index(e) for e in angle_comb]
        for angle_comb in (left_angle_combs + right_angle_combs)
    ]
    X_output = []
    for X in X_input:
        keypoints = X[:, filter_indices, :]
        angles = []
        for angle_comb_index in angle_comb_indices:
            a = np.squeeze(keypoints[:, angle_comb_index[0], :])
            b = np.squeeze(keypoints[:, angle_comb_index[1], :])
            c = np.squeeze(keypoints[:, angle_comb_index[2], :])
            angles.append(angle_from_three_points(a, b, c))
        X_output.append(np.hstack(angles))
    return X_output
