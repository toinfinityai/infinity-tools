import os
import uuid
import cv2
import numpy as np
import numpy.typing as npt
from typing import List, Optional


def parse_video_frames(video_path: str, height: Optional[int] = None, width: Optional[int] = None) -> npt.NDArray:
    """Parses video frames from a video.

    Args:
        video_path: Path to the video to parse.
        height: Optional height to resize frames into (preserves aspect ratio).
        width: Optional height to resize frames into (preserves aspect ratio).

    Returns:
        A numpy array containing the video frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if height is not None:
        if width is not None:
            raise ValueError(f"Both `height` and `width` provided")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if height is not None:
            new_w = int(frame.shape[1] * height / frame.shape[0])
            frame = cv2.resize(frame, (new_w, height))
        elif width is not None:
            new_h = int(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, new_h))
        frames.append(frame)
    frames = np.array(frames)
    return frames


def stack_videos(
    paths: List[str],
    clip_to_minimum: bool = True,
    axis: int = 2,
    num_pad_axis: int = 0,
    fixed_size: Optional[int] = None,
    output_path: Optional[str] = None,
) -> str:
    """Performs video stacking frame-by-frame. Capable of vertical and horizontal stacking.

    Args:
        paths: List of videos to concatenate along axis.
        clip_to_minimum: If True, clips to the shortest video. Will fail if False and mismatched.
        axis: Axis to concatenate along (1 = vertical; 2 = horizontal).
        num_pad_axis: Number of frames to pad the axis with.

    Returns:
        A path to the stacked video (embedded, actual)
    """

    if axis not in [1, 2]:
        raise ValueError(f"`axis` ({axis}) must be 1 or 2")

    heights = []
    widths = []
    fps = []
    for path in paths:
        cap = cv2.VideoCapture(path)
        _, frame = cap.read()
        heights.append(frame.shape[0])
        widths.append(frame.shape[1])
        fps.append(int(cap.get(cv2.CAP_PROP_FPS)))

    if not (all(e == fps[0] for e in fps)):
        raise ValueError(f"FPS values are different across videos: {fps}")

    parse_kwargs = {}
    if axis == 1:
        if fixed_size is None:
            if not all(e == widths[0] for e in widths):
                raise ValueError("Not all widths are the same size")
        else:
            parse_kwargs = {"width": fixed_size}
    elif axis == 2:
        if fixed_size is None:
            if not all(e == heights[0] for e in heights):
                raise ValueError("Not all heights are the same size")
        else:
            parse_kwargs = {"height": fixed_size}
    all_frames = tuple([parse_video_frames(path, **parse_kwargs) for path in paths])

    if clip_to_minimum:
        minimum_frames = min([frame.shape[0] for frame in all_frames])
        all_frames = tuple([frame[:minimum_frames, :] for frame in all_frames])

    all_frames = np.concatenate(all_frames, axis=axis)

    if num_pad_axis:
        for i in range(num_pad_axis):
            new_video = (np.ones((len(all_frames), widths[0], heights[0], 3)) * 255).astype(np.uint8)
            all_frames = np.concatenate([all_frames, new_video], axis=axis)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_dims = tuple(all_frames.shape[1:3][::-1])

    if output_path is None:
        output_path = os.path.join(os.path.dirname(paths[0]), f"{uuid.uuid4()}.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps[0], video_dims)
    for frame in all_frames:
        out.write(frame)
    out.release()
    return output_path


def create_grid_of_videos(video_paths: List[str], num_per_row: int) -> str:
    """Creates a grid of videos, and pads rows with blank frames to fill out the grid.

    Args:
        video_paths: List of paths to videos.
        num_per_row: Number of videos per row.

    Returns:
        A path to the stacked video.
    """
    num_videos = len(video_paths)
    video_paths = np.array(video_paths, dtype=object)
    if num_videos % num_per_row == 0:
        num_rows = num_videos // num_per_row
        row_indices = np.arange(0, num_videos).reshape((num_rows, num_per_row))

    else:
        num_rows = num_videos // num_per_row + 1
        num_to_pad = (num_per_row * num_rows) - num_videos
        row_indices = np.arange(0, num_videos + num_to_pad).reshape((num_rows, num_per_row))

    if num_rows == 1:
        return stack_videos(video_paths, axis=2)

    else:
        rows = []
        for row_ind in row_indices:
            if max(row_ind) <= len(video_paths) - 1:
                row = stack_videos(video_paths[row_ind], axis=2)

            else:
                row = stack_videos(video_paths[min(row_ind) :], num_pad_axis=num_to_pad, axis=2)
            rows.append(row)
        return stack_videos(rows, axis=1)


def overlay_repcount_pred(video_path: str, pred_count: List[int], output_path: str, font_scale: int = 3):
    """Overlays predicted rep count value onto original test video.

    Args:
        video_path: Path to real-world test video.
        pred_count: List of the predict rep count value for each video frame.
        output_path: Output path where video will be saved.
    """

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(20)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    num_samples = len(pred_count)
    frame_idx = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image = cv2.putText(
            img=image,
            text=f"Rep count: {pred_count[frame_idx]}",
            org=(20, h - 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=int(font_scale * 2.5),
            lineType=cv2.LINE_AA,
        )
        out.write(image)
        frame_idx += 1
        if frame_idx >= num_samples:
            break
    out.release()
