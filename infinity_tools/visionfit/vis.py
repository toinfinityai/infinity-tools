import glob
import json
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.colors as colors
from pathlib import Path
from typing import Any, List, Tuple, Dict
from zipfile import ZipFile
from IPython.display import HTML, display, clear_output
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from plotly.subplots import make_subplots
from pycocotools.coco import COCO
from infinity_tools.common.vis.videos import (
    parse_video_frames,
    stack_videos,
    create_grid_of_videos,
)
from infinity_tools.visionfit.datagen import apply_movenet_to_video, preprocess_frame


def unzip(zipped_folder: str, output_dir: str):
    """Unzips contents of zip file to output directory."""
    with ZipFile(zipped_folder, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def visualize_all_labels(video_job_folder: str) -> str:
    """Visualizes labels for a video job returned from the API.

    Args:
        video_job_folder: Path to the folder containing the video job results.

    Returns:
        Path to resulting animation video.
    """
    output_directory = os.path.join(video_job_folder, "output")
    os.makedirs(output_directory, exist_ok=True)

    video_rgb_path = os.path.join(video_job_folder, "video.rgb.mp4")
    video_json_path = os.path.join(video_job_folder, "video.rgb.json")
    zipped_video_path = os.path.join(video_job_folder, "video.rgb.zip")
    job_json_path = os.path.join(video_job_folder, "job.json")
    job_params = json.load(open(job_json_path))["params"]

    video_rgb_extracted = os.path.join(video_job_folder, "video.rgb")
    os.makedirs(video_rgb_extracted, exist_ok=True)
    unzip(zipped_video_path, video_rgb_extracted)

    imgs = parse_video_frames(video_rgb_path)
    fps = int(job_params["frame_rate"])
    rep_count = parse_rep_count_from_json(video_json_path)
    coco = COCO(video_json_path)

    bounding_box_path = create_bounding_boxes_video(os.path.join(output_directory, "bounding_box.mp4"), imgs, fps, coco)
    skeleton_path = create_keypoint_connections_video(os.path.join(output_directory, "skeleton.mp4"), imgs, fps, coco)
    cuboids_path = create_cuboids_video(os.path.join(output_directory, "cuboids.mp4"), imgs, fps, coco)
    _3D_path = create_3D_keypoints_video(
        os.path.join(output_directory, "3D_keypoints.mp4"),
        fps,
        coco,
        150,
        imgs.shape[1],
        imgs.shape[2],
    )
    segmentation_path = create_segmentation_video(
        os.path.join(output_directory, "segmentation.mp4"),
        video_rgb_extracted,
        fps,
        imgs.shape[1],
        imgs.shape[2],
    )

    clear_output()

    row1 = [video_rgb_path, segmentation_path, bounding_box_path]
    label_videos_row_1 = stack_videos(row1, axis=2)
    row2 = [cuboids_path, skeleton_path, _3D_path]
    label_videos_row_2 = stack_videos(row2, axis=2)
    label_grid_path = stack_videos([label_videos_row_1, label_videos_row_2], axis=1)
    ts_path = animate_time_series(
        os.path.join(output_directory, "timeseries.mp4"),
        rep_count,
        fps,
        width_in_pixels=imgs.shape[1] * 2,
        height_in_pixels=imgs.shape[1] * 2,
    )
    return stack_videos([label_grid_path, ts_path], axis=2)


def parse_rep_count_from_json(json_path: str, rep_count_col: str = "rep_count_from_start") -> List[float]:
    return [x[rep_count_col] for x in json.load(open(json_path))["images"]]


def animate_time_series(
    output_path: str,
    y_axis: List[float],
    fps: int,
    dpi: int = 150,
    width_in_pixels: int = 300,
    height_in_pixels: int = 300,
    title: str = "Rep Count",
    xlabel: str = "Time [s]",
) -> str:
    """Generates time series animation.

    Args:
        output_path: Filename to save output video.
        y_axis: The time series to animate.
        fps: Frame rate of output video.
        width_in_pixels: Width of output video.
        height_in_pixels: Height of output video.
        dpi: DPI of output video.
        title: Title of output video.
        xlabel: X-axis label of output video.

    Returns:
        Path to resulting animation video.
    """

    num_frames = len(y_axis)
    interval = 1 / fps * 1000
    time = np.arange(num_frames) / fps

    fig_height = height_in_pixels / dpi
    fig_width = width_in_pixels / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)
    ax.plot(time, y_axis)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid("on")

    ylim = ax.get_ylim()
    (line,) = ax.plot([0, 0], ylim, color="red")
    fig.tight_layout()

    def update(frame: int, time: npt.NDArray, ylim: Tuple[float, float]):
        line.set_data([time[frame], time[frame]], ylim)

    anim = FuncAnimation(
        fig=fig,
        func=update,
        frames=num_frames,
        fargs=(
            time,
            ylim,
        ),
        interval=interval,
        blit=False,
    )

    anim.save(output_path)
    plt.close()
    return output_path


def create_bounding_boxes_video(output_path: str, imgs: npt.NDArray, fps: int, coco: Any) -> str:
    """Overlays bounding box annotations onto video.

    Args:
        output_path: Path to output video.
        imgs: Frames of the video.
        fps: Frame rate of input video.
        coco: COCO data.

    Returns:
        Path to resulting animation video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    image_dims = (imgs.shape[1], imgs.shape[2])
    out = cv2.VideoWriter(output_path, fourcc, fps, image_dims)

    for img, img_data in zip(imgs, coco.imgs.values()):
        canvas = img.copy()
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "bbox" not in ann:
                continue
            x, y, w, h = tuple(np.array(ann["bbox"]).astype(int))
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=2)
        out.write(canvas)
    out.release()
    return output_path


def get_project_root() -> str:
    """Utility method to get the project root directory."""
    return Path(__file__).parent.parent


def get_armature_connections() -> Dict:
    """Returns armature connections from local json file."""
    project_root = get_project_root()
    armature_json = os.path.join(project_root, "common", "vis", "assets", "armature_connections.json")
    return json.load(open(armature_json, "r"))


def create_keypoint_connections_video(output_path: str, imgs: npt.NDArray, fps: int, coco: Any) -> str:
    """Overlays keypoint connection annotations onto video.

    Args:
        output_path: Path to output video.
        imgs: Frames of the video.
        coco: COCO data.
        job_params: Job parameters from the API.

    Returns:
        Path to resulting animation video.
    """
    kp_connections = get_armature_connections()
    image_dims = (imgs.shape[1], imgs.shape[2])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, image_dims)

    for img, img_data in zip(imgs, coco.imgs.values()):
        canvas = img.copy()
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "armature_keypoints" not in ann:
                continue
            keypoints = ann["armature_keypoints"]
            for parent, child in kp_connections:
                x0 = keypoints[parent]["x"]
                y0 = keypoints[parent]["y"]
                x1 = keypoints[child]["x"]
                y1 = keypoints[child]["y"]
                cv2.line(canvas, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=2)
                cv2.circle(canvas, (x0, y0), radius=4, color=(255, 255, 255), thickness=-1)

        out.write(canvas)
    out.release()
    return output_path


def create_cuboids_video(output_path: str, imgs: npt.NDArray, fps: int, coco: Any) -> str:
    """Overlays cuboid annotations onto video.

    Args:
        output_path: Path to output video
        imgs: Frames of the video
        coco: COCO data
        job_params: Job parameters from the API

    Returns:
        Path to resulting animation video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    image_dims = (imgs.shape[1], imgs.shape[2])
    out = cv2.VideoWriter(output_path, fourcc, fps, image_dims)
    cuboid_edges = [
        [0, 1],
        [0, 4],
        [0, 3],
        [1, 2],
        [2, 3],
        [3, 7],
        [2, 6],
        [1, 5],
        [4, 5],
        [5, 6],
        [6, 7],
        [4, 7],
    ]

    for img, img_data in zip(imgs, coco.imgs.values()):
        canvas = img.copy()
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "cuboid_coordinates" not in ann:
                continue
            cuboid_points = ann["cuboid_coordinates"]
            for edge in cuboid_edges:
                start_point = [cuboid_points[edge[0]]["x"], cuboid_points[edge[0]]["y"]]
                end_point = [cuboid_points[edge[1]]["x"], cuboid_points[edge[1]]["y"]]
                color = tuple([int(255 * x) for x in coco.cats[ann["category_id"]]["color"][::-1]])
                color = (255, 255, 255)
                canvas = cv2.line(
                    canvas,
                    tuple(start_point),
                    tuple(end_point),
                    color=color,
                    thickness=2,
                )
        out.write(canvas)
    out.release()
    return output_path


def create_2D_keypoints_video(output_path: str, imgs: npt.NDArray, fps: int, coco: Any) -> str:
    """Overlays 2D keypoints onto video.

    Args:
        output_path: Path to output video
        imgs: Frames of the video
        coco: COCO data
        job_params: Job parameters from the API

    Returns:
        Path to resulting animation video.
    """

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    image_dims = (imgs.shape[1], imgs.shape[2])
    out = cv2.VideoWriter(output_path, fourcc, fps, image_dims)

    for img, img_data in zip(imgs, coco.imgs.values()):
        canvas = img.copy()
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "armature_keypoints" not in ann:
                continue
            keypoints = ann["armature_keypoints"]
            for keypoint_name, keypoint_info in keypoints.items():
                if keypoint_name == "root":
                    continue
                x, y = keypoint_info["x"], keypoint_info["y"]
                cv2.circle(canvas, (x, y), radius=3, color=(255, 255, 255), thickness=-1)
        out.write(canvas)
    out.release()
    return output_path


def create_3D_keypoints_video(
    output_path: str,
    fps: int,
    coco: Any,
    dpi: int = 150,
    width_in_pixels: int = 200,
    height_in_pixels: int = 200,
) -> str:
    """Visualizes 3D keypoint annotations as video.

    Args:
        output_path: Path to output video
        coco: COCO data
        job_params: Job parameters from the API

    Returns:
        Path to resulting animation video.
    """
    fig_height = height_in_pixels / dpi
    fig_width = width_in_pixels / dpi
    figsize = (fig_width, fig_height)
    fig = plt.figure(dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    global_coords = []
    for img_data in coco.imgs.values():
        img_id = img_data["id"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "armature_keypoints" not in ann:
                continue
            keypoints = ann["armature_keypoints"]
            for _, keypoint_info in keypoints.items():
                global_coords.append(
                    [
                        keypoint_info["x_global"],
                        keypoint_info["y_global"],
                        keypoint_info["z_global"],
                    ]
                )
    global_coords = np.array(global_coords).reshape(-1, len(keypoints), 3)

    def update(num):
        graph._offsets3d = (
            global_coords[num, :, 0],
            global_coords[num, :, 1],
            global_coords[num, :, 2],
        )
        angle = num * 180 / global_coords.shape[0]
        ax.view_init(30, angle)

    graph = ax.scatter(global_coords[0, :, 0], global_coords[0, :, 1], global_coords[0, :, 2])

    ax.set_box_aspect(np.ptp(global_coords, axis=(0, 1)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid("off")

    ani = animation.FuncAnimation(fig, update, global_coords.shape[0], blit=False, interval=1000 / fps)
    ani.save(output_path)
    plt.close()
    return output_path


def create_segmentation_video(output_path: str, folder: str, fps: int, image_width: int, image_height: int) -> str:
    """Creates a video of frame-wise segmentation masks.

    Args:
        output_path: Path to output video.
        folder: Folder which contains the per-frame data.
        job_params: Job parameters from the API.

    Returns:
        Path to resulting animation video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (image_width, image_height))
    iseg_paths = sorted(glob.glob(os.path.join(folder, "*.iseg.png")))
    for img_path in iseg_paths:
        out.write(cv2.imread(img_path))
    out.release()
    return output_path


def summarize_batch_results_as_dataframe(batch_folder: str) -> pd.DataFrame:
    """Compiles job parameters and post-job metadata associated with a batch.

    Args:
        batch_folder: Path to batch folder containing individual job results.

    Returns:
        Dataframe containing batch job parameters and any metadata extracted
        from the resulting job annotations.
    """

    def _convert_to_float(x):
        """Handler for when the job params have been converted to strings."""
        try:
            return x.astype(float)
        except:  # noqa: E722
            return x

    rgb_jsons = glob.glob(os.path.join(batch_folder, "**/video.rgb.json"), recursive=True)

    metadata = []
    for rgb_json in rgb_jsons:
        job_id = os.path.basename(os.path.dirname(rgb_json))
        full_path = os.path.join(batch_folder, job_id)
        job_params_path = os.path.join(os.path.dirname(rgb_json), "job.json")
        job_params = json.load(open(job_params_path))["params"]
        json_data = json.load(open(rgb_json))
        num_frames = len(json_data["images"])
        anns = json_data["annotations"]
        person_cat_id = {e["name"]: e["id"] for e in json_data["categories"]}["person"]
        avg_percent_in_fov = np.mean([ann["percent_in_fov"] for ann in anns if ann["category_id"] == person_cat_id])
        avg_percent_occlusion = np.mean(
            [ann["percent_occlusion"] for ann in anns if ann["category_id"] == person_cat_id]
        )
        video_metadata = {
            "job_path": full_path,
            "job_id": job_id,
            "num_frames": num_frames,
            "avg_percent_in_fov": avg_percent_in_fov,
            "avg_percent_occlusion": avg_percent_occlusion,
        }
        metadata.append({**video_metadata, **job_params})
    df = pd.DataFrame(metadata)
    df = df.apply(_convert_to_float, axis=0)
    return df


def visualize_batch_results(batch_folder: str):
    """Generates histograms of job parameters and extracted metadata for a batch."""

    def _convert_to_float(x):
        """Handler for when the job params have been converted to strings."""
        try:
            return x.astype(float)
        except:  # noqa: E722
            return x

    df = summarize_batch_results_as_dataframe(batch_folder)
    columns_to_keep = [column for column in df.columns if column not in ["job_path", "job_id", "state"]]
    df = df[columns_to_keep]
    color = colors.qualitative.Plotly[0]
    df = df.apply(_convert_to_float, axis=0)

    num_per_row = 4
    row_num = len(df.columns) // num_per_row + 1
    fig = make_subplots(rows=row_num, cols=num_per_row, subplot_titles=df.columns)

    for i, col_name in enumerate(df.columns):
        row = i // num_per_row + 1
        col = i % num_per_row + 1
        if row * col > len(df.columns):
            break

        subfig = go.Figure(data=[go.Histogram(x=df[col_name], name=col_name, nbinsx=10, marker=dict(color=color))])
        fig.add_trace(subfig.data[0], row=row, col=col)
        fig.update_layout(
            template="plotly_white",
            title=f"Summary of Batch | {Path(batch_folder).stem}",
            height=200 * row_num,
            width=300 * num_per_row,
            bargap=0.1,
            showlegend=False,
        )

    fig.show("svg")


def display_batch_results(batch_folder: str) -> None:
    """Displays batch results as dataframe."""
    metadata = summarize_batch_results_as_dataframe(batch_folder)
    display(HTML(metadata.to_html()))


def crop_pad_video(video_path: str, image_width: int = 192, image_height: int = 192) -> str:
    """Crops and pads a video to a standard size."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = video_path.replace(".mp4", "_cropped.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (image_width, image_height))
    video_frames = parse_video_frames(video_path)
    video_frames = [preprocess_frame(frame).numpy().squeeze().astype(np.uint8) for frame in video_frames]
    for frame in video_frames:
        out.write(frame)
    out.release()
    return output_path


def overlay_movenet_on_video(video_path: str, output_path: str):
    """Applies MoveNet to video and returns rendered video path."""
    cropped_video = crop_pad_video(video_path)  # pre-crop and pad here to get aspect ratio.
    landmarks_from_video = apply_movenet_to_video(video_path)
    return visualize_landmarks(cropped_video, landmarks_from_video, output_path)


def view_batch_folder_with_movenet_overlay(batch_folder: str, num_per_row: int = 4, limit_videos: int = 12) -> str:
    """Visualizes a batch folder with movenet overlay."""
    video_paths = glob.glob(os.path.join(batch_folder, "**/*.rgb.mp4"))[:limit_videos]
    output_paths = [path.replace(".rgb.mp4", ".rgb_movenet.mp4") for path in video_paths]
    movenet_paths = [overlay_movenet_on_video(path, output) for path, output in list(zip(video_paths, output_paths))]
    return create_grid_of_videos(movenet_paths, num_per_row=num_per_row)


def visualize_landmarks(video_path: str, landmarks: List[npt.NDArray], output_path: str) -> str:
    """Visualizes MoveNet landmarks for a single video.

    Args:
        video_path: Path to video that will be processed.
        landmarks: Numpy array of shape (num_frames x 17 x 4) containing
            predicted MoveNeet keypoints.
        output_path: Path to save rendered video.

    Returns:
        output_path
    """

    KEYPOINT_PARENT_CHILD = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (image_width, image_height))

    frame_idx = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        for parent, child in KEYPOINT_PARENT_CHILD:
            x0 = int(image_width * landmarks[frame_idx][parent][1])
            y0 = int(image_height * landmarks[frame_idx][parent][0])
            x1 = int(image_width * landmarks[frame_idx][child][1])
            y1 = int(image_height * landmarks[frame_idx][child][0])
            cv2.line(image, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=1)
            cv2.circle(image, (x0, y0), radius=2, color=(176, 129, 30), thickness=-1)
        out.write(image)
        frame_idx += 1
    out.release()

    return output_path
