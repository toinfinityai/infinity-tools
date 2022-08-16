import glob
import json
import os
import pandas as pd
import cv2
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.colors as colors
from pathlib import Path
from typing import Any, Tuple
from zipfile import ZipFile
from IPython.display import HTML, display, clear_output
from plotly.subplots import make_subplots
from pycocotools.coco import COCO
from infinity_tools.common.vis.videos import (
    parse_video_frames,
    stack_videos,
)


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
    image_dims = (imgs.shape[2], imgs.shape[1])
    fps = int(job_params["frame_rate"])
    coco = COCO(video_json_path)

    bounding_box_path = create_bounding_boxes_video(
        os.path.join(output_directory, "bounding_box.mp4"), imgs, fps, coco, image_dims
    )
    segmentation_path = create_segmentation_video(
        os.path.join(output_directory, "segmentation.mp4"),
        video_rgb_extracted,
        fps,
        image_dims,
    )

    clear_output()

    row1 = [video_rgb_path, bounding_box_path, segmentation_path]
    return stack_videos(row1, axis=2)


def create_bounding_boxes_video(
    output_path: str, imgs: npt.NDArray, fps: int, coco: Any, image_dims: Tuple[int, int]
) -> str:
    """Overlays bounding box annotations onto video.

    Args:
        output_path: Path to output video.
        imgs: Frames of the video.
        fps: Frame rate of input video.
        coco: COCO data.
        image_dims: image dimensions (width, height)

    Returns:
        Path to resulting animation video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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


def create_segmentation_video(output_path: str, folder: str, fps: int, image_dims: Tuple[int, int]) -> str:
    """Creates a video of frame-wise segmentation masks.

    Args:
        output_path: Path to output video.
        folder: Folder which contains the per-frame data.
        fps: frame rate of video
        image_dims: image dimensions (width, height)

    Returns:
        Path to resulting animation video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, image_dims)
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
        video_metadata = {
            "job_path": full_path,
            "job_id": job_id,
            "num_frames": num_frames,
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
