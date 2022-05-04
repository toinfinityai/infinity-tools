import cv2
import imageio
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as colors
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from IPython.display import HTML, Image, display


def display_parameters(parameters: List[Tuple[str]]):
    df = pd.DataFrame(
        parameters,
        columns=["Parameter", "Description", "Default Value", "Allowable Range"],
    )
    display(HTML(df.to_html()))


def summarize_job_params(job_params: List[Dict]):
    display(HTML(pd.DataFrame(job_params).describe().to_html()))


def visualize_job_params(job_params: List[Dict]):
    """Generates histograms of job parameter distributions."""

    def _convert_to_float(x):
        """Handler for when the job params have been converted to strings."""
        try:
            return x.astype(float)
        except:  # noqa: E722
            return x

    color = colors.qualitative.Plotly[0]
    df = pd.DataFrame(job_params)
    df = df.apply(_convert_to_float, axis=0)
    df = df.iloc[:, df.columns != "state"]

    num_per_row = 4
    row_num = len(df.columns) // num_per_row + 1
    fig = make_subplots(rows=row_num, cols=num_per_row, subplot_titles=df.columns)

    for i, col_name in enumerate(df.columns):
        row = i // num_per_row + 1
        col = i % num_per_row + 1
        if row * col > len(df.columns):
            break

        subfig = go.Figure(
            data=[
                go.Histogram(
                    x=df[col_name], name=col_name, nbinsx=10, marker=dict(color=color)
                )
            ]
        )
        fig.add_trace(subfig.data[0], row=row, col=col)
        fig.update_layout(
            template="plotly_white",
            title="Job Distributions",
            height=200 * row_num,
            width=300 * num_per_row,
            bargap=0.1,
            showlegend=False,
        )
    fig.show("svg")


def display_image(image_path: str, display_width: int = 600):
    display(
        Image(data=open(image_path, "rb").read(), format="png", width=display_width)
    )


def display_plotly_as_image(fig: go.Figure, output_path: str, display_width: int = 600):
    """Displays a plotly figure as an image.

    Args:
        fig: Plotly figure to display.
        output_path: Path to save image to.
        display_width: Width of image to display.
    """
    pio.write_image(fig, output_path)
    display_image(output_path, display_width)


def display_video_as_gif(
    video_path: str,
    output_path: Optional[str] = None,
    downsample_resolution: int = 1,
    downsample_frames: int = 1,
    display_width: int = 600,
):
    """Displays a video as a gif, using the display feature native to notebooks.

    This method will also allow for persistent caching of the gif for easy (and performant) viewing.

    Args:
        video_path: Path to the video to display.
        output_path: Path to the output gif.
        downsample_resolution: Downsample the video to this resolution (e.g. downsample_resolution=2 will downsample the video to 1/2 the resolution).
        downsample_frames: Downsample the video to this number of frames (e.g. downsample_frames=2 will downsample the video to 1/2 the number of frames).
        display_width: Width of the image to display.
    """

    if output_path is None:
        output_path = video_path.replace(".mp4", ".gif")

    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_index % downsample_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[::downsample_resolution, ::downsample_resolution, :]
            frames.append(frame)
        frame_index += 1
    imageio.mimsave(
        output_path, frames, format="GIF", duration=1 / fps * downsample_frames
    )
    display_image(output_path, display_width)
