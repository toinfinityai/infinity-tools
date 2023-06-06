import cv2
import imageio
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from IPython.display import HTML, Image, display
import numpy as np
import matplotlib as mpl
from enum import Enum
import re
from dataclasses import dataclass
import requests


def display_parameters(api: Any, token: str):
    """Displays generator parameters returned by API.

    Args:
        api: api module containing the desired server URL and generator name.
        token: User token required for authentication when fetching parameters from API.
    """
    r = requests.get(
        f"{api.SERVER_URL}/api/jobs/{api.GENERATOR}/",
        headers={"Authorization": f"Token {token}"},
    )
    params = r.json()["params"]
    params = {e["name"]: e for e in params}
    df = pd.DataFrame(params).transpose().drop(columns="description").sort_index().drop("state", errors="ignore")
    display(HTML(df.to_html()))


def summarize_job_params(job_params: List[Dict]):
    display(HTML(pd.DataFrame(job_params).describe().to_html()))


class InfinityPlot(Enum):
    PIE = 1
    HISTOGRAM = 2


def assign_plot_type(df: pd.DataFrame, col_name: str) -> InfinityPlot:
    """Assigns a plot type to a column based on the data it contains.

    Args:
        df: DataFrame containing the column.
        col_name: Name of the column.

    Returns:
        InfinityPlot: Plot type for the column.
    """
    if len(df[col_name].unique()) == 1:
        return InfinityPlot.PIE
    else:
        return InfinityPlot.HISTOGRAM


@dataclass(frozen=True)
class HexColor:
    v: str

    def __post_init__(self):
        if re.fullmatch(r"#[A-F0-9]{6}", self.v) is None:
            raise ValueError(f"Hex color {self.v} not in format #RRGGBB")

    def __str__(self) -> str:
        return self.v


def generate_infinity_scale(
    n: int, color1: HexColor = HexColor("#E31B88"), color2: HexColor = HexColor("#337DEC")
) -> List[str]:
    """Creates a linear scale between two Infinity colors.

    Args:
        n (int): Number of colors to generate.
        color1 (HexColor): First color e.g. HexColor(#RRGGBB).
        color2 (HexColor): Second color e.g. HexColor(#RRGGBB).

    Returns:
        list: List of colors.
    """
    START = np.array(mpl.colors.to_rgb(str(color1)))
    END = np.array(mpl.colors.to_rgb(str(color2)))
    convert_to_hex = lambda x: mpl.colors.to_hex((1 - x) * START + x * END)
    colors = [convert_to_hex(i) for i in np.linspace(0, 1, n)]
    return colors


def get_tick_values_and_labels(df: pd.DataFrame, col_name: str, max_xaxis_label_size: int = 7):
    """Gets the tick labels for a column. Plotly requires tick labels to be set after adding to subplots.
    Extracting labels in this function allows for customization of the tick labels.

    Args:
        df: DataFrame containing the column.
        col_name: Name of the column.
        max_size: Maximum size of the tick labels.

    Returns:
        list: List of tick labels.
    """

    def _trim_string(s: str, max_size: int) -> str:
        if len(s) > max_size:
            return s[:max_size] + ".."
        else:
            return s

    df = df[df[col_name].notna()]
    tickvals = list(range(len(df[col_name].unique())))
    ticktext = [_trim_string(d, max_xaxis_label_size) for d in sorted(df[col_name].unique())]

    return tickvals, ticktext


def plot_infinity_histogram(
    df: pd.DataFrame, col_name: str, stratify_var: Optional[str] = None
) -> plotly.graph_objects.Figure:
    """Plots a histogram of the column.

    Args:
        df (pandas.DataFrame): DataFrame containing the column.
        col_name (str): Name of the column.
        stratify_var (str): Name of the column to use for separating the data.

    Returns:
        plotly.graph_objects.Figure: Plotly figure.
    """
    fig = go.Figure()
    df = df[df[col_name].notna()]

    # Validate that the column can be stratified by stratify_var.
    if stratify_var is not None:
        is_stratifiable = ~df[[col_name, stratify_var]].isnull().values.any()

    # Create a stacked histogram of the column.
    if stratify_var is not None and is_stratifiable:
        unique_values = sorted(df[stratify_var].unique())
        infinity_colors = generate_infinity_scale(len(unique_values))
        for i, group in enumerate(unique_values):
            group_df = df[df[stratify_var] == group]
            histogram = go.Histogram(
                x=sorted(group_df[col_name]),
                name=str(group),
                marker=dict(color=infinity_colors[i]),
                showlegend=True if col_name == stratify_var else False,
                legendgroup="group1",
            )
            fig.add_trace(histogram)

    # Create a grouped histogram of the column.
    else:
        histogram = go.Histogram(
            x=df[col_name],
            name="All",
            marker=dict(color="#0097A7"),
            showlegend=True if col_name == stratify_var else False,
            legendgroup="group1",
        )
        fig.add_trace(histogram)

    return fig


def plot_infinity_pie(df: pd.DataFrame, col_name: str, stratify_var: str) -> plotly.graph_objects.Figure:
    """Plots a pie chart of the column.

    Args:
        df (pandas.DataFrame): DataFrame containing the column.
        col_name (str): Name of the column.
        stratify_var (str): Name of the column to use for separating the data.

    Returns:
        plotly.graph_objects.Figure: Plotly figure.
    """
    df = df[df[col_name].notna()]
    labels = df.sort_values(by=col_name)[col_name].unique()
    values = df.sort_values(by=col_name).groupby(col_name).count().values[:, 0]

    if col_name == stratify_var:
        colors = generate_infinity_scale(len(labels))
    else:
        colors = generate_infinity_scale(len(labels), HexColor("#0097A7"), HexColor("#F6F6F6"))

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(width=0.2, color="grey")),
                sort=False,
                showlegend=True if col_name == stratify_var else False,
                legendgroup="group1",
                insidetextorientation="horizontal",
            )
        ]
    )

    # After 5, the pie charts start to get a little crammed with labels.
    if len(labels) > 5:
        fig.update_traces(textposition="inside", textinfo="percent", textfont_size=10)
    else:
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)

    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    return fig


def create_specs(df: pd.DataFrame, num_per_row: int) -> List[List[Dict]]:
    """Produce a list of specs for a grid of Plotly subplots.
    Plotly requires a specs array of arrays to specify pie plot subplots before building.

    Args:
        columns: number of columns in the grid
        num_per_row: number of rows in the grid

    Returns:
        specs: list of lists of subplot specs
    """
    specs = []
    current_row = []
    for i, col in enumerate(df.columns):
        plot_type = assign_plot_type(df, col)
        if plot_type == InfinityPlot.PIE:
            current_row.append({"type": "pie"})
        else:
            current_row.append({})

        # Add filled rows to the specs array.
        if len(current_row) == num_per_row:
            specs.append(current_row)
            current_row = []

        # Fill final row.
        if i == len(df.columns) - 1 and len(current_row) < num_per_row:
            remaining_cols = num_per_row - len(current_row)
            for _ in range(remaining_cols):
                current_row.append({})
            specs.append(current_row)

    return specs


def visualize_job_params(
    job_params: List[Dict],
    num_per_row: int = 4,
    subplot_size: int = 300,
    renderer: str = None,
    stratify_var: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_xaxis_label_size: int = 7,
    plot_title: str = "Job Parameters",
) -> plotly.graph_objects.Figure:
    """Generates histograms of job parameter distributions.

    Args:
        job_params: Job parameters to visualize.
        num_per_row: number of columns in the grid.
        subplot_size: size of the subplots (pixels).
        renderer: Plotly renderer. If None, uses the default interactive renderer.
            Note: If you want plots to persist on Github, set renderer to "png".
        stratify_var: Name of the column to use for separating the data in histograms.
        max_xaxis_label_size: Maximum size of the x-axis labels.
        output_dir: Directory to save the plot. If None, does not save the plot.
        plot_title: Title of the plot.

    Returns:
        fig: Plotly subplots.
    """

    def _convert_to_float(x: Any) -> float:
        """Handler for when numerical job params have been converted to strings."""
        if isinstance(x, bool) or isinstance(x, float):
            return x
        else:
            try:
                return x.astype(float)
            except:  # noqa: E722
                return x

    df = pd.DataFrame(job_params)
    df = df.applymap(_convert_to_float)
    df = df.iloc[:, df.columns != "state"]

    row_num = len(df.columns) // num_per_row + 1
    specs = create_specs(df, num_per_row)
    fig = make_subplots(
        rows=row_num,
        cols=num_per_row,
        subplot_titles=df.columns,
        specs=specs,
        horizontal_spacing=0.05,
        vertical_spacing=0.075,
    )

    for i, col_name in enumerate(df.columns):
        row = i // num_per_row + 1
        col = i % num_per_row + 1

        plot_type = assign_plot_type(df, col_name)
        if plot_type == InfinityPlot.PIE:
            subfig = plot_infinity_pie(df, col_name, stratify_var)
            for data in subfig.data:
                fig.append_trace(data, row=row, col=col)

        elif plot_type == InfinityPlot.HISTOGRAM:
            subfig = plot_infinity_histogram(df, col_name, stratify_var)
            for data in subfig.data:
                fig.append_trace(data, row=row, col=col)
                if isinstance(df[col_name].iloc[0], str):
                    tickvalues, ticklabels = get_tick_values_and_labels(df, col_name, max_xaxis_label_size)
                    fig.update_xaxes(
                        tickmode="array", tickangle=45, tickvals=tickvalues, ticktext=ticklabels, row=row, col=col
                    )

        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    fig.update_layout(
        template="plotly_white",
        title=plot_title,
        height=subplot_size * row_num,
        width=subplot_size * num_per_row + subplot_size,
        barmode="stack",
        font=dict(size=12),
        legend_title_text=stratify_var,
    )

    fig.update_annotations(font_size=16)
    if output_dir is not None:
        fig.write_image(output_dir)
    fig.show(renderer)


def display_image(image_path: str, display_width: int = 600):
    display(Image(data=open(image_path, "rb").read(), format="png", width=display_width))


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
    imageio.mimsave(output_path, frames, format="GIF", duration=1 / fps * downsample_frames)
    display_image(output_path, display_width)
