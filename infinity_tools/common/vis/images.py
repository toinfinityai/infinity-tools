import glob
import os
from math import ceil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.image import imread
from mpl_toolkits.axes_grid1 import ImageGrid


def view_previews(folder_paths: List[str]):
    """Renders all images matching a *.png pattern in a list of folders.

    Args:
        preview_folder_paths: A list of paths to folders containing .png files.
    """
    all_images = []
    for preview_folder_path in folder_paths:
        all_images.extend(get_all_preview_images(preview_folder_path))
    view_list_of_images(all_images)


def view_list_of_images(image_paths: List[str]):
    """Visualizes previews in a dynamic grid

    Args:
        image_paths: A list of images to view in a grid.
    """

    n_ids = len(image_paths)
    fig = plt.figure(figsize=(15.0, 10.0))
    ncols = 5
    nrows = ceil(n_ids / ncols)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        axes_pad=(0.1, 0.3),
        share_all=True,
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    hide_spines_from_all_plt()

    i = 0
    for ax in grid:
        im = imread(f"{image_paths[i]}")
        ax.imshow(im)
        ax.set_title(f"{Path(image_paths[i]).parent.stem[:7]}...")
        i += 1
        if i >= n_ids:
            break
    plt.show()


def hide_spines_from_all_plt():
    """Hides the axis spines from view for all active figures."""
    figures = [x for x in plt._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        for ax in figure.canvas.figure.get_axes():
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["bottom"].set_color("none")
            ax.spines["left"].set_color("none")


def get_all_preview_images(base_folder: str) -> List[str]:
    """Returns a list of all preview images in the given folder

    Args:
        base_folder: Folder containing a subfolder of completed previews returned from the API.

    Returns:
        List of paths to all preview images in the given folder, sorted in chronological order.
    """
    imgs = glob.glob(os.path.join(base_folder, "*.png"))
    imgs.sort(key=os.path.getmtime)
    return imgs
