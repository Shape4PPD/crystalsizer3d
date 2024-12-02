from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from torch import Tensor

from crystalsizer3d.util.utils import to_numpy, to_rgb

dpi = plt.rcParams['figure.dpi']


def threshold_img(X: np.ndarray, threshold: float) -> np.ndarray:
    X_thresh = X.copy()
    X_thresh[X < threshold] = 0
    return X_thresh


def save_img(X: np.ndarray | Tensor, lbl: str, save_dir: Path):
    if isinstance(X, Tensor):
        X = to_numpy(X)
    X = X.squeeze()
    if X.ndim == 3 and X.shape[0] == 3:
        X = X.transpose(1, 2, 0)
    img = (X * 255).astype(np.uint8)
    Image.fromarray(img).save(save_dir / f'{lbl}.png')


def save_img_with_keypoint_overlay(
        X: np.ndarray | Tensor,
        X_kp: np.ndarray | Tensor,
        lbl: str,
        lbl2: str,
        save_dir: Path,
        alpha_max: float = 0.9
):
    if isinstance(X, Tensor):
        X = to_numpy(X)
    if isinstance(X_kp, Tensor):
        X_kp = to_numpy(X_kp)
    X = X.squeeze()
    if X.ndim == 3 and X.shape[0] == 3:
        X = X.transpose(1, 2, 0)
    fig_size = (X.shape[1] / dpi, X.shape[0] / dpi)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(X)
    alpha = X_kp.copy()
    # alpha = alpha**2
    alpha = alpha / alpha.max() * alpha_max
    ax.imshow(X_kp, cmap='hot', alpha=alpha)
    ax.set_axis_off()
    fig.savefig(
        save_dir / f'{lbl}_overlaid_kp_from_{lbl2}.png',
        bbox_inches='tight', pad_inches=0
    )
    plt.close(fig)


def save_img_with_keypoint_markers(
        X: np.ndarray | Tensor,
        coords: np.ndarray | Tensor,
        lbl: str,
        save_dir: Path,
        marker_type='x',
        suffix: str = '_markers'
):
    if isinstance(X, Tensor):
        X = to_numpy(X)
    if isinstance(coords, Tensor):
        coords = to_numpy(coords)
    X = X.squeeze()
    if X.ndim == 3 and X.shape[0] == 3:
        X = X.transpose(1, 2, 0)
    marker_size = max(5, min(500, X.shape[0] / 5))
    fig_size = (X.shape[1] / dpi, X.shape[0] / dpi)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(X)
    if marker_type == 'x':
        ax.scatter(coords[:, 0], coords[:, 1], marker='x',
                   c='r', linewidths=1, s=marker_size, alpha=0.7)
    else:
        ax.scatter(coords[:, 0], coords[:, 1],
                   facecolors=(1, 0, 0, 0.3),
                   edgecolors=(1, 0, 0, 0.6),
                   linewidths=0.5, s=marker_size * 0.8)
        ax.scatter(coords[:, 0], coords[:, 1], marker='x',
                   c='r', s=1, alpha=0.5)  # Red dots in the centre of each marker
    ax.set_axis_off()
    fig.savefig(
        save_dir / f'{lbl}{suffix}.png',
        bbox_inches='tight', pad_inches=0
    )
    plt.close(fig)


def save_img_with_keypoint_markers2(
        X: np.ndarray | Tensor,
        coords: np.ndarray | Tensor,
        keypoint_radius: int = 30,
        fill_colour: str = 'lightgreen',
        outline_colour: str = 'darkgreen',
        lbl: str = None,
        save_dir: Path = None,
        suffix: str = '_markers',
        **kwargs
) -> Image.Image:
    if isinstance(X, Tensor):
        X = to_numpy(X)
    if isinstance(coords, Tensor):
        coords = to_numpy(coords)
    X = X.squeeze()
    if X.ndim == 3 and X.shape[0] == 3:
        X = X.transpose(1, 2, 0)

    kp_fill_colour = tuple((np.array(to_rgb(fill_colour) + (0.3,)) * 255).astype(np.uint8).tolist())
    kp_outline_colour = tuple((np.array(to_rgb(outline_colour) + (1,)) * 255).astype(np.uint8).tolist())
    if X.dtype == np.uint8:
        img = Image.fromarray(X)
    else:
        img = Image.fromarray((X * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img, 'RGBA')

    # Add the keypoints
    for (x, y) in coords:
        draw.circle((x, y), keypoint_radius, fill=kp_fill_colour, outline=kp_outline_colour,
                    width=keypoint_radius // 6)

    if save_dir:
        img.save(save_dir / f'{lbl}{suffix}.png')

    return img


def save_img_grid(
        X: np.ndarray | Tensor,
        lbl: str,
        save_dir: Path,
        coords: List[np.ndarray | Tensor] | None = None,
        marker_type='o'
):
    """
    Save a grid of images with keypoints overlaid.
    """
    if X.ndim == 3:
        X = X[:, None]  # Add in a channel dimension
    assert X.ndim == 4 and X.shape[1] in [1, 3], 'Images must be (B, [C,] H, W).'
    if isinstance(X, Tensor):
        X = to_numpy(X)
    if X.shape[1] in [1, 3]:
        X = X.transpose(0, 2, 3, 1)
    if coords is not None:
        assert len(coords) == len(X), 'Number of images and coordinates must match.'
        if isinstance(coords[0], Tensor):
            coords = [to_numpy(c) for c in coords]
    else:
        coords = [None] * len(X)

    n_rows = int(np.ceil(np.sqrt(len(coords))))
    n_cols = int(np.ceil(len(coords) / n_rows))
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(
        n_rows, n_cols,
        top=0.99, bottom=0.01, right=0.99, left=0.01,
        hspace=0.02, wspace=0.02
    )

    for i, (X_i, coords_i) in enumerate(zip(X, coords)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(X_i)
        if coords_i is not None:
            if marker_type == 'x':
                ax.scatter(*coords_i.T, marker='x', c='r', linewidths=1, s=75, alpha=0.7)
            else:
                ax.scatter(*coords_i.T, facecolors=(1, 0, 0, 0.3), edgecolors=(1, 0, 0, 0.6),
                           linewidths=0.5, s=150)
                ax.scatter(*coords_i.T, marker='x', c='r', s=1, alpha=0.5)
        ax.set_axis_off()

    fig.savefig(save_dir / f'{lbl}.png')
    plt.close(fig)
