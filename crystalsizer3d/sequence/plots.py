from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.gridspec import GridSpec

from crystalsizer3d.crystal import Crystal
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.util.utils import get_crystal_face_groups, init_tensor, smooth_signal, to_rgb

plot_extension = 'png'  # or svg
line_styles = ['--', '-.', ':']
marker_styles = ['o', 'x', 's', '+', 'v', '^', '<', '>', 'd', 'p', 'P', '*', 'h', 'H', '|', '_']


def get_face_group_colours(n_groups: int, cmap: str = 'turbo') -> np.ndarray:
    """Get a set of colours for the face groups."""
    return plt.get_cmap(cmap)(np.linspace(0, 1, n_groups))


def calculate_luminance(rgb):
    """Calculate the luminance of an RGB colour."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def get_colour_variations(
        base_colour: str,
        n_shades: int,
        hue_range: float = 0.15,
        sat_range: float = 0.25,
        val_range: float = 0.25,
        max_luminance: float = 0.6
):
    """
    Generate a set of colour variations around a base colour.
    """
    # Convert base color from string to RGB to HSV
    base_hsv = rgb_to_hsv(to_rgb(base_colour)[:3])

    # Create the value ranges
    hue_min = max(0, base_hsv[0] - hue_range / 2)
    hue_max = min(1, hue_min + hue_range)
    sat_min = max(0, base_hsv[1] - sat_range / 2)
    sat_max = min(1, sat_min + sat_range)
    val_min = max(0, base_hsv[2] - val_range / 2)
    val_max = min(1, val_min + val_range)
    hue_vals = np.linspace(hue_min, hue_max, n_shades)
    sat_vals = np.linspace(sat_min, sat_max, n_shades)
    val_vals = np.linspace(val_min, val_max, n_shades)

    # Adjust hue and saturation to create distinct shades
    variations = []
    for i in range(n_shades):
        variation_hsv = base_hsv.copy()
        variation_hsv[0] = hue_vals[i]
        variation_hsv[1] = sat_vals[i]
        variation_hsv[2] = val_vals[i]

        # Convert to RGB and calculate luminance
        rgb_colour = hsv_to_rgb(variation_hsv)
        luminance = calculate_luminance(rgb_colour)

        # Adjust if luminance is below threshold
        while luminance > max_luminance:
            # Gradually reduce value and boost saturation to improve contrast
            variation_hsv[2] = max(0, variation_hsv[2] - 0.05)  # reduce brightness
            variation_hsv[1] = min(1, variation_hsv[1] + 0.05)  # increase saturation slightly
            rgb_colour = hsv_to_rgb(variation_hsv)
            luminance = calculate_luminance(rgb_colour)

        variations.append(rgb_colour)

    return variations


def get_hkl_label(hkl: np.ndarray, is_group: bool = False) -> str:
    """
    Get a LaTeX label for the hkl indices, replacing negative indices with overlines.
    """
    brackets = ['\{', '\}'] if is_group else ['(', ')']
    return f'${brackets[0]}' + ''.join([f'{mi}' if mi > -1 else f'\\bar{mi * -1}' for mi in hkl]) + f'{brackets[1]}$'


def plot_losses(
        losses_final: List[float],
        losses_all: List[List[float]],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path,
        smoothing_window: int = 11
):
    """
    Plot the losses.
    """
    x_vals = [idx for idx, _ in image_paths]

    # Normalize the indices for the colormap
    norm = plt.Normalize(x_vals[0], x_vals[-1])
    cmap = plt.get_cmap('plasma')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # ScalarMappable needs an array, even if not used here

    fig, axes = plt.subplots(2, figsize=(12, 8))

    # Final losses for each image in the sequence
    ax = axes[0]
    ax.set_title('Final losses')
    ax.grid()
    ax.plot(x_vals, losses_final)
    ax.set_xlabel('Image index')
    ax.set_ylabel('Loss')

    # Loss convergence per image
    ax = axes[1]
    ax.set_title('Loss convergence')
    ax.grid()
    for i, v in enumerate(losses_all):
        ax.plot(smooth_signal(v, window_size=smoothing_window), color=cmap(norm(x_vals[i])), alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Image index', rotation=270, labelpad=15)

    fig.tight_layout()
    plt.savefig(save_dir / f'losses.{plot_extension}')


def plot_face_property_values(
        manager: Manager,
        property_name: str,
        property_values: np.ndarray,
        image_paths: List[Tuple[int, Path]],
        save_dir: Path,
        measurement_idxs: np.ndarray = None,
        measurement_values: np.ndarray = None,
        show_mean: bool = False,
        show_std: bool = False
):
    """
    Plot face distances or areas.
    """
    x_vals = [idx for idx, _ in image_paths]
    groups = get_crystal_face_groups(manager)
    n_groups = len(groups)
    group_colours = get_face_group_colours(n_groups)

    if property_name == 'areas':
        y_label = 'Face area'
    elif property_name == 'distances':
        y_label = 'Distance'
    else:
        raise ValueError(f'Invalid property name: {property_name}')

    # Make a grid of plots showing the values for each face group
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 4))
    gs = GridSpec(
        n_rows, n_cols,
        top=0.95, bottom=0.08, right=0.99, left=0.05,
        hspace=0.3, wspace=0.2
    )
    for i, (group_hkl, group_idxs) in enumerate(groups.items()):
        colour = group_colours[i]
        colour_variants = get_colour_variations(colour, len(group_idxs))
        y = property_values[:, list(group_idxs.values())]
        y_measured = measurement_values[:, list(group_idxs.values())] if measurement_values is not None else None
        y_mean = y.mean(axis=1)
        y_std = y.std(axis=1)
        lbls = [get_hkl_label(hkl) for hkl in list(group_idxs.keys())]

        ax = fig.add_subplot(gs[i])
        ax.set_title(get_hkl_label(group_hkl, is_group=True))

        # Plot the mean +/- std
        if show_mean:
            ax.plot(x_vals, y_mean, c=colour, label='Mean', zorder=1000)
        if show_std:
            ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std, color=colour, alpha=0.1)

        # Plot the individual faces
        for j, (y_j, lbl, colour_j) in enumerate(zip(y.T, lbls, colour_variants)):
            ax.plot(x_vals, y_j, label=lbl, c=colour_j,
                    linestyle=line_styles[j % len(line_styles)], linewidth=0.8, alpha=0.8)
            if y_measured is not None:
                ax.plot(measurement_idxs, y_measured[:, j], label=lbl + ' (manual)', c=colour_j,
                        marker=marker_styles[j % len(marker_styles)], linestyle='none', markersize=5, alpha=0.7)

        ax.set_xlabel('Image index')
        ax.set_ylabel(y_label)
        ax.legend(loc='lower right')
    plt.savefig(save_dir / f'{property_name}_grouped.{plot_extension}')

    # Make a plot showing the mean values all together
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.set_title(f'Mean {property_name}')
    ax.grid()
    for i, (group_hkl, group_idxs) in enumerate(groups.items()):
        colour = group_colours[i]
        y = property_values[:, list(group_idxs.values())]
        y_mean = y.mean(axis=1)
        y_std = y.std(axis=1)
        ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std, color=colour, alpha=0.1)
        ax.plot(x_vals, y_mean, c=colour, label=get_hkl_label(group_hkl, is_group=True))
    ax.set_xlabel('Image index')
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_dir / f'{property_name}_mean.{plot_extension}')


def plot_distances(
        manager: Manager,
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path,
        measurements: Dict[str, np.ndarray] = None
):
    """
    Plot face distances.
    """
    distances = parameters['distances']
    scales = parameters['scale']
    plot_face_property_values(
        manager=manager,
        property_name='distances',
        property_values=distances * scales[:, None],
        image_paths=image_paths,
        save_dir=save_dir,
        measurement_idxs=measurements['idx'] if measurements is not None else None,
        measurement_values=measurements['distances'] * measurements['scale'][:, None]
        if measurements is not None else None
    )


def plot_areas(
        manager: Manager,
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path,
        measurements: Dict[str, np.ndarray] = None
):
    """
    Plot face areas.
    """
    distances = parameters['distances']
    scales = parameters['scale']

    # Get a crystal object
    ds = manager.ds
    cs = ds.csd_proxy.load(ds.dataset_args.crystal_id)
    crystal = Crystal(
        lattice_unit_cell=cs.lattice_unit_cell,
        lattice_angles=cs.lattice_angles,
        miller_indices=ds.miller_indices,
        point_group_symbol=cs.point_group_symbol,
        dtype=torch.float64
    )

    # Calculate the face areas
    areas = np.zeros_like(distances)
    for i in range(len(distances)):
        crystal.build_mesh(distances=init_tensor(distances[i], dtype=torch.float64))
        unscaled_areas = np.array([crystal.areas[tuple(hkl.tolist())] for hkl in crystal.all_miller_indices])
        areas[i] = unscaled_areas * scales[i]**2

    # Calculate the face areas for the manual measurements
    areas_m = None
    if measurements is not None:
        distances_m = measurements['distances']
        scales_m = measurements['scale']
        areas_m = np.zeros_like(distances_m)
        for i in range(len(distances_m)):
            crystal.build_mesh(distances=init_tensor(distances_m[i], dtype=torch.float64))
            unscaled_areas_m = np.array([crystal.areas[tuple(hkl.tolist())] for hkl in crystal.all_miller_indices])
            areas_m[i] = unscaled_areas_m * scales_m[i]**2

    plot_face_property_values(
        manager=manager,
        property_name='areas',
        property_values=areas,
        image_paths=image_paths,
        save_dir=save_dir,
        measurement_idxs=measurements['idx'] if measurements is not None else None,
        measurement_values=areas_m
    )


def plot_origin(
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path,
        measurements: Dict[str, np.ndarray] = None
):
    """
    Plot origin position.
    """
    origins = parameters['origin']
    x_vals = [idx for idx, _ in image_paths]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.set_title('Origin position')
    ax.grid()
    for i in range(3):
        ax.plot(x_vals, origins[:, i], label='xyz'[i])
    if measurements is not None:
        origins = measurements['origin']
        x_vals = measurements['idx']
        for i in range(3):
            ax.plot(x_vals, origins[:, i], label='xyz'[i] + ' (manual)', linestyle='none',
                    marker=marker_styles[i], markersize=5, alpha=0.7)
    ax.set_xlabel('Image index')
    ax.set_ylabel('Position')
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_dir / f'origin.{plot_extension}')


def plot_rotation(
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path,
        measurements: Dict[str, np.ndarray] = None
):
    """
    Plot rotation.
    """
    rotations = parameters['rotation']
    x_vals = [idx for idx, _ in image_paths]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.set_title('Axis-angle rotation vector components')
    ax.grid()
    for i in range(3):
        ax.plot(x_vals, rotations[:, i], label='$R_' + 'xyz'[i] + '$')
    if measurements is not None:
        rotations = measurements['rotation']
        x_vals = measurements['idx']
        for i in range(3):
            ax.plot(x_vals, rotations[:, i], label='$R_' + 'xyz'[i] + '$ (manual)', linestyle='none',
                    marker=marker_styles[i], markersize=5, alpha=0.7)
    ax.set_xlabel('Image index')
    ax.set_ylabel('Component value')
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_dir / f'rotation.{plot_extension}')
