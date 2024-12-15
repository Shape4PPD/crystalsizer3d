from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from torch import Tensor

from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.projector import Projector
from crystalsizer3d.refiner.keypoint_detection import to_absolute_coordinates
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.util.utils import get_crystal_face_groups, init_tensor, smooth_signal, to_numpy, to_rgb

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
        property_name: str,
        property_values: np.ndarray,
        face_groups: Dict[Tuple[int, int, int], Dict[Tuple[int, int, int], int]],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path = None,
        measurement_idxs: np.ndarray = None,
        measurement_values: np.ndarray = None,
        show_mean: bool = False,
        show_std: bool = False,
        make_means_plot: bool = True
) -> Figure | Tuple[Figure, Figure]:
    """
    Plot face distances or areas.
    """
    x_vals = [idx for idx, _ in image_paths]
    n_groups = len(face_groups)
    group_colours = get_face_group_colours(n_groups)

    if property_name == 'areas':
        y_label = 'Face area'
    elif property_name == 'distances':
        y_label = 'Distance'
    else:
        raise ValueError(f'Invalid property name: {property_name}')

    # Restrict the measurements to within the range predicted
    if measurement_idxs is not None:
        include_mask = np.array([x_vals[0] <= idx <= x_vals[-1] for idx in measurement_idxs])
        measurement_idxs = measurement_idxs[include_mask]
        measurement_values = measurement_values[include_mask]

    # Make a grid of plots showing the values for each face group
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))
    fig_grouped = plt.figure(figsize=(n_cols * 6, n_rows * 4))
    gs = GridSpec(
        n_rows, n_cols,
        top=0.95, bottom=0.08, right=0.99, left=0.05,
        hspace=0.3, wspace=0.2
    )
    for i, (group_hkl, group_idxs) in enumerate(face_groups.items()):
        colour = group_colours[i]
        colour_variants = get_colour_variations(colour, len(group_idxs))
        y = property_values[:, list(group_idxs.values())]
        y_measured = measurement_values[:, list(group_idxs.values())] if measurement_values is not None else None
        y_mean = y.mean(axis=1)
        y_std = y.std(axis=1)
        lbls = [get_hkl_label(hkl) for hkl in list(group_idxs.keys())]

        ax = fig_grouped.add_subplot(gs[i])
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
    if save_dir is not None:
        plt.savefig(save_dir / f'{property_name}_grouped.{plot_extension}')
    if not make_means_plot:
        return fig_grouped

    # Make a plot showing the mean values all together
    fig_mean, ax = plt.subplots(1, figsize=(12, 8))
    ax.set_title(f'Mean {property_name}')
    ax.grid()
    for i, (group_hkl, group_idxs) in enumerate(face_groups.items()):
        colour = group_colours[i]
        y = property_values[:, list(group_idxs.values())]
        y_mean = y.mean(axis=1)
        y_std = y.std(axis=1)
        ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std, color=colour, alpha=0.1)
        ax.plot(x_vals, y_mean, c=colour, label=get_hkl_label(group_hkl, is_group=True))
    ax.set_xlabel('Image index')
    ax.set_ylabel(y_label)
    ax.legend()
    fig_mean.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir / f'{property_name}_mean.{plot_extension}')

    return fig_grouped, fig_mean


def plot_distances(
        parameters: Dict[str, np.ndarray],
        face_groups: Dict[Tuple[int, int, int], Dict[Tuple[int, int, int], int]],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path = None,
        measurements: Dict[str, np.ndarray] = None,
        make_means_plot: bool = True,
        **kwargs
) -> Figure | Tuple[Figure, Figure]:
    """
    Plot face distances.
    """
    distances = parameters['distances']
    scales = parameters['scale']
    return plot_face_property_values(
        property_name='distances',
        property_values=distances * scales[:, None],
        face_groups=face_groups,
        image_paths=image_paths,
        save_dir=save_dir,
        measurement_idxs=measurements['idx'] if measurements is not None else None,
        measurement_values=measurements['distances'] * measurements['scale'][:, None]
        if measurements is not None else None,
        make_means_plot=make_means_plot
    )


def plot_areas(
        manager: Manager,
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path = None,
        face_groups: Dict[Tuple[int, int, int], Dict[Tuple[int, int, int], int]] = None,
        measurements: Dict[str, np.ndarray] = None,
        make_means_plot: bool = True,
        **kwargs
) -> Figure | Tuple[Figure, Figure]:
    """
    Plot face areas.
    """
    distances = parameters['distances']
    scales = parameters['scale']
    if isinstance(distances, Tensor):
        distances = to_numpy(distances)
    if isinstance(scales, Tensor):
        scales = to_numpy(scales)

    # Get a crystal object
    ds = manager.ds
    cs = CSDProxy().load(ds.dataset_args.crystal_id)
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

    if face_groups is None:
        face_groups = get_crystal_face_groups(manager)

    return plot_face_property_values(
        property_name='areas',
        property_values=areas,
        face_groups=face_groups,
        image_paths=image_paths,
        save_dir=save_dir,
        measurement_idxs=measurements['idx'] if measurements is not None else None,
        measurement_values=areas_m,
        make_means_plot=make_means_plot
    )


def plot_origin(
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path = None,
        measurements: Dict[str, np.ndarray] = None,
        **kwargs
) -> Figure:
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
    if save_dir is not None:
        plt.savefig(save_dir / f'origin.{plot_extension}')
    return fig


def plot_rotation(
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path = None,
        measurements: Dict[str, np.ndarray] = None,
        **kwargs
) -> Figure:
    """
    Plot rotation.
    """

    def canonicalise_rotations(rotations: np.ndarray) -> np.ndarray:
        """Canonicalise the rotation vectors."""
        angles = np.linalg.norm(rotations, axis=-1)
        return rotations / angles[:, None] * (angles % (2 * np.pi))

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
    if save_dir is not None:
        plt.savefig(save_dir / f'rotation.{plot_extension}')
    return fig


def plot_material_properties(
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path = None,
        measurements: Dict[str, np.ndarray] = None,
        **kwargs
) -> Figure:
    """
    Plot IOR and roughness.
    """
    x_vals = [idx for idx, _ in image_paths]
    fig, axes = plt.subplots(2, figsize=(10, 8), sharex=True)
    for ax, prop_name in zip(axes, ['ior', 'roughness']):
        ax.set_title(f'{prop_name}')
        ax.grid()
        y = parameters['material_' + prop_name]
        ax.plot(x_vals, y, label=prop_name)
        if measurements is not None and prop_name in measurements:
            y = measurements[prop_name]
            ax.plot(measurements['idx'], y, label='Manual', linestyle='none',
                    marker='o', markersize=5, alpha=0.7)
        ax.set_xlabel('Image index')
        ax.set_ylabel(prop_name)
        ax.legend()
    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir / f'material.{plot_extension}')
    return fig


def plot_light_radiance(
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path = None,
        **kwargs
) -> Figure:
    """
    Plot rotation.
    """
    radiance = parameters['light_radiance']
    x_vals = [idx for idx, _ in image_paths]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.set_title('Light radiance (RGB)')
    ax.grid()
    for i in range(3):
        ax.plot(x_vals, radiance[:, i], label='RGB'[i])
    ax.set_xlabel('Image index')
    ax.set_ylabel('Component value')
    ax.legend()
    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir / f'light_radiance.{plot_extension}')
    return fig


@torch.no_grad()
def annotate_image(
        image_path: Path,
        scene: Scene | None = None,
        crystal: Crystal | None = None,
        zoom: float | None = None,
        keypoints: np.ndarray | None = None,
        edge_points: np.ndarray | None = None,
        edge_point_deltas: np.ndarray | None = None,
        wf_line_width: int = 3,
        keypoint_radius: int = 15,
        kp_fill_colour: str = 'lightgreen',
        kp_outline_colour: str = 'darkgreen',
        ep_colour: str = 'yellow',
        epd_colour: str = 'orange',
) -> Image:
    """
    Draw the projected wireframe onto an image.
    """
    assert (scene is None and crystal is not None and zoom is not None) \
           or (scene is not None and crystal is None and zoom is None), \
        'Either provide a scene and no crystal or zoom, or no scene and a crystal and zoom.'

    # Get the crystal and zoom from the scene
    if scene is not None:
        crystal = scene.crystal
        zoom = orthographic_scale_factor(scene)

    # Load the image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    image_size = min(img.size)
    if img.size[0] != img.size[1]:
        offset_l = (img.size[0] - image_size) // 2
        offset_t = (img.size[1] - image_size) // 2
    else:
        offset_l = 0
        offset_t = 0

    # Set up the projector
    projector = Projector(
        crystal=crystal,
        image_size=(image_size, image_size),
        zoom=zoom,
        transparent_background=True,
        multi_line=True,
        rtol=1e-2
    )

    # Draw the wireframe
    draw = ImageDraw.Draw(img, 'RGB')
    for ref_face_idx, face_segments in projector.edge_segments.items():
        if len(face_segments) == 0:
            continue
        colour = projector.colour_facing_towards if ref_face_idx == 'facing' else projector.colour_facing_away
        colour = tuple((colour * 255).int().tolist())
        for segment in face_segments:
            l = segment.clone()
            l[:, 0] = torch.clamp(l[:, 0], 1, projector.image_size[1] - 2) + offset_l
            l[:, 1] = torch.clamp(l[:, 1], 1, projector.image_size[0] - 2) + offset_t
            draw.line(xy=[tuple(l[0].int().tolist()), tuple(l[1].int().tolist())],
                      fill=colour, width=wf_line_width)

    # Add the keypoints
    if keypoints is not None:
        draw = ImageDraw.Draw(img, 'RGBA')
        keypoints = to_absolute_coordinates(keypoints, image_size)
        kp_fill_colour = tuple((np.array(to_rgb(kp_fill_colour) + (0.3,)) * 255).astype(np.uint8).tolist())
        kp_outline_colour = tuple((np.array(to_rgb(kp_outline_colour) + (1,)) * 255).astype(np.uint8).tolist())
        for (x, y) in keypoints:
            draw.circle((x, y), keypoint_radius, fill=kp_fill_colour, outline=kp_outline_colour,
                        width=keypoint_radius // 6)

    # Add the edge points
    if edge_points is not None:
        draw = ImageDraw.Draw(img, 'RGBA')
        edge_points = to_absolute_coordinates(edge_points, image_size)
        edge_point_deltas = edge_point_deltas / 2 * image_size
        ep_colour = tuple((np.array(to_rgb(ep_colour) + (0.3,)) * 255).astype(np.uint8).tolist())
        epd_colour = tuple((np.array(to_rgb(epd_colour) + (1,)) * 255).astype(np.uint8).tolist())
        for i, ((x, y), (dx, dy)) in enumerate(zip(edge_points, edge_point_deltas)):
            draw.circle((x, y), 5, fill=ep_colour, outline=epd_colour)
            draw.line([x, y, x + dx, y + dy], fill=epd_colour, width=keypoint_radius // 6)

    return img


@torch.no_grad()
def annotate_image_with_keypoints(
        image_path: Path,
        keypoints: np.ndarray | None = None,
        keypoint_radius: int = 15,
        kp_fill_colour: str = 'lightgreen',
        kp_outline_colour: str = 'darkgreen',
) -> Image:
    """
    Draw the keypoints onto an image.
    """
    # Load the image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    image_size = min(img.size)

    # Draw the keypoints
    draw = ImageDraw.Draw(img, 'RGBA')
    keypoints = to_absolute_coordinates(keypoints, image_size)
    kp_fill_colour = tuple((np.array(to_rgb(kp_fill_colour) + (0.3,)) * 255).astype(np.uint8).tolist())
    kp_outline_colour = tuple((np.array(to_rgb(kp_outline_colour) + (1,)) * 255).astype(np.uint8).tolist())
    for (x, y) in keypoints:
        draw.circle((x, y), keypoint_radius, fill=kp_fill_colour, outline=kp_outline_colour,
                    width=keypoint_radius // 6)

    return img
