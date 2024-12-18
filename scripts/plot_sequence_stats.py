import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from mayavi import mlab

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.args.refiner_args import PREDICTOR_ARG_NAMES, PREDICTOR_ARG_NAMES_BS1, RefinerArgs
from crystalsizer3d.args.sequence_fitter_args import SequenceFitterArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.sequence.plots import annotate_image, get_colour_variations, get_face_group_colours, get_hkl_label, \
    line_styles, marker_styles
from crystalsizer3d.sequence.sequence_fitter import SEQUENCE_DATA_PATH
from crystalsizer3d.sequence.utils import get_image_paths, load_manual_measurements
from crystalsizer3d.util.image_scale import calculate_distance
from crystalsizer3d.sequence.volumes import generate_or_load_manual_measurement_volumes, generate_or_load_volumes
from crystalsizer3d.util.plots import make_3d_digital_crystal_image
from crystalsizer3d.util.utils import get_crystal_face_groups, hash_data, init_tensor, json_to_numpy, print_args, \
    set_seed, to_dict, to_numpy, to_rgb

# Off-screen rendering
mlab.options.offscreen = True


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to plot some sequence stats.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for the random number generator.')

    # Sequence
    parser.add_argument('--sf-path', type=Path, required=True,
                        help='Path to the sequence fitter directory.')
    parser.add_argument('--start-image', type=int, default=0,
                        help='Start with this image.')
    parser.add_argument('--end-image', type=int, default=-1,
                        help='End with this image.')
    parser.add_argument('--frame-idxs', type=lambda s: (int(i) for i in s.split(',')), default=(3, 100, 502),
                        help='Plot these frame idxs.')
    parser.add_argument('--interval-mins', type=int, default=5,
                        help='Interval in minutes between frames.')
    parser.add_argument('--pixel-to-mm', type=float, default=None,
                        help="Pixels to millimeters conversion factor. If not provided, it will be calculated dynamically.")

    args = parser.parse_args()

    assert args.sf_path.exists(), f'Path does not exist: {args.sf_path}'

    # Set the random seed
    set_seed(args.seed)

    return args


def _init() -> Tuple[Namespace, Path]:
    """
    Initialise the dataset and get the command line arguments.
    """
    args = get_args()
    print_args(args)

    # Write the args to the output dir
    output_dir = LOGS_PATH / START_TIMESTAMP
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'args.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    return args, output_dir


def _get_ts(x_vals: np.ndarray, interval: int) -> List[float]:
    """
    Get the timestamps as cumulative seconds.
    """
    ts = [interval * 60 * i for i in range(len(x_vals))]  # Store cumulative time in seconds
    return ts


def _format_xtime(ax: plt.Axes, ts: List[float], add_xlabel: bool = True):
    """
    Format the x-axis label with cumulative hours.
    """

    def format_func(x, _):
        hours = int(x // 3600)  # Convert seconds to hours
        return f'{hours}'

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_func))

    # Automatically set major ticks at 6 or 12-hour intervals, depending on the length
    max_hours = ts[-1] / 3600  # Total hours
    if max_hours <= 30:
        interval = 6  # Use 6-hour intervals for shorter time ranges
    else:
        interval = 12  # Use 12-hour intervals for longer time ranges

    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=interval * 3600))  # Set major ticks every interval hours

    ax.set_xlim([ts[0], ts[-1]])
    if add_xlabel:
        ax.set_xlabel('Hours elapsed')


def plot_sequence_errors_for_sampler():
    """
    Plot the sequence errors as a bar chart for illustrating the adaptive sampler.
    """
    args, output_dir = _init()
    bs = 5
    N = 20

    # Load the losses
    with open(args.sf_path / 'losses.json', 'r') as f:
        losses = json_to_numpy(json.load(f))
    losses = losses['total/train']
    losses = losses[::len(losses) // N]

    # Sample some frames
    probabilities = losses / losses.sum()
    sampled_idxs = np.random.choice(len(losses), bs, p=probabilities, replace=False)

    # Set up the colours
    colours = ['lightgrey'] * len(losses)
    for idx in sampled_idxs:
        colours[idx] = 'darkred'

    fig, ax = plt.subplots(
        1, 1,
        figsize=(0.85, 0.47),
        gridspec_kw={'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1}
    )
    # fig = plt.figure(figsize=(5, 3))
    ax.bar(range(len(losses)), losses, color=colours)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(output_dir / 'sequence_errors.svg')
    plt.show()


def plot_3d_crystals():
    """
    Plot the 3d crystals.
    """
    args, output_dir = _init()
    frame_idxs = args.frame_idxs

    for i, idx in enumerate(frame_idxs):
        logger.info(f'Plotting digital crystal for frame {idx} ({i + 1}/{len(frame_idxs)})')
        scene = Scene.from_yml(args.sf_path / 'scenes' / 'eval' / f'{idx:04d}.yml')
        crystal = scene.crystal
        dig_pred = make_3d_digital_crystal_image(
            crystal=crystal,
            res=1000,
            wireframe_radius_factor=0.03,
            azim=-10,
            elev=65,
            roll=-85,
            distance=4,
        )
        dig_pred.save(output_dir / f'digital_predicted_{idx:02d}.png')


def plot_distances_and_areas():
    """
    Plot the distances.
    """
    args, output_dir = _init()
    show_n = 5000

    # Load the args
    with open(args.sf_path / 'args_sequence_fitter.yml', 'r') as f:
        sf_args = SequenceFitterArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_refiner.yml', 'r') as f:
        ref_args = RefinerArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_runtime.yml', 'r') as f:
        runtime_args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    # Instantiate a manager
    manager = Manager.load(
        model_path=ref_args.predictor_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': False,
                'batch_size': 1
            },
        },
        save_dir=output_dir
    )

    # Set up some variables
    logger.info('Setting up plot variables.')
    face_groups = get_crystal_face_groups(manager)
    image_paths = get_image_paths(sf_args, load_all=True)[:show_n]
    x_vals = [idx for idx, _ in image_paths]
    n_groups = len(face_groups)
    group_colours = get_face_group_colours(n_groups)

    # Check if pixel_to_mm is provided
    if args.pixel_to_mm is None:
        logger.info("No pixel-to-mm conversion factor provided. Calculating dynamically...")
        args.pixel_to_mm = calculate_distance(image_paths[0][1])

    # Output the pixel-to-mm conversion factor
    logger.info(f"Pixel-to-mm conversion factor: {args.pixel_to_mm:.4f} mm/pixel")

    scene = Scene.from_yml(sf_args.initial_scene)
    im = Image.open(image_paths[-1][1])
    width, height = im.size
    zoom = orthographic_scale_factor(scene)
    dist_to_mm = zoom * args.pixel_to_mm * min(width, height)

    # Load the measurements
    logger.info(f'Loading manual measurements from {runtime_args.measurements_dir}')
    measurements = load_manual_measurements(
        measurements_dir=Path(runtime_args.measurements_dir),
        manager=manager
    )
    measurement_idxs = measurements['idx']
    distances_m = measurements['distances']
    scales_m = measurements['scale'] * dist_to_mm

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

    # Load the parameters
    param_path = args.sf_path / 'parameters.json'
    logger.info(f'Loading parameters from {param_path}.')
    with open(param_path, 'r') as f:
        parameters = json_to_numpy(json.load(f))
    distances = parameters['eval']['distances'][:show_n]
    scales = parameters['eval']['scale'][:show_n] * dist_to_mm

    # Calculate the face areas
    logger.info('Calculating face areas.')
    areas = np.zeros_like(distances)
    for i in range(len(distances)):
        crystal.build_mesh(distances=init_tensor(distances[i], dtype=torch.float64))
        unscaled_areas = np.array([crystal.areas[tuple(hkl.tolist())] for hkl in crystal.all_miller_indices])
        areas[i] = unscaled_areas * scales[i]**2

    # Calculate the face areas for the manual measurements
    logger.info('Calculating face areas for the manual measurements.')
    areas_m = np.zeros_like(distances_m)
    for i in range(len(distances_m)):
        crystal.build_mesh(distances=init_tensor(distances_m[i], dtype=torch.float64))
        unscaled_areas_m = np.array([crystal.areas[tuple(hkl.tolist())] for hkl in crystal.all_miller_indices])
        areas_m[i] = unscaled_areas_m * scales_m[i]**2

    # Restrict the measurements to within the range predicted
    include_mask = np.array([x_vals[0] <= idx <= x_vals[-1] for idx in measurement_idxs])
    measurement_idxs = measurement_idxs[include_mask]
    distances_m = distances_m[include_mask]
    scales_m = scales_m[include_mask]
    areas_m = areas_m[include_mask]

    # Get the timestamps
    ts = _get_ts(x_vals, args.interval_mins)
    m_idxs = np.array([(x_vals == m_idx).nonzero()[0] for m_idx in measurement_idxs]).squeeze()
    m_ts = [ts[m_idx] for m_idx in m_idxs]

    # Set font sizes
    plt.rc('axes', titlesize=7, titlepad=1)  # fontsize of the title
    plt.rc('axes', labelsize=6, labelpad=1)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=6)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=7)  # fontsize of the legend
    plt.rc('xtick.major', pad=2, size=2)
    plt.rc('xtick.minor', pad=2, size=1)
    plt.rc('ytick.major', pad=1, size=2)
    plt.rc('axes', linewidth=0.5)

    def format_tick(value, _):
        return f'{value:.2g}'

    # Make a grid of plots showing the values for each face group
    n_cols = 6  # int(np.ceil(np.sqrt(n_groups)))
    n_rows = 2  # int(np.ceil(n_groups / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5, 1.6),
        gridspec_kw=dict(
            top=0.92, bottom=0.13, right=0.99, left=0.05,
            hspace=0.1, wspace=0.4
        ),
    )
    for i, (group_hkl, group_idxs) in enumerate(face_groups.items()):
        logger.info(f'Plotting group {i + 1}/{n_groups}')
        colour = group_colours[i]
        colour_variants = get_colour_variations(colour, len(group_idxs))
        lbls = [get_hkl_label(hkl) for hkl in list(group_idxs.keys())]

        axd = axes[0, i]
        axa = axes[1, i]
        axd.set_title(get_hkl_label(group_hkl, is_group=True))
        d = distances[:, list(group_idxs.values())] * scales[:, None]
        dm = distances_m[:, list(group_idxs.values())] * scales_m[:, None]
        a = areas[:, list(group_idxs.values())]
        am = areas_m[:, list(group_idxs.values())]

        for ax, y, y_m in zip([axd, axa], [d, a], [dm, am]):
            _format_xtime(ax, ts, add_xlabel=(ax == axa))
            for j, (y_j, lbl, colour_j) in enumerate(zip(y.T, lbls, colour_variants)):
                ax.plot(ts, y_j, label=lbl, c=colour_j,
                        linestyle=line_styles[j % len(line_styles)], linewidth=0.4, alpha=0.8)
                marker = marker_styles[j % len(marker_styles)]
                scatter_args = dict(
                    label=lbl + ' (manual)',
                    color=colour_j,
                    marker=marker,
                    alpha=0.8,
                    linewidth=0.2
                )
                if marker in ['o', 's']:
                    scatter_args['facecolors'] = 'none'
                    scatter_args['s'] = 1.5
                else:
                    scatter_args['s'] = 2
                ax.scatter(m_ts, y_m[:, j], **scatter_args)

            if i == 0:
                ax.set_ylabel('Distance (mm)' if ax == axd else 'Area (mm²)')
            if ax != axa:
                ax.tick_params(labelbottom=False)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_tick))

    plt.savefig(output_dir / 'distances_and_areas.svg', transparent=True)
    plt.show()


def make_3d_digital_crystal_image_with_highlight(
        crystal: Crystal,
        highlight_hkls: List[Tuple[int, int, int]],
        highlight_colours: List[np.ndarray],
        res: int = 100,
        bg_col: float = 1.,
        wireframe_radius_factor: float = 0.1,
        surface_colour: str = 'skyblue',
        wireframe_colour: str = 'cornflowerblue',
        opacity: float = 0.6,
        opacity_highlight: float = 1,
        azim: float = 150,
        elev: float = 160,
        distance: Optional[float] = None,
        roll: float = 0
) -> Image:
    """
    Make a 3D image of the crystal.
    """
    assert len(highlight_hkls) == len(highlight_colours)
    fig = mlab.figure(size=(res * 2, res * 2), bgcolor=(bg_col, bg_col, bg_col))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Add faces individually
    for hkl, fv_idxs in crystal.faces.items():
        fv = to_numpy(crystal.vertices[fv_idxs])
        cp = fv.mean(axis=0)
        mfv = np.stack([cp, *fv])
        N = len(fv)
        jdx = np.arange(N)
        mfi = np.stack([np.zeros(N, dtype=np.int64), jdx % N + 1, (jdx + 1) % N + 1], axis=1)

        if hkl in highlight_hkls:
            colour = tuple(highlight_colours[highlight_hkls.index(hkl)].tolist())
            opacity_face = opacity_highlight
        else:
            colour = to_rgb(surface_colour)
            opacity_face = opacity

        mlab.triangular_mesh(*mfv.T, mfi, figure=fig, color=colour, opacity=opacity_face)

    # Add wireframe
    tube_radius = max(0.0001, crystal.distances[0].item() * wireframe_radius_factor)
    for fv_idxs in crystal.faces.values():
        fv = to_numpy(crystal.vertices[fv_idxs])
        fv = np.vstack([fv, fv[0]])  # Close the loop
        mlab.plot3d(*fv.T, color=to_rgb(wireframe_colour), tube_radius=tube_radius)

    # Render
    mlab.view(figure=fig, azimuth=azim, elevation=elev, distance=distance, roll=roll, focalpoint=np.zeros(3))

    # # Useful for getting the view parameters when recording from the gui:
    # mlab.show()
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [-3.6962036386805432, 0.6413960469922849, 1.3880525106450643]
    # scene.scene.camera.focal_point = [0.0, 0.0, 0.0]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [0.335754138129981, -0.09354125783100545, 0.9372935462340426]
    # scene.scene.camera.clipping_range = [1.0143450579340336, 7.546385011525549]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    # mlab.show()
    # exit()

    # fig.scene.render()
    frame = mlab.screenshot(mode='rgba', antialiased=True, figure=fig)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = Image.fromarray((frame * 255).astype(np.uint8), 'RGBA')

    mlab.close()

    return img


def plot_3d_crystals_with_highlighted_faces():
    """
    Plot some 3d crystals with highlighted faces.
    """
    args, output_dir = _init()
    frame_idx = args.frame_idxs[-1]

    # Load the args
    with open(args.sf_path / 'args_refiner.yml', 'r') as f:
        ref_args = RefinerArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))

    # Instantiate a manager
    manager = Manager.load(
        model_path=ref_args.predictor_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': False,
                'batch_size': 1
            },
        },
        save_dir=output_dir
    )

    # Get the face groups
    face_groups = get_crystal_face_groups(manager)
    n_groups = len(face_groups)
    group_colours = get_face_group_colours(n_groups)

    # Load the template crystal
    scene = Scene.from_yml(args.sf_path / 'scenes' / 'eval' / f'{frame_idx:04d}.yml')
    crystal = scene.crystal
    crystal.origin.data = torch.zeros_like(crystal.origin)

    for i, (group_hkl, group_idxs) in enumerate(face_groups.items()):
        colour = group_colours[i]
        colour_variants = get_colour_variations(colour, len(group_idxs))
        img = make_3d_digital_crystal_image_with_highlight(
            crystal=crystal,
            highlight_hkls=list(group_idxs.keys()),
            highlight_colours=colour_variants,
            res=1000,
            wireframe_radius_factor=0.02,
            surface_colour='lightgrey',
            wireframe_colour='darkgrey',
            opacity=0.3,
            opacity_highlight=0.8,
            azim=170,
            elev=70,
            roll=95,
            distance=4.3,
        )
        group_lbl = ','.join([str(hkl) for hkl in group_hkl])
        img.save(output_dir / f'crystal_highlight_{group_lbl}.png')


def plot_volume():
    """
    Plot the volume.
    """
    args, output_dir = _init()
    show_n = 5000

    # Load the args
    with open(args.sf_path / 'args_sequence_fitter.yml', 'r') as f:
        sf_args = SequenceFitterArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_refiner.yml', 'r') as f:
        ref_args = RefinerArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_runtime.yml', 'r') as f:
        runtime_args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    # Load the volumes
    volumes = generate_or_load_volumes(
        sf_path=args.sf_path,
        cache_only=False
    )
    volumes_m = generate_or_load_manual_measurement_volumes(
        measurements_dir=Path(runtime_args.measurements_dir),
        predictor_model_path=ref_args.predictor_model_path,
        cache_only=False
    )

    # Instantiate a manager
    manager = Manager.load(
        model_path=ref_args.predictor_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': False,
                'batch_size': 1
            },
        },
        save_dir=output_dir
    )

    # Load the measurements
    logger.info(f'Loading manual measurements from {runtime_args.measurements_dir}')
    measurements = load_manual_measurements(
        measurements_dir=Path(runtime_args.measurements_dir),
        manager=manager
    )
    measurement_idxs = measurements['idx']

    # Set up some variables
    logger.info('Setting up plot variables.')
    image_paths = get_image_paths(sf_args, load_all=True)[:show_n]
    x_vals = [idx for idx, _ in image_paths]

    # Restrict the measurements to within the range predicted
    include_mask = np.array([x_vals[0] <= idx <= x_vals[-1] for idx in measurement_idxs])
    measurement_idxs = measurement_idxs[include_mask]
    volumes_m = volumes_m[include_mask]

    # Set font sizes
    plt.rc('axes', titlesize=7, titlepad=1)  # fontsize of the title
    plt.rc('axes', labelsize=6, labelpad=2)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=6)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=2, size=2)
    plt.rc('xtick.minor', pad=2, size=1)
    plt.rc('ytick.major', pad=1, size=2)
    plt.rc('axes', linewidth=0.5)

    # Make a single plot showing the values for each face group
    fig, ax = plt.subplots(
        1, 1,
        figsize=(1.7, 1.5),
        gridspec_kw=dict(
            top=0.98, bottom=0.15, right=0.99, left=0.2,
            hspace=0.1, wspace=0.4
        ),
    )
    ax.set_ylabel('Volume (mm³)')
    ts = _get_ts(x_vals, args.interval_mins)
    m_idxs = np.array([(x_vals == m_idx).nonzero()[0] for m_idx in measurement_idxs]).squeeze()
    m_ts = [ts[m_idx] for m_idx in m_idxs]
    _format_xtime(ax, ts)
    ax.plot(ts, volumes, c='black', linewidth=0.6)
    ax.plot(m_ts, volumes_m, c='grey', marker='o', linestyle='none',
            markersize=5, markerfacecolor='none')
    plt.savefig(output_dir / 'volumes.svg')
    plt.show()


def plot_concentration():
    """
    Plot the concentration.
    """
    args, output_dir = _init()
    show_n = 5000

    # Variables
    initial_concentration = 16
    crystal_density = 1.47
    cuvette_volume = 0.5

    # Load the args
    with open(args.sf_path / 'args_sequence_fitter.yml', 'r') as f:
        sf_args = SequenceFitterArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_refiner.yml', 'r') as f:
        ref_args = RefinerArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_runtime.yml', 'r') as f:
        runtime_args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    # Load the volumes
    volumes = generate_or_load_volumes(
        sf_path=args.sf_path,
        cache_only=False
    )
    volumes_m = generate_or_load_manual_measurement_volumes(
        measurements_dir=Path(runtime_args.measurements_dir),
        predictor_model_path=ref_args.predictor_model_path,
        cache_only=False
    )

    # Calculate the concentrations
    concentrations = np.zeros(len(volumes))
    concentrations[0] = initial_concentration
    for i in range(1, len(volumes)):
        concentrations[i] = concentrations[i - 1] - (volumes[i] - volumes[i - 1]) * crystal_density / cuvette_volume
    concentrations_m = np.zeros(len(volumes_m))
    concentrations_m[0] = initial_concentration
    for i in range(1, len(volumes_m)):
        concentrations_m[i] = concentrations_m[i - 1] - (
                volumes_m[i] - volumes_m[i - 1]) * crystal_density / cuvette_volume

    # Instantiate a manager
    manager = Manager.load(
        model_path=ref_args.predictor_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': False,
                'batch_size': 1
            },
        },
        save_dir=output_dir
    )

    # Load the measurements
    logger.info(f'Loading manual measurements from {runtime_args.measurements_dir}')
    measurements = load_manual_measurements(
        measurements_dir=Path(runtime_args.measurements_dir),
        manager=manager
    )
    measurement_idxs = measurements['idx']

    # Set up some variables
    logger.info('Setting up plot variables.')
    image_paths = get_image_paths(sf_args, load_all=True)[:show_n]
    x_vals = [idx for idx, _ in image_paths]

    # Restrict the measurements to within the range predicted
    include_mask = np.array([x_vals[0] <= idx <= x_vals[-1] for idx in measurement_idxs])
    measurement_idxs = measurement_idxs[include_mask]
    concentrations_m = concentrations_m[include_mask]

    # Set font sizes
    plt.rc('axes', titlesize=7, titlepad=1)  # fontsize of the title
    plt.rc('axes', labelsize=6, labelpad=2)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=6)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=2, size=2)
    plt.rc('xtick.minor', pad=2, size=1)
    plt.rc('ytick.major', pad=1, size=2)
    plt.rc('axes', linewidth=0.5)

    # Make a single plot showing the values for each face group
    fig, ax = plt.subplots(
        1, 1,
        figsize=(1.7, 1.5),
        gridspec_kw=dict(
            top=0.98, bottom=0.15, right=0.99, left=0.2,
            hspace=0.1, wspace=0.4
        ),
    )
    ax.set_ylabel('Concentration (mg/ml)')
    ts = _get_ts(x_vals, args.interval_mins)
    m_idxs = np.array([(x_vals == m_idx).nonzero()[0] for m_idx in measurement_idxs]).squeeze()
    m_ts = [ts[m_idx] for m_idx in m_idxs]
    _format_xtime(ax, ts)
    ax.plot(ts, concentrations, c='black', linewidth=0.6)
    ax.plot(m_ts, concentrations_m, c='grey', marker='o', linestyle='none',
            markersize=5, markerfacecolor='none')
    plt.savefig(output_dir / 'concentrations.svg')
    plt.show()


def create_legend_plot():
    """
    Create a plot showing the legends.
    """
    line_styles = ['-', '--', '-.', ':']
    marker_styles = ['o', 'x', 's', '+']

    # Create dummy line objects for automatic measurements
    automatic_lines = [
        mlines.Line2D([], [], color='black', linestyle=style) for style in line_styles
    ]

    # Create dummy marker objects for manual measurements
    manual_markers = [
        mlines.Line2D([], [], color='black', linestyle='None', marker=marker) for marker in marker_styles
    ]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')  # Turn off the axis

    # Create the legend with grouped handles
    legend_handles = automatic_lines + manual_markers
    legend_labels = ['Automatic Measurements'] * len(line_styles) + ['Manual Measurements'] * len(marker_styles)

    # Adding grouped handles with labels
    ax.legend(legend_handles, legend_labels, loc='center', frameon=False, ncol=2, handletextpad=2)

    args, output_dir = _init()
    plt.savefig(output_dir / 'legend.svg', transparent=True)
    plt.show()


def plot_sequence_errors():
    """
    Plot the sequence errors.
    """
    args, output_dir = _init()
    N = 20

    # Load the args
    with open(args.sf_path / 'args_sequence_fitter.yml', 'r') as f:
        sf_args = SequenceFitterArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_refiner.yml', 'r') as f:
        ref_args = RefinerArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_runtime.yml', 'r') as f:
        runtime_args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    # Instantiate a manager
    manager = Manager.load(
        model_path=ref_args.predictor_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': False,
                'batch_size': 1
            },
        },
        save_dir=output_dir
    )

    # Load the initial prediction losses
    base_path = SEQUENCE_DATA_PATH / f'{sf_args.images_dir.stem}_{hash_data(sf_args.images_dir.absolute())}'
    pred_args = {k: getattr(ref_args, k) for k in PREDICTOR_ARG_NAMES}
    if ref_args.initial_pred_batch_size == 1:
        pred_hash = hash_data({k: getattr(ref_args, k) for k in PREDICTOR_ARG_NAMES_BS1})
    else:
        pred_hash = hash_data(pred_args)
    model_id = pred_args['predictor_model_path'].stem[:4]
    pred_cache_dir = base_path / 'predictor' / f'{model_id}_{pred_hash}'
    losses_path = pred_cache_dir / f'losses_{hash_data(pred_args)}.json'
    assert losses_path.exists(), f'Initial prediction losses file does not exist: {losses_path}'
    with open(losses_path, 'r') as f:
        losses_init = json_to_numpy(json.load(f))

    # Load the refined losses
    with open(args.sf_path / 'losses.json', 'r') as f:
        losses = json_to_numpy(json.load(f))
    losses = losses['eval']

    # Load measurements
    measurements = load_manual_measurements(
        measurements_dir=Path(runtime_args.measurements_dir),
        manager=manager
    )
    measurement_idxs = measurements['idx']

    # Load the image paths and set up the x-values
    image_paths = get_image_paths(sf_args, load_all=True)
    x_vals = [idx for idx, _ in image_paths]
    ts = _get_ts(x_vals, args.interval_mins)
    m_idxs = np.concatenate([(x_vals == m_idx).nonzero()[0] for m_idx in measurement_idxs])
    m_ts = [ts[m_idx] for m_idx in m_idxs]

    # Plot the losses
    fig, axes = plt.subplots(
        3, 1,
        figsize=(5, 3),
        sharex=True,
        gridspec_kw={'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1}
    )

    c_init = '#0d58a1'
    c_refined = '#6a35b5'

    # Measurement errors
    lm_init = losses_init['measurement'][m_idxs]
    lm_refined = losses['measurement'][m_idxs]
    ax = axes[0]
    # ax.set_title('Measurement Errors')
    ax.plot(m_ts, lm_init, c=c_init, label='Initial', marker='o', linestyle=':', markerfacecolor='none')
    ax.plot(m_ts, lm_refined, c=c_refined, label='Refined', marker='o', linestyle=':', markerfacecolor='none')
    ax.set_ylabel('Measurement Error')
    ax.set_yscale('log')
    # _format_xtime(ax, ts)

    # L2 errors
    ax = axes[1]
    # ax.set_title('L2 Errors')
    ax.plot(ts, losses_init['perceptual'], c=c_init, label='Initial')
    ax.plot(ts, losses['perceptual'], c=c_refined, label='Refined')
    ax.set_ylabel('L2 Error')
    ax.set_yscale('log')
    # _format_xtime(ax, ts)

    # Keypoint errors
    ax = axes[2]
    # ax.set_title('Keypoint Errors')
    ax.plot(ts, losses_init['keypoints'], c=c_init, label='Initial')
    ax.plot(ts, losses['keypoints'], c=c_refined, label='Refined')
    ax.set_ylabel('Keypoint Error')
    ax.set_yscale('log')
    _format_xtime(ax, ts, add_xlabel=True)

    # plt.savefig(output_dir / 'sequence_errors.svg')
    plt.show()


def plot_test_cube():
    """
    Plot a test cube with distances of 1mm between origin and each face for scale matching
    """
    args, output_dir = _init()
    show_n = 5000

    # Load the args
    with open(args.sf_path / 'args_sequence_fitter.yml', 'r') as f:
        sf_args = SequenceFitterArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))

    image_paths = get_image_paths(sf_args, load_all=True)[:show_n]

    # Check if pixel_to_mm is provided
    if args.pixel_to_mm is None:
        logger.info("No pixel-to-mm conversion factor provided. Calculating dynamically...")
        args.pixel_to_mm = calculate_distance(image_paths[0][1])

    # Output the pixel-to-mm conversion factor
    logger.info(f"Pixel-to-mm conversion factor: {args.pixel_to_mm:.4f} mm/pixel")

    scene = Scene.from_yml(sf_args.initial_scene)
    im = Image.open(image_paths[-1][1])
    width, height = im.size
    zoom = orthographic_scale_factor(scene)
    dist_to_mm = zoom * args.pixel_to_mm * min(width, height)
    # Get a crystal object
    TEST_CUBE = {
        'lattice_unit_cell': [1, 1, 1],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'point_group_symbol': '222',
        'scale': 1,
        'origin': [0, 0, 0],
        'distances': [1. / dist_to_mm, 1. / dist_to_mm, 1. / dist_to_mm],
        'rotation': [0., 0., 0.],
        'material_ior': 1.2,
        'material_roughness': 0.01
    }
    cube = Crystal(**TEST_CUBE['cube'])

    img = annotate_image(image_paths[-1][1],
                         crystal=cube,
                         zoom=zoom)
    plt.imshow(img)
    plt.savefig(output_dir / 'cube_test.png')
    plt.show()


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    # plot_test_cube()
    # plot_sequence_errors_for_sampler()
    # plot_3d_crystals()
    plot_distances_and_areas()
    # plot_3d_crystals_with_highlighted_faces()
    # plot_volume()
    plot_concentration()
    # create_legend_plot()
    # plot_sequence_errors()
