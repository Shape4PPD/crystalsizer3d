import argparse
import copy
import glob
import json
import os
import shutil
import sys
import time
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.gridspec import GridSpec
from torch import Tensor

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.util.utils import FlexibleJSONEncoder, hash_data, init_tensor, print_args, smooth_signal, str2bool, \
    to_dict, to_numpy, to_rgb

plot_extension = 'png'  # or svg

DENOISER_ARG_NAMES = [
    'denoiser_model_path', 'denoiser_n_tiles', 'denoiser_tile_overlap',
    'denoiser_oversize_input', 'denoiser_max_img_size', 'denoiser_batch_size',
]

KEYPOINTS_ARG_NAMES = [
    'keypoints_model_path', 'keypoints_oversize_input', 'keypoints_max_img_size', 'keypoints_batch_size',
    'keypoints_min_distance', 'keypoints_threshold', 'keypoints_exclude_border', 'keypoints_blur_kernel_relative_size',
    'keypoints_n_patches', 'keypoints_patch_size', 'keypoints_patch_search_res', 'keypoints_attenuation_sigma',
    'keypoints_max_attenuation_factor', 'keypoints_low_res_catchment_distance', 'keypoints_loss_type'
]

PREDICTOR_ARG_NAMES = [
    'predictor_model_path', 'initial_pred_noise_min', 'initial_pred_noise_max', 'initial_pred_oversize_input',
    'initial_pred_max_img_size', 'multiscale', 'use_keypoints', 'n_patches', 'w_img_l1', 'w_img_l2', 'w_perceptual',
    'w_latent', 'w_rcf', 'w_overshoot', 'w_symmetry', 'w_z_pos', 'w_rotation_xy', 'w_patches', 'w_fullsize',
    'w_switch_probs', 'w_keypoints', 'w_anchors', 'l_decay_l1', 'l_decay_l2', 'l_decay_perceptual', 'l_decay_latent',
    'l_decay_rcf', 'perceptual_model', 'latents_model', 'mv2_config_path', 'mv2_checkpoint_path', 'rcf_model_path',
    'rcf_loss_type', 'keypoints_loss_type'
]

ARG_NAMES = {
    'denoiser': DENOISER_ARG_NAMES,
    'keypoints': KEYPOINTS_ARG_NAMES,
    'predictor': PREDICTOR_ARG_NAMES
}

refiner: Refiner = None


def get_args(printout: bool = True) -> Tuple[Namespace, RefinerArgs]:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to track crystal growth.')

    # Target sequence
    parser.add_argument('--args-path', type=Path,
                        help='Load refiner arguments from this path, any arguments set on the command-line will take preference.')
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Directory containing the sequence of images.')
    parser.add_argument('--image-ext', type=str, default='jpg',
                        help='Image extension.')
    parser.add_argument('--start-image', type=int, default=0,
                        help='Start processing from this image.')
    parser.add_argument('--end-image', type=int, default=-1,
                        help='End processing at this image.')
    parser.add_argument('--every-n-images', type=int, default=1,
                        help='Only process every N images.')
    # parser.add_argument('--measurements-xls', type=Path,
    #                     help='Path to an xls file containing manual measurements.')

    # todo: add initial crystal fit from path

    # Refining settings
    RefinerArgs.add_args(parser)

    # Refining setting overrides for subsequent frames
    excluded_args = ['_path', 'ds_idx', 'denoiser_', 'initial_pred_', 'seed', '_model', 'img_size', 'wireframe', 'azim',
                     'elev', 'roll', '_colour']
    parser2 = ArgumentParser()
    RefinerArgs.add_args(parser2)
    for action in parser2._actions:
        if any([exc in action.dest for exc in excluded_args]) or isinstance(action, argparse._HelpAction):
            continue
        new_action = copy.deepcopy(action)
        new_action.dest = f'{action.dest}_seq'
        new_action.option_strings = [opt + '-seq' for opt in action.option_strings]
        new_action.default = None
        new_action.help = action.help + ' For subsequent frames.'
        parser._add_action(new_action)

    # Output
    parser.add_argument('--save-annotated-images', type=str2bool, default=True,
                        help='Save annotated images.')
    parser.add_argument('--save-annotated-masks', type=str2bool, default=True,
                        help='Save annotated masks.')
    parser.add_argument('--make-videos', type=str2bool, default=True,
                        help='Make video of the annotated masks/images (whatever was generated).')

    # Parse the command line arguments
    cli_args, _ = parser.parse_known_args()

    # Load any args from file and set these as the defaults for any arguments that weren't set
    if cli_args.args_path is not None:
        assert cli_args.args_path.exists(), f'Args path does not exist: {cli_args.args_path}'
        with open(cli_args.args_path, 'r') as f:
            args_yml = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**args_yml)

    # Parse the command line arguments again
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holder
    refiner_args = RefinerArgs.from_args(args)

    # Check arguments are valid
    assert args.images_dir.exists(), f'Images directory {args.images_dir} does not exist.'

    return args, refiner_args


def _json_to_numpy(data: Any) -> Any:
    """
    Convert json data to numpy arrays where possible.
    """
    if isinstance(data, dict):
        return {k: _json_to_numpy(v) for k, v in data.items()}
    if isinstance(data, list):
        types = [type(v) for v in data]
        if all([t in [int, float] for t in types]):
            return np.array(data)
        if all([t in [list, dict] for t in types]):
            n_entries = [len(v) for v in data]
            if len(set(n_entries)) == 1:
                return np.array(data)
        return [_json_to_numpy(v) for v in data]


def _get_image_paths(args: Namespace):
    """
    Load the images into an image sequence
    """
    pathspec = str(args.images_dir.absolute()) + '/*.' + args.image_ext
    all_image_paths = sorted(glob.glob(pathspec))
    image_paths = [(idx, Path(all_image_paths[idx])) for idx in range(
        args.start_image,
        len(all_image_paths) if args.end_image == -1 else args.end_image,
        args.every_n_images
    )]
    logger.info(f'Found {len(all_image_paths)} images in {args.images_dir}. Using {len(image_paths)} images.')
    return image_paths


def _calculate_crystals(
        args: Namespace,
        refiner_args: RefinerArgs,
        save_dir_seq: Path,
        image_paths: List[Tuple[int, Path]]
) -> Dict[str, Any]:
    """
    Calculate the crystals.
    """
    global refiner
    N = len(image_paths)

    # Ensure the model cache directories exists with the correct arguments
    cache_dirs = {}
    for model_name, arg_names in ARG_NAMES.items():
        model_args = {k: getattr(refiner_args, k) for k in arg_names}
        model_id = model_args[f'{model_name}_model_path'].stem[:4]
        model_cache_dir = save_dir_seq / model_name / f'{model_id}_{hash_data(model_args)}'
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dirs[model_name] = model_cache_dir
        with open(model_cache_dir / 'args.yml', 'w') as f:
            yaml.dump(model_args, f)

    # Create a refiner cache directory unique for the entire argument set - including refiner args and runtime args
    refiner_cache_dir = save_dir_seq / 'refiner' / hash_data(to_dict(args))
    refiner_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs['refiner'] = refiner_cache_dir
    with open(refiner_cache_dir / 'args.yml', 'w') as f:
        yaml.dump(to_dict(args), f)

    def _cache_paths(idx: int):
        active_cache = refiner.save_dir / 'cache'
        return [
            (cache_dirs['denoiser'] / f'X_target_denoised_{idx:05d}.pt', active_cache / 'X_target_denoised.pt'),
            (cache_dirs['keypoints'] / f'keypoints_{idx:05d}.pt', active_cache / 'keypoints.pt'),
            (cache_dirs['predictor'] / f'scene_{idx:05d}.yml', active_cache / 'scene.yml'),
            (cache_dirs['predictor'] / f'X_pred_{idx:05d}.pt', active_cache / 'X_pred.pt'),
        ]

    def _copy_existing_data(idx: int):
        for long_term, short_term in _cache_paths(idx):
            if long_term.exists():
                shutil.copy(long_term, short_term)

    def _add_to_cache(idx: int):
        for long_term, short_term in _cache_paths(idx):
            if short_term.exists():
                shutil.copy(short_term, long_term)

    # Instantiate the refiner
    idx0, img_path0 = image_paths[0]
    refiner_args.image_path = img_path0
    save_dir_image = refiner_cache_dir / f'image_{idx0:04d}'
    refiner = Refiner(
        args=refiner_args,
        output_dir=save_dir_image,
        destroy_denoiser=False,
        destroy_keypoint_detector=False,
        destroy_predictor=False,
        do_init=False,
    )

    # Instantiate the parameters and statistics logs
    losses_all = []
    stats_all = {}
    parameters_all = {}
    losses_final = []
    stats_final = {}
    parameters_final = {}

    def after_refine_step(step: int, loss: Tensor, stats: Dict[str, float]):
        losses_i.append(loss.item())
        for k, v in stats.items():
            if k not in stats_i:
                stats_i[k] = []
            stats_i[k].append(v)
        for k in parameters_i:
            if k == 'light_radiance':
                v = to_numpy(refiner.scene.light_radiance)
            elif k == 'conj_switch_probs':
                v = to_numpy(refiner.conj_switch_probs)
            else:
                v = to_numpy(getattr(refiner.crystal, k))
            parameters_i[k].append(v)

    # Iterate over the image sequence
    scene_init = None
    applied_seq_args = False
    for i, (idx, image_path) in enumerate(image_paths):
        logger.info(f'Processing image idx {idx} ({i + 1}/{N}): {image_path}.')
        refiner_args.image_path = image_path
        save_dir_image = refiner_cache_dir / f'image_{idx:04d}'

        # Check the refiner cache to see if this image has already been processed
        if save_dir_image.exists() and (save_dir_image / 'scene.yml').exists():
            logger.info(f'Image {idx} already processed. Loading data.')
            scene_init = Scene.from_yml(save_dir_image / 'scene.yml')

            # Load the losses, stats and parameters
            data = []
            for name in ['losses', 'stats', 'parameters']:
                with open(save_dir_image / f'{name}.json', 'r') as f:
                    data.append(json.load(f))
            losses_i, stats_i, parameters_i = data

        # Generate the result
        else:
            losses_i = []
            stats_i = {}
            parameters_i = {
                'distances': [],
                'origin': [],
                'scale': [],
                'rotation': [],
                'material_roughness': [],
                'material_ior': [],
                'light_radiance': [],
                'conj_switch_probs': []
            }

            # If the save dir exists then remove it as the crystal data is not there, so any data is invalid
            if save_dir_image.exists():
                shutil.rmtree(save_dir_image)

            # Update the refiner output directory
            refiner.init_save_dir(save_dir_image)
            refiner.init_tb_logger()

            # Copy over cached data if available
            _copy_existing_data(idx)

            # Initialise the targets - should use cache if available
            refiner.init_X_target()
            refiner.init_X_target_denoised()

            # Generate the keypoints if needed - should use cache if available
            if refiner_args.use_keypoints:
                refiner.keypoint_targets = None
                refiner.init_keypoint_targets()

            # Make the initial prediction - should use cache if available
            if scene_init is None:
                refiner.make_initial_prediction()

            # Otherwise, use the previous solution as the starting point
            else:
                refiner.set_initial_scene(scene_init)

                # Update any parameters that should be overridden for subsequent frames
                if not applied_seq_args:
                    for k in refiner_args.to_dict().keys():
                        k_seq = f'{k}_seq'
                        if hasattr(args, k_seq) and getattr(args, k_seq) is not None:
                            logger.info(f'Updating "{k}" with subsequent setting: '
                                        f'"{getattr(args, k_seq)}" (from "{getattr(refiner_args, k)}")')
                            setattr(refiner_args, k, getattr(args, k_seq))
                    applied_seq_args = True

            # Copy the data out of the run dir
            _add_to_cache(idx)

            # Refine the crystal fit
            refiner.step = 0
            refiner.train(callback=after_refine_step)

            # Rescale the distances and scales (note this doesn't canonicalise them, but comes close)
            distances_i = np.array(parameters_i['distances'])
            d_max = distances_i.max(axis=1)
            parameters_i['distances'] = distances_i / d_max[:, None]
            parameters_i['scale'] = np.array(parameters_i['scale']) * d_max

            # Canonicalise the final crystal morphology
            refiner.crystal.canonicalise()
            parameters_i['distances'][-1] = to_numpy(refiner.crystal.distances)
            parameters_i['scale'][-1] = to_numpy(refiner.crystal.scale)

            # Use the last crystal scene as the template for the next batch
            scene_init = refiner.scene

            # Store the result
            refiner.crystal.to_json(save_dir_image / 'crystal.json')
            refiner.scene.to_yml(save_dir_image / 'scene.yml')

            # Save the losses, stats and parameters
            for name, data in zip(['losses', 'stats', 'parameters'], [losses_i, stats_i, parameters_i]):
                with open(save_dir_image / f'{name}.json', 'w') as f:
                    json.dump(data, f, indent=2, cls=FlexibleJSONEncoder)

            # Clean up the directory
            shutil.rmtree(save_dir_image / 'cache')
            for subdir in ['initial_prediction', 'denoise_patches', 'keypoints', 'optimisation']:
                if not any((save_dir_image / subdir).iterdir()):
                    shutil.rmtree(save_dir_image / subdir)

        # Combine the losses, stats and parameters for this image into the sequence
        losses_all.append(losses_i)
        losses_final.append(losses_i[-1])
        for k, v in stats_i.items():
            if k not in stats_all:
                stats_all[k] = []
            if k not in stats_final:
                stats_final[k] = np.zeros((N,))
            stats_all[k].append(stats_i[k])
            stats_final[k][i] = stats_i[k][-1]
        for k, v in parameters_i.items():
            v = np.array(v)
            parameters_i[k] = v
            if k not in parameters_all:
                parameters_all[k] = []
            if k not in parameters_final:
                parameters_final[k] = np.zeros((N, *v[0].shape))
            parameters_all[k].append(v)
            parameters_final[k][i] = v[-1]

    # Save the final losses, stats and parameters
    data = {}
    for name, data_pair in zip(['losses', 'stats', 'parameters'],
                               [(losses_all, losses_final), (stats_all, stats_final),
                                (parameters_all, parameters_final)]):
        for all_or_final, datum in zip(['all', 'final'], data_pair):
            with open(refiner_cache_dir / f'{name}_{all_or_final}.json', 'w') as f:
                json.dump(datum, f, indent=2, cls=FlexibleJSONEncoder)
            data[f'{name}_{all_or_final}'] = datum

    return data


def _generate_or_load_crystals(
        args: Namespace,
        refiner_args: RefinerArgs,
        save_dir_seq: Path,
        image_paths: List[Tuple[int, Path]],
        cache_only: bool = False
) -> Dict[str, Any]:
    """
    Generate or load the crystal fits.
    """
    cache_dir = save_dir_seq / 'refiner' / hash_data(to_dict(args))

    try:
        data = {}
        for name in ['losses', 'stats', 'parameters']:
            for all_or_final in ['all', 'final']:
                key = f'{name}_{all_or_final}'
                with open(cache_dir / f'{name}_{all_or_final}.json', 'r') as f:
                    data[key] = _json_to_numpy(json.load(f))

    except Exception as e:
        logger.warning(f'Could not load data: {e}')
        data = None

    if data is None:
        if cache_only:
            raise RuntimeError(f'Cache could not be loaded!')
        logger.info('Processing crystal sequence.')
        data = _calculate_crystals(args, refiner_args, save_dir_seq, image_paths)

    return data


def _make_video(images_or_masks: str, save_dir: Path):
    """
    Make a video of the annotated images or masks.
    """
    assert images_or_masks in ['images', 'masks'], f'Invalid images_or_masks: {images_or_masks}'
    imgs_dir = save_dir / f'annotated_{images_or_masks}'
    save_path = save_dir / f'annotated_{images_or_masks}.mp4'
    logger.info(f'Making video of {images_or_masks} in {imgs_dir} to {save_path}.')
    cmd = f'ffmpeg -y -framerate 25 -pattern_type glob -i "{str(imgs_dir.absolute())}/*.png" -c:v libx264 -pix_fmt yuv420p "{save_path}"'
    logger.info(f'Running command: {cmd}')
    os.system(cmd)


def _get_crystal_groups(manager: Manager) -> Dict[Tuple[int, int, int], Dict[Tuple[int, int, int], int]]:
    """
    Get the crystal groups.
    """
    groups = {}
    for i, group_hkl in enumerate(manager.ds.dataset_args.miller_indices):
        group_idxs = (manager.crystal.symmetry_idx == i).nonzero().squeeze()
        groups[group_hkl] = OrderedDict([
            (tuple(manager.crystal.all_miller_indices[j].tolist()), int(j))
            for j in group_idxs
        ])
    return groups


def _get_face_group_colours(n_groups: int, cmap: str = 'turbo') -> np.ndarray:
    """
    Get a set of colours for the face groups.
    """
    return plt.get_cmap(cmap)(np.linspace(0, 1, n_groups))


def _get_colour_variations(
        base_colour: str,
        n_shades: int,
        hue_range: float = 0.1,
        sat_range: float = 0.2,
        val_range: float = 0.2
):
    """
    Generate a set of colour variations.
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
        variations.append(hsv_to_rgb(variation_hsv))

    return variations


def _get_hkl_label(hkl: np.ndarray, is_group: bool = False) -> str:
    """
    Get a LaTeX label for the hkl indices, replacing negative indices with overlines.
    """
    brackets = ['\{', '\}'] if is_group else ['(', ')']
    return f'${brackets[0]}' + ''.join([f'{mi}' if mi > -1 else f'\\bar{mi * -1}' for mi in hkl]) + f'{brackets[1]}$'


def _plot_losses(
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


def _plot_face_property_values(
        property_name: str,
        property_values: np.ndarray,
        image_paths: List[Tuple[int, Path]],
        save_dir: Path
):
    """
    Plot face distances or areas.
    """
    x_vals = [idx for idx, _ in image_paths]
    groups = _get_crystal_groups(refiner.manager)
    n_groups = len(groups)
    group_colours = _get_face_group_colours(n_groups)

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
    line_styles = ['--', '-.', ':']
    for i, (group_hkl, group_idxs) in enumerate(groups.items()):
        colour = group_colours[i]
        colour_variants = _get_colour_variations(colour, len(group_idxs))
        y = property_values[:, list(group_idxs.values())]
        y_mean = y.mean(axis=1)
        y_std = y.std(axis=1)
        lbls = [_get_hkl_label(hkl) for hkl in list(group_idxs.keys())]

        ax = fig.add_subplot(gs[i])
        ax.set_title(_get_hkl_label(group_hkl, is_group=True))

        # Plot the mean +/- std
        ax.plot(x_vals, y_mean, c=colour, label='Mean', zorder=1000)
        ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std, color=colour, alpha=0.1)

        # Plot the individual faces
        for j, (y_j, lbl, colour_j) in enumerate(zip(y.T, lbls, colour_variants)):
            ax.plot(x_vals, y_j, label=lbl, c=colour_j,
                    linestyle=line_styles[j % len(line_styles)], linewidth=0.8, alpha=0.8)

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
        ax.plot(x_vals, y_mean, c=colour, label=_get_hkl_label(group_hkl, is_group=True))
    ax.set_xlabel('Image index')
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_dir / f'{property_name}_mean.{plot_extension}')


def _plot_distances(
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path
):
    """
    Plot face distances.
    """
    distances = parameters['distances']
    scales = parameters['scale']
    _plot_face_property_values(
        property_name='distances',
        property_values=distances * scales[:, None],
        image_paths=image_paths,
        save_dir=save_dir
    )


def _plot_areas(
        parameters_final: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path
):
    """
    Plot face areas.
    """
    distances = parameters_final['distances']
    scales = parameters_final['scale']

    # Get a crystal object
    ds = refiner.manager.ds
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

    _plot_face_property_values(
        property_name='areas',
        property_values=areas,
        image_paths=image_paths,
        save_dir=save_dir
    )


def _plot_origin(
        parameters_final: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path
):
    """
    Plot origin position.
    """
    origins = parameters_final['origin']
    x_vals = [idx for idx, _ in image_paths]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.set_title('Origin position')
    ax.grid()
    for i in range(3):
        ax.plot(x_vals, origins[:, i], label='xyz'[i])
    ax.set_xlabel('Image index')
    ax.set_ylabel('Position')
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_dir / f'origin.{plot_extension}')


def _plot_rotation(
        parameters_final: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path
):
    """
    Plot rotation.
    """
    rotations = parameters_final['rotation']
    x_vals = [idx for idx, _ in image_paths]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.set_title('Axis-angle rotation vector components')
    ax.grid()
    for i in range(3):
        ax.plot(x_vals, rotations[:, i], label='$R_' + 'xyz'[i] + '$')
    ax.set_xlabel('Image index')
    ax.set_ylabel('Component value')
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_dir / f'rotation.{plot_extension}')


def track_sequence():
    """
    Track a sequence of crystals.
    """
    global refiner
    args, refiner_args = get_args()

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Make save directories
    save_dir_seq = LOGS_PATH / f'{args.images_dir.name}'
    run_dir_name = START_TIMESTAMP
    run_dir_name = 'START_TIMESTAMP'
    if args.start_image != 0 or args.end_image != -1 or args.every_n_images != 1:
        run_dir_name += f'_[{args.start_image}:{args.end_image}:{args.every_n_images}]'
    save_dir_run = save_dir_seq / 'runs' / run_dir_name
    if save_dir_run.exists():
        shutil.rmtree(save_dir_run)
    save_dir_run.mkdir(parents=True, exist_ok=True)

    # Save arguments to file
    with open(save_dir_run / 'args.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        spec['save_dir_seq'] = str(save_dir_seq)
        yaml.dump(spec, f)

    # Generate or load the crystals
    image_paths = _get_image_paths(args)
    data = _generate_or_load_crystals(args, refiner_args, save_dir_seq, image_paths, cache_only=False)

    # Instantiate a refiner if it isn't there already
    if refiner is None:
        refiner = Refiner(args=refiner_args, do_init=False)

    # Make some plots
    # _plot_losses(data['losses_final'], data['losses_all'], image_paths, save_dir_run)
    # _plot_distances(data['parameters_final'], image_paths, save_dir_run)
    _plot_areas(data['parameters_final'], image_paths, save_dir_run)

    # Print how long this took - split into hours, minutes, seconds
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'Finished in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.')


def plot_run():
    """
    Load an existing run and plot results.
    """
    global refiner
    parser = ArgumentParser(description='CrystalSizer3D script to track crystal growth.')
    parser.add_argument('--run-dir', type=Path, help='Directory containing the run data.')
    run_dir = parser.parse_known_args()[0].run_dir
    assert run_dir.exists(), f'Run directory {run_dir} does not exist.'

    # Load the args from the run dir
    with open(run_dir / 'args.yml', 'r') as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    args = Namespace(**args_dict)
    args.images_dir = Path(args.images_dir)
    refiner_args = RefinerArgs.from_args(args)
    save_dir_seq = LOGS_PATH / f'{args.images_dir.name}'
    # save_dir_seq = Path(args['save_dir_seq'])
    assert save_dir_seq.exists(), f'Sequence save directory {save_dir_seq} does not exist.'
    image_paths = _get_image_paths(args)

    # Load the data
    data = {}
    for name in ['losses', 'stats', 'parameters']:
        for all_or_final in ['all', 'final']:
            key = f'{name}_{all_or_final}'
            with open(run_dir / f'{name}_{all_or_final}.json', 'r') as f:
                data[key] = _json_to_numpy(json.load(f))

    # Instantiate a refiner
    refiner = Refiner(args=refiner_args, do_init=False)

    # Make plots
    _plot_losses(data['losses_final'], data['losses_all'], image_paths, run_dir)
    _plot_distances(data['parameters_final'], image_paths, run_dir)
    _plot_areas(data['parameters_final'], image_paths, run_dir)
    _plot_origin(data['parameters_final'], image_paths, run_dir)
    _plot_rotation(data['parameters_final'], image_paths, run_dir)

    # Make annotated images and videos
    # todo


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1][:9] == '--run-dir':
        plot_run()
    else:
        track_sequence()
