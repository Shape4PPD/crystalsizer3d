import argparse
import copy
import glob
import json
import os
import shutil
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
from torch import Tensor
from torchvision.transforms.functional import center_crop

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.args.refiner_args import DENOISER_ARG_NAMES, KEYPOINTS_ARG_NAMES, PREDICTOR_ARG_NAMES, RefinerArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.projector import Projector
from crystalsizer3d.refiner.denoising import denoise_image
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.sequence.plots import plot_areas, plot_distances, plot_losses, plot_origin, plot_rotation
from crystalsizer3d.sequence.utils import get_image_paths
from crystalsizer3d.util.kalman_filter import KalmanFilter
from crystalsizer3d.util.utils import FlexibleJSONEncoder, get_crystal_face_groups, hash_data, init_tensor, \
    json_to_numpy, print_args, str2bool, to_dict, to_numpy

ARG_NAMES = {
    'denoiser': DENOISER_ARG_NAMES,
    'keypoints': KEYPOINTS_ARG_NAMES,
    'predictor': PREDICTOR_ARG_NAMES
}

PARAMETER_KEYS = ['scale', 'distances', 'origin', 'rotation', 'material_roughness', 'material_ior', 'light_radiance']

refiner: Refiner = None


def get_args(printout: bool = True) -> Tuple[Namespace, RefinerArgs]:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to track crystal growth.')

    # Target sequence
    parser.add_argument('--args-path', type=Path,
                        help='Load refiner arguments from this path, any arguments set on the command-line will take preference.')
    parser.add_argument('--images-dir', type=Path,
                        help='Directory containing the sequence of images.')
    parser.add_argument('--image-ext', type=str, default='jpg',
                        help='Image extension.')
    parser.add_argument('--start-image', type=int, default=0,
                        help='Start processing from this image.')
    parser.add_argument('--end-image', type=int, default=-1,
                        help='End processing at this image.')
    parser.add_argument('--every-n-images', type=int, default=1,
                        help='Only process every N images.')
    parser.add_argument('--initial-scene', type=Path,
                        help='Path to the initial scene file. Will be used in place of the initial prediction.')
    parser.add_argument('--enforce-positive-growth', type=str2bool, default=False,
                        help='Enforce positive growth of the crystal distances.')

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

    # Remove the args_path argument as it is not needed
    delattr(args, 'args_path')

    return args, refiner_args


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
        destroy_predictor=True,
        do_init=False,
    )

    # Load the initial scene if provided
    scene_init = None
    if args.initial_scene is not None:
        assert args.initial_scene.exists(), f'Initial scene file {args.initial_scene} does not exist.'
        logger.info(f'Loading initial scene data from {args.initial_scene}.')
        scene_init = Scene.from_yml(args.initial_scene)

    # Initialise the Kalman filter for better temporal smoothing
    kalman_filter = KalmanFilter(
        param_dim=len(refiner.manager.ds.labels_distances),
        process_variance=1e-3,
        measurement_variance=1e-4
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
    applied_seq_args = False
    distances_est = None
    distances_min = None
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
                    data.append(json_to_numpy(json.load(f)))
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
            if i == 0:
                refiner.make_initial_prediction()

                # Update the parameters from the initial scene
                if scene_init is not None:
                    refiner.scene.crystal.copy_parameters_from(scene_init.crystal)
                    refiner.scene.light_radiance.data = init_tensor(scene_init.light_radiance)

                # Update the Kalman filter with the initial distances
                distances_actual = (refiner.scene.crystal.distances * refiner.scene.crystal.scale).detach()
                kalman_filter.update(distances_actual)

            # Otherwise, use the previous solution as the starting point
            else:
                refiner.set_initial_scene(scene_init)
                d_prev = scene_init.crystal.distances.clone().detach()

                # Predict the next state using the Kalman filter
                distances_est = kalman_filter.predict() / scene_init.crystal.scale.clone().detach()
                if i < 3:  # Use the previous values for the first few frames while filter is still learning
                    distances_est = d_prev

                # If enforcing positive growth, pass the previous frame's distances as the minimum
                distances_min = d_prev if args.enforce_positive_growth else None

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
            refiner.train(callback=after_refine_step, distances_est=distances_est, distances_min=distances_min)
            if distances_min is not None:
                assert torch.all(refiner.crystal.distances >= distances_min)

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

        # Update the Kalman filter
        distances_actual = torch.from_numpy(parameters_i['distances'][-1] * parameters_i['scale'][-1])
        kalman_filter.update(distances_actual)

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
                    data[key] = json_to_numpy(json.load(f))

    except Exception as e:
        logger.warning(f'Could not load data: {e}')
        data = None

    if data is None:
        if cache_only:
            raise RuntimeError(f'Cache could not be loaded!')
        logger.info('Processing crystal sequence.')
        data = _calculate_crystals(args, refiner_args, save_dir_seq, image_paths)

    return data


def _generate_images(
        args: Namespace,
        parameters: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path,
        overwrite: bool = False
):
    """
    Generate annotated, denoised and rendered images.
    """
    logger.info('Generating images.')
    global refiner
    wf_line_width = 3

    # Make the save directories
    save_dirs = {}
    for which in ['original', 'denoised', 'rendered']:
        save_dirs[which] = {}
        for annotated in [True, False]:
            if which == 'original' and not annotated:
                continue
            imgs_dir = save_dir / 'images' / (which + ('_annotated' if annotated else ''))
            if imgs_dir.exists():
                if not overwrite:
                    logger.warning(f'Images directory {imgs_dir} already exists. '
                                   f'Pass overwrite=True to overwrite.')
                    continue
                logger.info(f'Overwriting annotated images directory {imgs_dir}.')
                shutil.rmtree(imgs_dir)
            imgs_dir.mkdir(parents=True)
            save_dirs[which]['annotated' if annotated else 'not_annotated'] = imgs_dir
    if sum([len(v) for v in save_dirs.values()]) == 0:
        logger.info('All images already generated. Skipping.')
        return

    # Load the scene from the first image
    logger.info('Loading the scene from the first image.')
    idx0, img_path0 = image_paths[0]
    image_dir = Path(args.save_dir_seq) / 'refiner' / args.refiner_dir / f'image_{idx0:04d}'
    img0 = Image.open(img_path0)
    image_size = min(img0.size)
    if img0.size[0] != img0.size[1]:
        offset_l = (img0.size[0] - image_size) // 2
        offset_t = (img0.size[1] - image_size) // 2
    else:
        offset_l = 0
        offset_t = 0
    scene = Scene.from_yml(image_dir / 'scene.yml')
    crystal = scene.crystal

    # Set up the projector
    logger.info('Setting up the projector.')
    projector = Projector(
        crystal=crystal,
        image_size=(image_size, image_size),
        zoom=orthographic_scale_factor(scene),
        transparent_background=True,
        multi_line=True,
        rtol=1e-2
    )

    # Set up the denoiser
    refiner.manager.load_network(refiner.args.denoiser_model_path, 'denoiser')

    # Load all the image paths including ones which weren't optimised
    image_paths_all = get_image_paths(args, load_all=True)

    # Prepare the parameters - interpolating the missing data
    data = {}
    x_old = np.linspace(0, 1, len(image_paths))
    x_new = np.linspace(0, 1, len(image_paths_all))
    for key in PARAMETER_KEYS:
        param_data = parameters[key]
        if param_data.ndim == 1:
            param_data = param_data[:, None]
        new_data = np.zeros((len(image_paths_all), param_data.shape[1]))
        for i in range(param_data.shape[1]):
            f = interp1d(x_old, param_data[:, i], kind='cubic')
            new_data[:, i] = f(x_new)
        data[key] = new_data.squeeze()

    # Make the annotated images
    logger.info('Generating images.')
    for i, (idx, image_path) in enumerate(image_paths_all):
        if (i + 1) % 5 == 0:
            logger.info(f'Generating images for image {i + 1}/{len(image_paths_all)}.')

        # Update the crystal and scene parameters
        for key in PARAMETER_KEYS:
            if hasattr(crystal, key):
                getattr(crystal, key).data = init_tensor(data[key][i])
            elif hasattr(scene, key):
                getattr(scene, key).data = init_tensor(data[key][i])
            else:
                raise ValueError(f'Invalid key: {key}')

        # Rebuild the crystal mesh and project the wireframe
        crystal.build_mesh()
        projector.project(generate_image=False)

        # Load the image
        imgs = {}
        img = Image.open(image_path)
        img = img.convert('RGB')
        imgs['original'] = img

        # Denoise the image
        X = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255
        if X.shape[-2] != X.shape[-1]:
            X = center_crop(X, min(X.shape[-2:]))
        X_denoised = denoise_image(
            manager=refiner.manager,
            X=X,
            n_tiles=refiner.args.denoiser_n_tiles,
            overlap=refiner.args.denoiser_tile_overlap,
            oversize_input=refiner.args.denoiser_oversize_input,
            max_img_size=refiner.args.denoiser_max_img_size,
            batch_size=refiner.args.denoiser_batch_size,
            return_patches=False
        )
        img_denoised = Image.fromarray((to_numpy(X_denoised).transpose(1, 2, 0) * 255).astype(np.uint8))
        imgs['denoised'] = img_denoised

        # Render the scene
        scene.build_mi_scene()  # Rebuild the scene with updated crystal and lighting parameters
        img_rendered = scene.render()
        imgs['rendered'] = Image.fromarray(img_rendered).resize((image_size, image_size))

        # Save the non-annotated denoised and rendered images
        for img_type, img in imgs.items():
            if img_type == 'original':  # Don't save the original image as it's just a duplication
                continue
            if 'not_annotated' in save_dirs[img_type]:
                img.save(save_dirs[img_type]['not_annotated'] / f'image_{idx:04d}.png')

        # Draw wireframes onto the images
        for img_type, img in imgs.items():
            if 'annotated' not in save_dirs[img_type]:
                continue

            # Add offsets for original images as these aren't square
            if img_type == 'original':
                wf_ol = offset_l
                wf_ot = offset_t
            else:
                wf_ol = 0
                wf_ot = 0

            draw = ImageDraw.Draw(img, 'RGB')
            for ref_face_idx, face_segments in projector.edge_segments.items():
                if len(face_segments) == 0:
                    continue
                colour = projector.colour_facing_towards if ref_face_idx == 'facing' else projector.colour_facing_away
                colour = tuple((colour * 255).int().tolist())
                for segment in face_segments:
                    l = segment.clone()
                    l[:, 0] = torch.clamp(l[:, 0], 1, projector.image_size[1] - 2) + wf_ol
                    l[:, 1] = torch.clamp(l[:, 1], 1, projector.image_size[0] - 2) + wf_ot

                    draw.line(xy=[tuple(l[0].int().tolist()), tuple(l[1].int().tolist())],
                              fill=colour, width=wf_line_width)
            img.save(save_dirs[img_type]['annotated'] / f'image_{idx:04d}.png')


def _generate_manual_measurement_overlay_images(
        args: Namespace,
        measurements: Dict[str, np.ndarray],
        image_paths: List[Tuple[int, Path]],
        save_dir: Path,
        overwrite: bool = False,
        **kwargs
):
    """
    Generate images with the manual measurement projected overlays.
    """
    logger.info('Generating manual measurement images.')
    global refiner
    wf_line_width = 3

    # Make the save directories
    imgs_dir = save_dir / 'images' / 'original_annotated_manual'
    if imgs_dir.exists():
        if not overwrite:
            logger.warning(f'Images directory {imgs_dir} already exists. '
                           f'Pass overwrite=True to overwrite.')
            return
        logger.info(f'Overwriting annotated images directory {imgs_dir}.')
        shutil.rmtree(imgs_dir)
    imgs_dir.mkdir(parents=True)

    # Load the scene from the first image
    logger.info('Loading the scene from the first image.')
    idx0, img_path0 = image_paths[0]
    image_dir = Path(args.save_dir_seq) / 'refiner' / args.refiner_dir / f'image_{idx0:04d}'
    img0 = Image.open(img_path0)
    image_size = min(img0.size)
    if img0.size[0] != img0.size[1]:
        offset_l = (img0.size[0] - image_size) // 2
        offset_t = (img0.size[1] - image_size) // 2
    else:
        offset_l = 0
        offset_t = 0
    scene = Scene.from_yml(image_dir / 'scene.yml')
    crystal = scene.crystal

    # Set up the projector
    logger.info('Setting up the projector.')
    projector = Projector(
        crystal=crystal,
        image_size=(image_size, image_size),
        zoom=orthographic_scale_factor(scene),
        transparent_background=True,
        multi_line=True,
        rtol=1e-2
    )

    # Load all the image paths including ones which weren't optimised
    measurement_idxs = measurements['idx']
    pathspec = str(args.images_dir.absolute()) + '/*.' + args.image_ext
    all_image_paths = sorted(glob.glob(pathspec))
    image_paths = [(idx, Path(all_image_paths[idx])) for idx in measurement_idxs]

    # Make the annotated images
    logger.info('Generating images.')
    for i, (idx, image_path) in enumerate(image_paths):
        if (i + 1) % 5 == 0:
            logger.info(f'Generating images for image {i + 1}/{len(image_paths)}.')

        # Update the crystal parameters
        for key in PARAMETER_KEYS:
            if key in measurements and hasattr(crystal, key):
                getattr(crystal, key).data = init_tensor(measurements[key][i])

        # Rebuild the crystal mesh and project the wireframe
        crystal.build_mesh()
        projector.project(generate_image=False)

        # Load the image
        img = Image.open(image_path)
        img = img.convert('RGB')

        # Draw wireframe onto the image
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
        img.save(imgs_dir / f'image_{idx:04d}.png')


def _generate_videos(
        args: Namespace,
        save_dir_run: Path,
        overwrite: bool = False
):
    """
    Make videos of the growth sequences.
    """
    image_paths_all = get_image_paths(args, load_all=True)
    video_paths = []
    videos_dir = save_dir_run / 'videos'
    videos_dir.mkdir(exist_ok=True)
    for which in ['original', 'denoised', 'rendered']:
        for annotated in [False, True]:
            imgs_dir = save_dir_run / 'images' / (which + ('_annotated' if annotated else ''))
            save_path = videos_dir / f'{which}{"_annotated" if annotated else ""}.mp4'
            video_paths.append(str(save_path.absolute()))
            if save_path.exists() and not overwrite:
                logger.warning(f'Growth video {save_path} already exists. Pass overwrite=True to overwrite.')
                continue

            # Make a temporary copy of the original (non-annotated) images as we don't duplicate these
            if which == 'original' and not annotated:
                if imgs_dir.exists():
                    shutil.rmtree(imgs_dir)
                imgs_dir.mkdir(parents=True)
                for idx, img_path in image_paths_all:
                    Image.open(img_path).save(imgs_dir / f'image_{idx:04d}.png')

            # Generate the video
            logger.info(f'Making growth video from {imgs_dir} to {save_path}.')
            escaped_images_dir = str(imgs_dir.absolute()).replace('[', '\\[').replace(']', '\\]')
            cmd = f'ffmpeg -y -framerate 25 -pattern_type glob -i "{escaped_images_dir}/*.png" -c:v libx264 -pix_fmt yuv420p "{save_path}"'
            logger.info(f'Running command: {cmd}')
            os.system(cmd)

            # Remove the temporary copy of the original images
            if which == 'original' and not annotated:
                shutil.rmtree(imgs_dir)

    # Make manual measurements video
    imgs_dir = save_dir_run / 'images' / 'original_annotated_manual'
    if imgs_dir.exists():
        save_path = videos_dir / 'original_annotated_manual.mp4'
        logger.info(f'Making growth video from {imgs_dir} to {save_path}.')
        escaped_images_dir = str(imgs_dir.absolute()).replace('[', '\\[').replace(']', '\\]')
        cmd = f'ffmpeg -y -framerate 25 -pattern_type glob -i "{escaped_images_dir}/*.png" -c:v libx264 -pix_fmt yuv420p "{save_path}"'
        logger.info(f'Running command: {cmd}')
        os.system(cmd)

    # Make a composite video
    save_path = videos_dir / 'composite.mp4'
    if save_path.exists() and not overwrite:
        logger.warning(f'Composite growth video {save_path} already exists. Pass overwrite=True to overwrite.')
        return
    logger.info(f'Making composite growth video from {video_paths} to {save_path}.')
    panel_res = min(Image.open(image_paths_all[0][1]).size) // 3
    cmd = f'''
    ffmpeg -y -i {" -i ".join(video_paths)} \
    -filter_complex "
    [0:v]scale=-1:{panel_res}, crop={panel_res}:{panel_res}:x=(in_w-{panel_res})/2[v0];
    [1:v]scale=-1:{panel_res}, crop={panel_res}:{panel_res}:x=(in_w-{panel_res})/2[v1];
    [2:v]scale={panel_res}:{panel_res}[v2];
    [3:v]scale={panel_res}:{panel_res}[v3];
    [4:v]scale={panel_res}:{panel_res}[v4];
    [5:v]scale={panel_res}:{panel_res}[v5];
    [v0][v2][v4]hstack=inputs=3[top];
    [v1][v3][v5]hstack=inputs=3[bottom];
    [top][bottom]vstack=inputs=2
    " -c:v libx264 -pix_fmt yuv420p "{save_path}"
    '''
    logger.info(f'Running command: {cmd}')
    os.system(cmd)


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
        spec['refiner_dir'] = hash_data(to_dict(args))
        for model_name, arg_names in ARG_NAMES.items():
            model_args = {k: getattr(refiner_args, k) for k in arg_names}
            model_id = model_args[f'{model_name}_model_path'].stem[:4]
            spec[model_name + '_dir'] = f'{model_id}_{hash_data(model_args)}'
        yaml.dump(spec, f)

    # Generate or load the crystals
    image_paths = get_image_paths(args)
    data = _generate_or_load_crystals(args, refiner_args, save_dir_seq, image_paths, cache_only=False)

    # Instantiate a refiner if it isn't there already
    if refiner is None:
        refiner = Refiner(args=refiner_args, do_init=False)
    face_groups = get_crystal_face_groups(refiner.manager)

    # Make some plots
    plot_losses(data['losses_final'], data['losses_all'], image_paths, save_dir_run)
    plot_distances(data['parameters_final'], face_groups, image_paths, save_dir_run)
    plot_areas(refiner.manager, data['parameters_final'], image_paths, save_dir_run, face_groups)
    plot_origin(data['parameters_final'], image_paths, save_dir_run)
    plot_rotation(data['parameters_final'], image_paths, save_dir_run)

    # Generate images and videos
    if args.save_annotated_images:
        _generate_images(args, data['parameters_final'], image_paths, save_dir_run)
    if args.make_videos:
        _generate_videos(args, save_dir_run)

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
    parser.add_argument('--overwrite', type=str2bool, default=False, help='Overwrite existing images and videos.')
    parser.add_argument('--measurements-dir', type=Path, help='Path to a directory containing manual measurements.')
    runtime_args = parser.parse_args()
    run_dir = runtime_args.run_dir
    overwrite = runtime_args.overwrite
    assert run_dir.exists(), f'Run directory {run_dir} does not exist.'

    # Load the args from the run dir
    with open(run_dir / 'args.yml', 'r') as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    args = Namespace(**args_dict)
    args.images_dir = Path(args.images_dir)
    save_dir_seq = Path(args.save_dir_seq)
    assert save_dir_seq.exists(), f'Sequence save directory {save_dir_seq} does not exist.'
    image_paths = get_image_paths(args)

    # Load the data
    refiner_dir = save_dir_seq / 'refiner' / args.refiner_dir
    assert refiner_dir.exists(), f'Refiner output directory {refiner_dir} does not exist.'
    data = {}
    for name in ['losses', 'stats', 'parameters']:
        for all_or_final in ['all', 'final']:
            key = f'{name}_{all_or_final}'
            with open(refiner_dir / f'{name}_{all_or_final}.json', 'r') as f:
                data[key] = json_to_numpy(json.load(f))

    # Instantiate a refiner
    refiner_args = RefinerArgs.from_args(args)
    refiner = Refiner(args=refiner_args, do_init=False)
    mi_ref = refiner.manager.crystal.all_miller_indices

    # Load manual measurements
    measurements = None
    if runtime_args.measurements_dir is not None:
        assert runtime_args.measurements_dir.exists(), f'Measurements directory {runtime_args.measurements_dir} does not exist.'

        # Expect measurements dir to contain files like XXXX.json
        keys = ['idx', 'distances', 'scale', 'origin', 'rotation', 'material_roughness', 'material_ior']
        measurements = {k: [] for k in keys}
        for file_path in runtime_args.measurements_dir.iterdir():
            if file_path.suffix != '.json':
                continue
            crystal = Crystal.from_json(file_path)
            idx = int(file_path.stem)
            for k in keys:
                if k == 'idx':
                    measurements[k].append(idx)
                elif k == 'distances':
                    mi = crystal.all_miller_indices
                    mi_idxs = ((mi[None, ...] == mi_ref[:, None]).all(dim=2)).nonzero(as_tuple=True)[1]
                    measurements[k].append(to_numpy(crystal.distances[mi_idxs]))
                elif k == 'origin':
                    # Adjust the origin so that the crystal's smallest z-coordinate is at z=0
                    z_offset = crystal.vertices.amin(dim=0)[2]
                    measurements[k].append(to_numpy(crystal.origin - torch.tensor([0, 0, z_offset])))
                else:
                    measurements[k].append(to_numpy(getattr(crystal, k)))
        for k in keys:
            measurements[k] = np.array(measurements[k])

        # Fix the origin for the automatic measurements to match the manual measurements
        ds = refiner.manager.ds
        cs = ds.csd_proxy.load(ds.dataset_args.crystal_id)
        crystal = Crystal(
            lattice_unit_cell=cs.lattice_unit_cell,
            lattice_angles=cs.lattice_angles,
            miller_indices=ds.miller_indices,
            point_group_symbol=cs.point_group_symbol,
        )

        def adjust_distances(distances_old, origin_old):
            distances_new = np.zeros_like(distances_old)
            for i in range(len(origin_old)):
                crystal.distances.data = init_tensor(distances_old[i])
                crystal.origin.data = init_tensor(origin_old[i])
                crystal.adjust_origin(origin0, verify=False)
                distances_new[i] = to_numpy(crystal.distances)
            return distances_new

        # Use the origin from the first measured frame for all other frames
        origin0 = init_tensor(measurements['origin'][0])
        params = data['parameters_final']
        params['distances'] = adjust_distances(params['distances'], params['origin'])
        params['origin'] = np.repeat(origin0[None, ...], len(params['origin']), axis=0)

        # Ensure the measurements also all have the same origin
        if len(np.unique(measurements['origin'], axis=0)) != 1:
            measurements['distances'] = adjust_distances(measurements['distances'], measurements['origin'])

    # Make plots
    plot_losses(data['losses_final'], data['losses_all'], image_paths, run_dir)
    plot_args = dict(
        parameters=data['parameters_final'],
        measurements=measurements,
        image_paths=image_paths,
        save_dir=run_dir,
    )
    face_groups = get_crystal_face_groups(refiner.manager)
    plot_distances(face_groups=face_groups, **plot_args)
    plot_areas(face_groups=face_groups, **plot_args)
    plot_origin(**plot_args)
    plot_rotation(**plot_args)

    # Generate images and videos
    _generate_images(args, parameters=data['parameters_final'],
                     image_paths=image_paths, save_dir=run_dir, overwrite=overwrite)
    if measurements is not None:
        _generate_manual_measurement_overlay_images(args, overwrite=overwrite, **plot_args)
    _generate_videos(args, run_dir, overwrite=overwrite)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1][:9] == '--run-dir':
        plot_run()
    else:
        track_sequence()
