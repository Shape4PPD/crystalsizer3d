import glob
import json
import os
import shutil
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from pims import ImageSequence
from torch import Tensor

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.util.utils import FlexibleJSONEncoder, hash_data, print_args, str2bool, to_dict, to_numpy

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
    'predictor_model_path', 'initial_pred_from', 'initial_pred_noise_min', 'initial_pred_noise_max',
    'working_image_size', 'multiscale', 'use_keypoints', 'n_patches', 'w_img_l1', 'w_img_l2', 'w_perceptual',
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


def get_args(printout: bool = True) -> Tuple[Namespace, RefinerArgs]:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to track crystal growth.')

    # Target
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

    # todo: add initial crystal fit

    # Refining settings
    RefinerArgs.add_args(parser)

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


def _load_measurements(image_seq: ImageSequence, args: Namespace) -> Dict[int, List[float]]:
    """
    Load a spreadsheet of measurements if available.
    """
    measurements = {}
    xls = args.measurements_xls
    if xls is None:
        # Try to find the measurements file in the parent directory of the images directory
        xls = args.images_dir.parent / 'measurements.xlsx'
        if not xls.exists():
            xls = None
    if xls is not None:
        if not xls.is_absolute():
            xls = args.images_dir / xls
        assert xls.exists(), \
            f'Measurements file {xls} does not exist.'
        assert xls.suffix in ['.xls', '.xlsx'], \
            f'Expecting an xls or xlsx file, not {xls}.'
        logger.info(f'Loading measurements from {xls}.')
        xl_file = pd.ExcelFile(xls)
        sheet_names = {n: n for n in xl_file.sheet_names}
        for sn in xl_file.sheet_names:
            if 'R' in sn and 'Repeat' not in sn:
                sheet_names[sn.replace('R', 'Repeat ')] = sn
        if args.images_dir.name in sheet_names:
            df = xl_file.parse(sheet_names[args.images_dir.name])
        elif len(sheet_names) == 1:
            df = xl_file.parse(xl_file.sheet_names[0])
        else:
            raise RuntimeError(f'Multiple sheets in xls and none match the images directory: "{args.images_dir.name}".')

        # Extract measurements
        for i, fp in enumerate(image_seq._filepaths):
            p = Path(fp).name
            res = df[df['File'] == p]
            if len(res) == 1:
                row = res.values[0]
                measurements[i] = [row[1], row[2], row[3]]

    return measurements


def _calculate_crystals(
        args: Namespace,
        refiner_args: RefinerArgs,
        save_dir_seq: Path,
        save_dir_run: Path,
        image_paths: List[Tuple[int, Path]]
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Calculate the crystals.
    """
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
            # (cache_dirs['keypoints'] / f'X_keypoints_{idx:05d}.pt', active_cache / f'X_keypoints.pt'),
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
    for i, (idx, image_path) in enumerate(image_paths):
        logger.info(f'Processing image idx {idx} ({i + 1}/{N}): {image_path}.')
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

            # Copy the data out of the run dir
            _add_to_cache(idx)

            # Refine the crystal fit
            refiner.step = 0
            refiner.train(callback=after_refine_step)

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
            Path(refiner.tb_logger.file_writer.event_writer._file_name).unlink()

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
    for name, data_pair in zip(['losses', 'stats', 'parameters'],
                               [(losses_all, losses_final), (stats_all, stats_final),
                                (parameters_all, parameters_final)]):
        for all_or_final, data in zip(['all', 'final'], data_pair):
            with open(save_dir_run / f'{name}_{all_or_final}.json', 'w') as f:
                json.dump(data, f, indent=2, cls=FlexibleJSONEncoder)

    # Make some plots
    fig, axes = plt.subplots(2, figsize=(12, 8))
    ax = axes[0]
    ax.set_title('Final losses')
    ax.plot(losses_final)
    ax.set_xlabel('Image index')
    ax.set_xticks(range(N))
    ax.set_xticklabels([str(idx) for idx, _ in image_paths])
    ax.set_ylabel('Loss')

    # Loss convergence per image
    ax = axes[1]
    cm = plt.get_cmap('tab20')
    ax.set_title('Loss convergence')
    for i, v in enumerate(losses_all):
        ax.plot(v, label=image_paths[i][0])
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    plt.savefig(save_dir_run / 'losses.png')

    # Plot distances
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.set_title('Distances')
    for i in range(parameters_final['distances'].shape[1]):
        v = parameters_final['distances'][:, i]
        ax.plot(v, label=i)
    ax.legend()
    ax.set_xlabel('Image index')
    ax.set_xticks(range(N))
    ax.set_xticklabels([str(idx) for idx, _ in image_paths])
    ax.set_ylabel('Distance')
    plt.savefig(save_dir_run / 'distances.png')

    # plt.show()

    exit()

    return crystals, losses


def _generate_or_load_crystals(
        args: Namespace,
        refiner_args: RefinerArgs,
        save_dir_seq: Path,
        save_dir_run: Path,
        image_paths: List[Tuple[int, Path]],
        cache_only: bool = False
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Generate or load the hexagons.
    """
    args_hash = hash_data(to_dict(args))
    cache_dir = save_dir_seq / f'args={args_hash}'

    parameters = None
    # todo: what are we caching??
    # cache_path = HEXAGONS_CACHE_DIR / f'{args.images_dir.name}_{hash_str}'
    # cache_fn = cache_path.with_suffix(cache_path.suffix + '.npz')
    # if not args.rebuild_crystals_cache and cache_fn.exists():
    #     try:
    #         data = np.load(cache_fn)
    #         crystals = data['hexagons']
    #         losses = {}
    #         for k in data.files:
    #             if k.startswith('losses_'):
    #                 losses[k[7:]] = list(data[k])
    #         logger.info(f'Loaded hexagons from cache: {cache_fn}')
    #     except Exception as e:
    #         logger.warning(f'Could not load cache: {e}')

    if parameters is None:
        if cache_only:
            raise RuntimeError(f'Cache could not be loaded!')
            # raise RuntimeError(f'Cache "{cache_fn}" could not be loaded!')
        logger.info('Processing crystal sequence.')
        parameters, losses = _calculate_crystals(args, refiner_args, save_dir_seq, save_dir_run, image_paths)
        save_arrs = {'crystals': crystals}
        for k, l in losses.items():
            save_arrs[f'losses_{k}'] = np.array(l)
        logger.info(f'Saving crystals data to {cache_path}.')
        np.savez(cache_path, **save_arrs)

    return crystals, losses


def _save_parameters_to_csv(image_seq: ImageSequence, hexagons: np.ndarray, save_dir: Path):
    """
    Save the hexagon parameters to csv.
    """
    save_path = save_dir / f'hexagon_parameters.csv'
    logger.info(f'Saving hexagon parameters to {save_path}.')
    data = hexagons.copy()

    # Take angles in the range [0, pi]
    data[:, 2:5] = np.mod(data[:, 2:5], np.pi)

    # Calculate face distances (sum of length pairs)
    lengths = data[:, 5:11].reshape(-1, 3, 2)
    distances = lengths.sum(axis=-1)
    data = np.concatenate([data, distances], axis=-1)

    # Take absolute temperatures
    data[:, 11] = np.abs(data[:, 11])

    # Add the filenames as the first column
    filenames = np.array([Path(fp).name for fp in image_seq._filepaths])
    data = np.concatenate([filenames[:, None], data], axis=-1)

    # Append distances to hexagons and write to file
    np.savetxt(save_path, data, '%s', delimiter=',',
               header='filename,x,y,a0,a1,a2,l0a,l0b,l1a,l1b,l2a,l2b,temp,d0,d1,d2')


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


def track_sequence():
    """
    Track a sequence of crystals.
    """
    args, refiner_args = get_args()

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Make save directories
    save_dir_seq = LOGS_PATH / f'{args.images_dir.name}'
    save_dir_run = save_dir_seq / 'runs' / f'{START_TIMESTAMP}'
    run_dir_name = f'{START_TIMESTAMP}'
    # run_dir_name = f'timestamp_tmp'
    if args.start_image != 0 or args.end_image != -1 or args.every_n_images != 1:
        run_dir_name += f'_[{args.start_image}:{args.end_image}:{args.every_n_images}]'
    save_dir_run = save_dir_seq / 'runs' / run_dir_name
    if save_dir_run.exists():
        shutil.rmtree(save_dir_run)
    save_dir_run.mkdir(parents=True, exist_ok=True)

    # Save arguments to file
    with open(save_dir_run / 'options.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    # Load the images into an image sequence
    pathspec = str(args.images_dir.absolute()) + '/*.' + args.image_ext
    all_image_paths = sorted(glob.glob(pathspec))
    image_paths = [(idx, Path(all_image_paths[idx])) for idx in range(
        args.start_image,
        len(all_image_paths) if args.end_image == -1 else args.end_image,
        args.every_n_images
    )]
    logger.info(f'Found {len(all_image_paths)} images in {args.images_dir}. Using {len(image_paths)} images.')

    # Generate or load the crystals
    parameters = _generate_or_load_crystals(args, refiner_args, save_dir_seq, save_dir_run, image_paths,
                                            cache_only=False)

    # # Load a spreadsheet of measurements if available
    # try:
    #     measurements = _load_measurements(image_seq, args)
    # except RuntimeError as e:
    #     logger.warning(f'Failed to load measurements: {e}')
    #     measurements = {}

    # Save the data to csv
    _save_parameters_to_csv(image_seq, parameters, save_dir)

    # Plot the parameters
    # todo

    # Print how long this took - split into hours, minutes, seconds
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'Finished in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.')


if __name__ == '__main__':
    track_sequence()
