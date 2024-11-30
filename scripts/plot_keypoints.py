import time
from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from torchvision.transforms.functional import center_crop, gaussian_blur, to_tensor

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.refiner.denoising import denoise_image, stitch_image, tile_image
from crystalsizer3d.refiner.keypoint_detection import detect_keypoints_batch, find_keypoints, to_absolute_coordinates
from crystalsizer3d.util.image_helpers import save_img, save_img_grid, save_img_with_keypoint_markers, \
    save_img_with_keypoint_markers2, save_img_with_keypoint_overlay, threshold_img
from crystalsizer3d.util.utils import print_args, str2bool, to_dict, to_numpy

dpi = plt.rcParams['figure.dpi']


class RuntimeArgs(BaseArgs):
    def __init__(
            self,

            # Target settings
            image_path: Optional[Path] = None,
            ds_idx: int = 0,
            batch_size: int = 1,

            # Denoising settings
            denoiser_model_path: Optional[Path] = None,
            denoiser_n_tiles: int = 4,
            denoiser_tile_overlap: float = 0.1,
            denoiser_oversize_input: bool = False,
            denoiser_max_img_size: int = 1024,
            denoiser_batch_size: int = 4,

            # Keypoint detection settings
            keypoints_model_path: Optional[Path] = None,
            keypoints_oversize_input: bool = False,
            keypoints_max_img_size: int = 1024,
            keypoints_batch_size: int = 4,
            keypoints_min_distance: int = 5,
            keypoints_threshold: float = 0.5,
            keypoints_exclude_border: float = 0.05,

            # Tiling method
            keypoints_pred_from: str = 'denoised',
            keypoints_n_tiles: int = 4,
            keypoints_tile_overlap: float = 0.1,

            # Multi-crop targeted method
            keypoints_blur_kernel_relative_size: float = 0.01,
            keypoints_n_patches: int = 16,
            keypoints_patch_size: int = 700,
            keypoints_patch_search_res: int = 256,
            keypoints_attenuation_sigma: float = 0.5,
            keypoints_max_attenuation_factor: float = 1.5,
            keypoints_low_res_catchment_distance: int = 100,

            # Plot settings
            img_size_3d: int = 400,
            wireframe_r_factor: float = 0.2,
            surface_colour_target: str = 'orange',
            wireframe_colour_target: str = 'darkorange',
            surface_colour_pred: str = 'skyblue',
            wireframe_colour_pred: str = 'cornflowerblue',
            plot_colour_target: str = 'red',
            plot_colour_pred: str = 'darkblue',

            **kwargs
    ):
        # Target settings
        if image_path is not None:
            assert image_path.exists(), f'Image path does not exist: {image_path}'
        self.image_path = image_path
        self.ds_idx = ds_idx
        self.batch_size = batch_size

        # Denoising settings
        if keypoints_pred_from == 'denoised':
            assert denoiser_model_path is not None, 'Denoiser model path must be set to predict keypoints from denoised images.'
        if denoiser_model_path is not None:
            assert denoiser_model_path.exists(), f'DN model path does not exist: {denoiser_model_path}'
        self.denoiser_model_path = denoiser_model_path
        self.denoiser_n_tiles = denoiser_n_tiles
        self.denoiser_tile_overlap = denoiser_tile_overlap
        self.denoiser_oversize_input = denoiser_oversize_input
        self.denoiser_max_img_size = denoiser_max_img_size
        self.denoiser_batch_size = denoiser_batch_size

        # Keypoint detection settings
        assert keypoints_model_path.exists(), f'Keypoints model path does not exist: {keypoints_model_path}'
        self.keypoints_model_path = keypoints_model_path
        self.keypoints_oversize_input = keypoints_oversize_input
        self.keypoints_max_img_size = keypoints_max_img_size
        self.keypoints_batch_size = keypoints_batch_size
        self.keypoints_min_distance = keypoints_min_distance
        self.keypoints_threshold = keypoints_threshold
        self.keypoints_exclude_border = keypoints_exclude_border

        # Tiling method
        self.keypoints_pred_from = keypoints_pred_from
        self.keypoints_n_tiles = keypoints_n_tiles
        self.keypoints_tile_overlap = keypoints_tile_overlap

        # Multi-crop targeted method
        self.keypoints_blur_kernel_relative_size = keypoints_blur_kernel_relative_size
        self.keypoints_n_patches = keypoints_n_patches
        self.keypoints_patch_size = keypoints_patch_size
        self.keypoints_patch_search_res = keypoints_patch_search_res
        self.keypoints_attenuation_sigma = keypoints_attenuation_sigma
        self.keypoints_max_attenuation_factor = keypoints_max_attenuation_factor
        self.keypoints_low_res_catchment_distance = keypoints_low_res_catchment_distance

        # Plot settings
        self.img_size_3d = img_size_3d
        self.wireframe_r_factor = wireframe_r_factor
        self.surface_colour_target = surface_colour_target
        self.wireframe_colour_target = wireframe_colour_target
        self.surface_colour_pred = surface_colour_pred
        self.wireframe_colour_pred = wireframe_colour_pred
        self.plot_colour_target = plot_colour_target
        self.plot_colour_pred = plot_colour_pred

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Runtime Args')
        parser.add_argument('--args-path', type=Path,
                            help='Load arguments from this path, any arguments set on the command-line will take preference.')

        # Target settings
        group.add_argument('--image-path', type=Path,
                           help='Path to the image to process. If set, will override the dataset entry.')
        group.add_argument('--ds-idx', type=int, default=0,
                           help='Index of the dataset entry to use.')
        group.add_argument('--batch-size', type=int, default=16,
                           help='Batch size for a noisy prediction batch.')

        # Denoising settings
        group.add_argument('--denoiser-model-path', type=Path,
                           help='Path to the denoising model\'s json file.')
        group.add_argument('--denoiser-n-tiles', type=int, default=9,
                           help='Number of tiles to split the image into for denoising.')
        group.add_argument('--denoiser-tile-overlap', type=float, default=0.05,
                           help='Ratio of overlap between tiles for denoising.')
        group.add_argument('--denoiser-oversize-input', type=str2bool, default=False,
                           help='Whether to resize the input images to the maximum image size.')
        group.add_argument('--denoiser-max-img-size', type=int, default=1024,
                           help='Maximum image size for denoising.')
        group.add_argument('--denoiser-batch-size', type=int, default=3,
                           help='Number of tiles to denoise at a time.')

        # Keypoint detection settings
        group.add_argument('--keypoints-model-path', type=Path,
                           help='Path to the keypoints model checkpoint.')
        group.add_argument('--keypoints-oversize-input', type=str2bool, default=False,
                           help='Whether to resize the input images to the maximum image size for keypoints detection.')
        group.add_argument('--keypoints-max-img-size', type=int, default=1024,
                           help='Maximum image size for keypoint detection.')
        group.add_argument('--keypoints-batch-size', type=int, default=3,
                           help='Number of tiles to detect keypoints at a time.')
        group.add_argument('--keypoints-min-distance', type=int, default=5,
                           help='Minimum pixel distance between keypoints.')
        group.add_argument('--keypoints-threshold', type=float, default=0.5,
                           help='Threshold for keypoints detection.')
        group.add_argument('--keypoints-exclude-border', type=float, default=0.05,
                           help='Exclude keypoints within this ratio of the border.')

        # Tiling method
        group.add_argument('--keypoints-pred-from', type=str, default='denoised', choices=['denoised', 'original'],
                           help='Calculate the keypoints from either the original or denoised image.')
        group.add_argument('--keypoints-n-tiles', type=int, default=9,
                           help='Number of tiles to split the image into for keypoints detection.')
        group.add_argument('--keypoints-tile-overlap', type=float, default=0.05,
                           help='Ratio of overlap between tiles for keypoints detection.')

        # Multi-crop targeted method
        group.add_argument('--keypoints-blur-kernel-relative-size', type=float, default=0.01,
                           help='Relative size of the blur kernel for initial keypoints detection from low res, denoised image.')
        group.add_argument('--keypoints-n-patches', type=int, default=16,
                           help='Number of patches to crop from the image for high res keypoints detection.')
        group.add_argument('--keypoints-patch-size', type=int, default=700,
                           help='Size of the crop patches.')
        group.add_argument('--keypoints-patch-search-res', type=int, default=256,
                           help='Resolution of the low-res keypoints heatmap to use for determining where to crop the patches.')
        group.add_argument('--keypoints-attenuation-sigma', type=float, default=0.5,
                           help='Sigma parameter for the Gaussian blob used to iteratively attenuate the keypoints heatmap.')
        group.add_argument('--keypoints-max-attenuation-factor', type=float, default=1.5,
                           help='Maximum Gaussian peak height used for the attenuation function.')
        group.add_argument('--keypoints-low-res-catchment-distance', type=int, default=100,
                           help='Catchment distance (in pixels) for high res keypoints from the original low res keypoints.')

        # Plot settings
        group.add_argument('--img-size-3d', type=int, default=400,
                           help='Size of the 3D digital crystal image.')
        group.add_argument('--wireframe-r-factor', type=float, default=0.2,
                           help='Wireframe radius factor, multiplied by the maximum dimension of the bounding box to calculate the final edge tube radius.')
        group.add_argument('--surface-colour-target', type=str, default='orange',
                           help='Target mesh surface colour.')
        group.add_argument('--wireframe-colour-target', type=str, default='darkorange',
                           help='Target mesh wireframe colour.')
        group.add_argument('--surface-colour-pred', type=str, default='skyblue',
                           help='Predicted mesh surface colour.')
        group.add_argument('--wireframe-colour-pred', type=str, default='cornflowerblue',
                           help='Predicted mesh wireframe colour.')
        group.add_argument('--plot-colour-target', type=str, default='darkorange',
                           help='Target parameters plot colour.')
        group.add_argument('--plot-colour-pred', type=str, default='cornflowerblue',
                           help='Predicted parameters plot colour.')

        return group


def parse_arguments(printout: bool = True) -> RuntimeArgs:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Detect keypoints in crystal images.')
    RuntimeArgs.add_args(parser)

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
    runtime_args = RuntimeArgs.from_args(args)

    return runtime_args


def _init(args: Optional[RuntimeArgs], method: str):
    if args is None:
        args = parse_arguments()

    # Create an output directory
    if args.image_path is None:
        target_str = f'ds_idx={args.ds_idx}'
    else:
        target_str = args.image_path.stem
    dir_name = f'{START_TIMESTAMP}_{args.keypoints_model_path.stem[:4]}_{target_str}'
    # dir_name = f'timestamp_{args.keypoints_model_path.stem[:4]}_{target_str}'
    if method[-6:] == '_batch':
        dir_name += f'_bs={args.batch_size}'
    save_dir = LOGS_PATH / method / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments to json file
    with open(save_dir / 'args.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    # Instantiate the manager from the checkpoint json path
    manager = Manager.load(
        model_path=args.keypoints_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': USE_CUDA,
                'batch_size': 1
            },
        },
        save_dir=save_dir
    )
    assert manager.keypoint_detector is not None, 'No keypoint detector found.'
    manager.enable_eval()  # Should be on as default, but set it just in case
    torch.set_grad_enabled(False)

    # Load the input image (and parameters if loading from the dataset)
    if args.image_path is None:
        item, image, image_clean, Y_target = manager.ds.load_item(args.ds_idx)
        Y_target = {
            k: torch.from_numpy(v).to(torch.float32).to(manager.device)
            for k, v in Y_target.items()
        }
        r_params_target = item['rendering_parameters']

        if manager.keypoint_detector_args.kd_use_clean_images:
            X_target = image_clean
        else:
            X_target = image
        X_target = to_tensor(X_target)

    else:
        X_target = to_tensor(Image.open(args.image_path))
        if X_target.shape[0] == 4:
            assert torch.allclose(X_target[3], torch.ones_like(X_target[3])), 'Transparent images not supported.'
            X_target = X_target[:3]
        Y_target = None
        r_params_target = None

        # Crop and resize the image to the working image size
        d = min(X_target.shape[-2:])
        X_target = center_crop(X_target, d)
        # X_target = crop(X_target, top=0, left=X_target.shape[-1] - d, height=d, width=d)
    X_target = X_target.to(manager.device)

    # Denoise the input image if required
    if args.keypoints_pred_from == 'denoised':
        assert args.denoiser_model_path is not None, 'Denoiser model path must be set to predict keypoints from denoised images.'
        manager.load_network(args.denoiser_model_path, 'denoiser')

        # Denoise the image
        logger.info('Denoising the image.')
        X_target_denoised = denoise_image(
            manager,
            X_target,
            n_tiles=args.denoiser_n_tiles,
            overlap=args.denoiser_tile_overlap,
            oversize_input=args.denoiser_oversize_input,
            max_img_size=args.denoiser_max_img_size,
            batch_size=args.denoiser_batch_size
        )
    else:
        X_target_denoised = None

    return args, manager, save_dir, X_target, X_target_denoised, Y_target, r_params_target


def plot_keypoints(args: Optional[RuntimeArgs] = None):
    """
    Plot the keypoints for a given image.
    """
    args, manager, save_dir, X_target, X_target_denoised, Y_target, r_params_target = _init(args, 'keypoints')
    resize_args = dict(mode='bilinear', align_corners=False)
    manager.keypoint_detector_args.kd_resize_input = False  # We'll manually resize the images
    img_size = args.keypoints_max_img_size if args.keypoints_oversize_input else manager.image_shape[-1]
    X = {
        'original': to_numpy(X_target),
        'denoised': to_numpy(X_target_denoised)
    }

    # Detect keypoints for original and denoised complete images
    logger.info('Detecting keypoints in complete image.')
    X_batch = X_target[None, ...]
    if X_target_denoised is not None:
        X_batch = torch.cat([X_batch, X_target_denoised[None, ...]])
    restore_size = False
    if args.keypoints_oversize_input and X_batch.shape[-1] > img_size \
            or not args.keypoints_oversize_input and X_batch.shape[-1] != img_size:
        restore_size = X_batch.shape[-1]
        X_batch = F.interpolate(X_batch, size=img_size, **resize_args)
    X_kp, _ = manager.detect_keypoints(X_batch)
    if restore_size:
        X_kp = F.interpolate(X_kp[:, None], size=restore_size, **resize_args)[:, 0]
    X_kp = {
        'original': to_numpy(X_kp[0]),
        'denoised': to_numpy(X_kp[1]) if X_target_denoised is not None else None
    }
    X_kp_thresh = {
        'original': threshold_img(X_kp['original'], args.keypoints_threshold),
        'denoised': threshold_img(X_kp['denoised'], args.keypoints_threshold)
        if X_target_denoised is not None else None
    }

    # Tile the image and denoised image into overlapping patches
    tile_args = dict(n_tiles=args.keypoints_n_tiles, overlap=args.keypoints_tile_overlap)
    X_og_patches, patch_positions = tile_image(X_target, **tile_args)
    if X_target_denoised is not None:
        X_dn_patches, _ = tile_image(X_target_denoised, **tile_args)
    X_patches = {
        'original': to_numpy(X_og_patches),
        'denoised': to_numpy(X_dn_patches) if X_target_denoised is not None else None
    }

    # Detect keypoints in the image patches
    logger.info('Detecting keypoints in image patches.')
    X_batch = X_patches['original']
    if X_target_denoised is not None:
        X_batch = torch.cat([X_og_patches, X_dn_patches])
    restore_size = False
    if args.keypoints_oversize_input and X_batch.shape[-1] > img_size \
            or not args.keypoints_oversize_input and X_batch.shape[-1] != img_size:
        restore_size = X_batch.shape[-1]
        X_batch = F.interpolate(X_batch, size=img_size, **resize_args)
    X_patches_kp = detect_keypoints_batch(manager, X_batch, batch_size=args.keypoints_batch_size)
    if restore_size:
        X_patches_kp = F.interpolate(X_patches_kp[:, None], size=restore_size, **resize_args)[:, 0]
    X_og_patches_kp = X_patches_kp[:len(X_og_patches)]
    if X_target_denoised is not None:
        X_dn_patches_kp = X_patches_kp[len(X_og_patches):]
    X_patches_kp = {
        'original': to_numpy(X_og_patches_kp),
        'denoised': to_numpy(X_dn_patches_kp) if X_target_denoised is not None else None
    }
    X_patches_kp_thresh = {
        'original': threshold_img(X_patches_kp['original'], args.keypoints_threshold),
        'denoised': threshold_img(X_patches_kp['denoised'], args.keypoints_threshold)
        if X_target_denoised is not None else None
    }

    # Stitch the patches back together
    stitch_args = dict(patch_positions=patch_positions, overlap=args.keypoints_tile_overlap)
    X_kp_stitched = {
        'original': to_numpy(stitch_image(X_og_patches_kp[:, None], **stitch_args).squeeze()),
        'denoised': to_numpy(stitch_image(X_dn_patches_kp[:, None], **stitch_args).squeeze())
        if X_target_denoised is not None else None
    }
    X_kp_stitched_thresh = {
        'original': threshold_img(X_kp_stitched['original'], args.keypoints_threshold),
        'denoised': threshold_img(X_kp_stitched['denoised'], args.keypoints_threshold)
        if X_target_denoised is not None else None
    }

    # Find peaks in the keypoints images generated from the original and denoised images
    for lbl in ['original', 'denoised']:
        if X_kp[lbl] is None:
            continue

        # Find the peaks in the complete image
        logger.info(f'Finding peaks in heatmap generated from complete {lbl} image.')
        if args.keypoints_exclude_border > 0:
            exclude_border = round(args.keypoints_exclude_border * X_kp[lbl].shape[-1])
        else:
            exclude_border = False
        coords_lbl = peak_local_max(
            X_kp_thresh[lbl],
            min_distance=args.keypoints_min_distance,
            threshold_abs=args.keypoints_threshold,
            exclude_border=exclude_border,
        )
        coords_lbl = coords_lbl[:, ::-1].copy()  # Flip the coordinates to (y, x) to match the projector
        logger.info(f'Found {len(coords_lbl)} keypoints.')

        # Save the keypoints heatmaps
        logger.info(f'Saving keypoints heatmaps generated from {lbl} image.')
        save_img(X[lbl], lbl, save_dir)
        save_img(X_kp[lbl], f'kp_from_{lbl}', save_dir)
        save_img(X_kp_thresh[lbl], f'kp_from_{lbl}_T={args.keypoints_threshold}', save_dir)

        # Save the keypoints heatmaps overlaid onto the original images
        logger.info(f'Saving keypoints heatmaps overlaid onto the original images.')
        for lbl2 in ['original', 'denoised']:
            if X_kp[lbl2] is not None:
                save_img_with_keypoint_overlay(X[lbl], X_kp[lbl2], lbl, lbl2, save_dir)
            if X_kp_stitched[lbl2] is not None:
                save_img_with_keypoint_overlay(X[lbl], X_kp_stitched[lbl2], lbl, lbl2 + '_stitched', save_dir)

        # Save the images with keypoints marked
        logger.info(f'Saving keypoints heatmaps generated from {lbl} image with marked keypoints.')
        save_img_with_keypoint_markers(X[lbl], coords_lbl, lbl + '_complete', save_dir, 'o')
        save_img_with_keypoint_markers(X_kp[lbl], coords_lbl, f'kp_from_{lbl}', save_dir, 'x')
        save_img_with_keypoint_markers(X_kp_thresh[lbl], coords_lbl, f'kp_from_{lbl}_T={args.keypoints_threshold}',
                                       save_dir, 'x')

        # Loop over patches
        for i in range(len(X_patches['original'])):
            lbl_patch = f'p{i:02d}_{lbl}'
            lbl_kp_patch = f'p{i:02d}_kp_from_{lbl}'

            # Find the peaks
            if args.keypoints_exclude_border > 0:
                exclude_border = round(args.keypoints_exclude_border * X_patches_kp[lbl][i].shape[-1])
            else:
                exclude_border = False
            coords_patch = peak_local_max(
                X_patches_kp_thresh[lbl][i],
                min_distance=args.keypoints_min_distance,
                threshold_abs=args.keypoints_threshold,
                exclude_border=exclude_border,
            )
            coords_patch = coords_patch[:, ::-1].copy()  # Flip the coordinates to (y, x) to match the projector
            logger.info(f'Found {len(coords_patch)} keypoints in patch.')

            # Save the patches and keypoints heatmaps
            logger.info(f'Saving patches and keypoints heatmaps for {lbl} image.')
            save_img(X_patches[lbl][i], lbl_patch, save_dir)
            save_img(X_patches_kp[lbl][i], lbl_kp_patch, save_dir)

            # Save the keypoints heatmaps overlaid onto the original images
            logger.info(f'Saving keypoints heatmaps overlaid onto the original images.')
            for lbl2 in ['original', 'denoised']:
                if X_patches_kp[lbl2] is not None:
                    save_img_with_keypoint_overlay(X_patches[lbl][i], X_patches_kp[lbl2][i], lbl_patch, lbl2, save_dir)

            # Save the patches with keypoints marked
            logger.info(f'Saving patches with marked keypoints for {lbl} image.')
            save_img_with_keypoint_markers(X_patches[lbl][i], coords_patch, lbl_patch, save_dir, 'o')
            save_img_with_keypoint_markers(X_patches_kp[lbl][i], coords_patch, lbl_kp_patch, save_dir, 'x')

        # Find the peaks in the stitched image
        logger.info(f'Finding peaks in heatmap generated from complete {lbl} image.')
        if args.keypoints_exclude_border > 0:
            exclude_border = round(args.keypoints_exclude_border * X_kp_stitched[lbl].shape[-1])
        else:
            exclude_border = False
        coords_stitched = peak_local_max(
            X_kp_stitched_thresh[lbl],
            min_distance=args.keypoints_min_distance,
            threshold_abs=args.keypoints_threshold,
            exclude_border=exclude_border
        )
        coords_stitched = coords_stitched[:, ::-1].copy()  # Flip the coordinates to (y, x) to match the projector
        logger.info(f'Found {len(coords_stitched)} keypoints.')

        # Save the keypoints heatmaps
        logger.info(f'Saving keypoints heatmaps generated from {lbl} image patches.')
        save_img(X_kp_stitched[lbl], f'kp_from_{lbl}_stitched', save_dir)
        save_img(
            threshold_img(X_kp_stitched[lbl], threshold=args.keypoints_threshold),
            f'kp_from_{lbl}_stitched_T={args.keypoints_threshold}', save_dir
        )

        # Save the keypoints heatmaps overlaid onto the original images
        logger.info(f'Saving keypoints heatmaps overlaid onto the original images.')
        for lbl2 in ['original', 'denoised']:
            if X_kp_stitched[lbl2] is None:
                continue
            save_img_with_keypoint_overlay(X[lbl], X_kp_stitched[lbl2], lbl, lbl2 + '_stitched', save_dir)

        # Save the images with keypoints marked
        logger.info(f'Saving keypoints heatmaps generated from {lbl} stitched image with marked keypoints.')
        save_img_with_keypoint_markers(X[lbl], coords_stitched, lbl + '_stitched', save_dir, 'o')
        save_img_with_keypoint_markers(X_kp_stitched[lbl], coords_stitched, f'kp_from_{lbl}_stitched', save_dir, 'x')
        save_img_with_keypoint_markers(X_kp_stitched_thresh[lbl], coords_stitched,
                                       f'kp_from_{lbl}_stitched_T={args.keypoints_threshold}', save_dir, 'x')


def plot_keypoints_noise_batch(args: Optional[RuntimeArgs] = None):
    """
    Plot the keypoints for a given image with added noise.
    """
    args, manager, save_dir, X_target, X_target_denoised, Y_target, r_params_target = _init(args,
                                                                                            'keypoints_noise_batch')
    resize_args = dict(mode='bilinear', align_corners=False)
    manager.keypoint_detector_args.kd_resize_input = False  # We'll manually resize the images
    img_size = args.keypoints_max_img_size if args.keypoints_oversize_input else manager.image_shape[-1]

    X = {}
    for Xt, lbl in zip([X_target, X_target_denoised], ['original', 'denoised']):
        if Xt is None:
            X[lbl] = None
            continue

        # Progressively add more noise to the second half of the batch
        Xt = Xt.repeat(args.batch_size, 1, 1, 1)
        noise = torch.randn_like(Xt)[:args.batch_size // 2] \
                * torch.linspace(0, 0.2, args.batch_size // 2, device=manager.device)[:, None, None, None]
        Xt[args.batch_size // 2:] = Xt[args.batch_size // 2:] + noise

        # Progressively blur the first half of the batch
        for j in range(args.batch_size // 2):
            Xt[args.batch_size // 2 - j - 1] = gaussian_blur(Xt[args.batch_size // 2 - j], kernel_size=[9, 9])

        X[lbl] = Xt

    # Detect keypoints for original and denoised complete images
    logger.info('Detecting keypoints in complete images.')
    X_batch = torch.cat([X[lbl] for lbl in ['original', 'denoised'] if X[lbl] is not None])
    restore_size = False
    if args.keypoints_oversize_input and X_batch.shape[-1] > img_size \
            or not args.keypoints_oversize_input and X_batch.shape[-1] != img_size:
        restore_size = X_batch.shape[-1]
        X_batch = F.interpolate(X_batch, size=img_size, **resize_args)
    X_kp = detect_keypoints_batch(manager, X_batch, batch_size=args.batch_size)
    if restore_size:
        X_kp = F.interpolate(X_kp[:, None], size=restore_size, **resize_args)[:, 0]
    X_kp = {
        'original': to_numpy(X_kp[:args.batch_size]),
        'denoised': to_numpy(X_kp[args.batch_size:]) if X_target_denoised is not None else None
    }
    X_kp_thresh = {
        'original': threshold_img(X_kp['original'], args.keypoints_threshold),
        'denoised': threshold_img(X_kp['denoised'], args.keypoints_threshold)
        if X_target_denoised is not None else None
    }

    # Tile the image and denoised image into overlapping patches
    tile_args = dict(n_tiles=args.keypoints_n_tiles, overlap=args.keypoints_tile_overlap)
    X_og_patches, patch_positions = tile_image(X['original'], **tile_args)
    if X_target_denoised is not None:
        X_dn_patches, _ = tile_image(X['denoised'], **tile_args)
    X_patches = {
        'original': to_numpy(X_og_patches),
        'denoised': to_numpy(X_dn_patches) if X_target_denoised is not None else None
    }

    # Detect keypoints in the image patches
    logger.info('Detecting keypoints in image patches.')
    X_batch = X_patches['original']
    if X_target_denoised is not None:
        X_batch = torch.cat([X_og_patches, X_dn_patches])
    X_batch = X_batch.reshape(-1, *X_batch.shape[2:])
    restore_size = False
    if args.keypoints_oversize_input and X_batch.shape[-1] > img_size \
            or not args.keypoints_oversize_input and X_batch.shape[-1] != img_size:
        restore_size = X_batch.shape[-1]
        X_batch = F.interpolate(X_batch, size=img_size, **resize_args)
    X_patches_kp = detect_keypoints_batch(manager, X_batch, batch_size=args.keypoints_batch_size)
    if restore_size:
        X_patches_kp = F.interpolate(X_patches_kp[:, None], size=restore_size, **resize_args)[:, 0]
    X_patches_kp = X_patches_kp.reshape(args.batch_size * 2, args.keypoints_n_tiles, *X_patches_kp.shape[1:])
    X_og_patches_kp = X_patches_kp[:len(X_og_patches)]
    if X_target_denoised is not None:
        X_dn_patches_kp = X_patches_kp[len(X_og_patches):]
    X_patches_kp = {
        'original': to_numpy(X_og_patches_kp),
        'denoised': to_numpy(X_dn_patches_kp)
        if X_target_denoised is not None else None
    }
    X_patches_kp_thresh = {
        'original': threshold_img(X_patches_kp['original'], args.keypoints_threshold),
        'denoised': threshold_img(X_patches_kp['denoised'], args.keypoints_threshold)
        if X_target_denoised is not None else None
    }

    # Stitch the patches back together
    stitch_args = dict(patch_positions=patch_positions, overlap=args.keypoints_tile_overlap)
    X_kp_stitched = {
        'original': to_numpy(stitch_image(X_og_patches_kp[:, :, None], **stitch_args).squeeze()),
        'denoised': to_numpy(stitch_image(X_dn_patches_kp[:, :, None], **stitch_args).squeeze())
        if X_target_denoised is not None else None
    }
    X_kp_stitched_thresh = {
        'original': threshold_img(X_kp_stitched['original'], args.keypoints_threshold),
        'denoised': threshold_img(X_kp_stitched['denoised'], args.keypoints_threshold)
        if X_target_denoised is not None else None
    }

    # Find peaks in the keypoints images generated from the original and denoised images
    for lbl in ['original', 'denoised']:
        if X_kp[lbl] is None:
            continue

        for i in range(args.batch_size):
            # Find the peaks in the complete image
            logger.info(
                f'Finding peaks in heatmap generated from complete {lbl} image with augmentation #{i + 1}/{args.batch_size}.')
            if args.keypoints_exclude_border > 0:
                exclude_border = round(args.keypoints_exclude_border * X_kp[lbl].shape[-1])
            else:
                exclude_border = False
            coords_lbl = peak_local_max(
                X_kp_thresh[lbl][i],
                min_distance=args.keypoints_min_distance,
                threshold_abs=args.keypoints_threshold,
                exclude_border=exclude_border,
            )
            coords_lbl = coords_lbl[:, ::-1].copy()  # Flip the coordinates to (y, x) to match the projector
            logger.info(f'Found {len(coords_lbl)} keypoints.')

            # Save the keypoints heatmaps
            logger.info(f'Saving keypoints heatmaps generated from {lbl} image.')
            save_img(X[lbl][i], f'{lbl}_n{i:02d}', save_dir)
            save_img(X_kp[lbl][i], f'kp_from_{lbl}_n{i:02d}', save_dir)
            save_img(X_kp_thresh[lbl][i], f'kp_from_{lbl}_n{i:02d}_T={args.keypoints_threshold}', save_dir)

            # Save the keypoints heatmaps overlaid onto the original images
            logger.info(f'Saving keypoints heatmaps overlaid onto the original images.')
            for lbl2 in ['original', 'denoised']:
                if X_kp[lbl2] is not None:
                    save_img_with_keypoint_overlay(X[lbl][i], X_kp[lbl2][i], f'{lbl}_n{i:02d}', lbl2, save_dir)
                if X_kp_stitched[lbl2] is not None:
                    save_img_with_keypoint_overlay(X[lbl][i], X_kp_stitched[lbl2][i], f'{lbl}_n{i:02d}',
                                                   lbl2 + '_stitched', save_dir)

            # Save the images with keypoints marked
            logger.info(f'Saving keypoints heatmaps generated from {lbl} image with marked keypoints.')
            save_img_with_keypoint_markers(X[lbl][i], coords_lbl, f'{lbl}_n{i:02d}_complete', save_dir, 'o')
            save_img_with_keypoint_markers(X_kp[lbl][i], coords_lbl, f'kp_from_{lbl}_n{i:02d}', save_dir, 'x')
            save_img_with_keypoint_markers(X_kp_thresh[lbl][i], coords_lbl,
                                           f'kp_from_{lbl}_n{i:02d}_T={args.keypoints_threshold}',
                                           save_dir, 'x')

            # Loop over patches
            for j in range(X_patches['original'].shape[1]):
                lbl_patch = f'p{j:02d}_{lbl}_n{i:02d}'
                lbl_kp_patch = f'p{j:02d}_kp_from_{lbl}_n{i:02d}'

                # Find the peaks
                if args.keypoints_exclude_border > 0:
                    exclude_border = round(args.keypoints_exclude_border * X_patches_kp[lbl][i].shape[-1])
                else:
                    exclude_border = False
                coords_patch = peak_local_max(
                    X_patches_kp_thresh[lbl][i, j],
                    min_distance=args.keypoints_min_distance,
                    threshold_abs=args.keypoints_threshold,
                    exclude_border=exclude_border,
                )
                coords_patch = coords_patch[:, ::-1].copy()  # Flip the coordinates to (y, x) to match the projector
                logger.info(f'Found {len(coords_patch)} keypoints in patch.')

                # Save the patches and keypoints heatmaps
                logger.info(f'Saving patches and keypoints heatmaps for {lbl} image.')
                save_img(X_patches[lbl][i, j], lbl_patch, save_dir)
                save_img(X_patches_kp[lbl][i, j], lbl_kp_patch, save_dir)

                # Save the keypoints heatmaps overlaid onto the original images
                logger.info(f'Saving keypoints heatmaps overlaid onto the original images.')
                for lbl2 in ['original', 'denoised']:
                    if X_patches_kp[lbl2] is not None:
                        save_img_with_keypoint_overlay(X_patches[lbl][i, j], X_patches_kp[lbl2][i, j], lbl_patch, lbl2,
                                                       save_dir)

                # Save the patches with keypoints marked
                logger.info(f'Saving patches with marked keypoints for {lbl} image.')
                save_img_with_keypoint_markers(X_patches[lbl][i, j], coords_patch, lbl_patch, save_dir, 'o')
                save_img_with_keypoint_markers(X_patches_kp[lbl][i, j], coords_patch, lbl_kp_patch, save_dir, 'x')

            # Find the peaks in the stitched image
            logger.info(f'Finding peaks in heatmap generated from complete {lbl} image.')
            if args.keypoints_exclude_border > 0:
                exclude_border = round(args.keypoints_exclude_border * X_kp_stitched[lbl].shape[-1])
            else:
                exclude_border = False
            coords_stitched = peak_local_max(
                X_kp_stitched_thresh[lbl][i],
                min_distance=args.keypoints_min_distance,
                threshold_abs=args.keypoints_threshold,
                exclude_border=exclude_border
            )
            coords_stitched = coords_stitched[:, ::-1].copy()  # Flip the coordinates to (y, x) to match the projector
            logger.info(f'Found {len(coords_stitched)} keypoints.')

            # Save the keypoints heatmaps
            logger.info(f'Saving keypoints heatmaps generated from {lbl} image patches.')
            save_img(X_kp_stitched[lbl][i], f'kp_from_{lbl}_stitched_n{i:02d}', save_dir)
            save_img(
                threshold_img(X_kp_stitched[lbl][i], threshold=args.keypoints_threshold),
                f'kp_from_{lbl}_stitched_T={args.keypoints_threshold}_n{i:02d}', save_dir
            )

            # Save the images with keypoints marked
            logger.info(f'Saving keypoints heatmaps generated from {lbl} stitched image with marked keypoints.')
            save_img_with_keypoint_markers(X[lbl][i], coords_stitched, f'{lbl}_stitched_n{i:02d}', save_dir, 'o')
            save_img_with_keypoint_markers(X_kp_stitched[lbl][i], coords_stitched, f'kp_from_{lbl}_stitched_n{i:02d}',
                                           save_dir, 'x')
            save_img_with_keypoint_markers(X_kp_stitched_thresh[lbl][i], coords_stitched,
                                           f'kp_from_{lbl}_stitched_T={args.keypoints_threshold}_n{i:02d}', save_dir,
                                           'x')


def plot_keypoints_targeted(args: Optional[RuntimeArgs] = None):
    """
    Plot the keypoints using a targeted approach.
    """
    args, manager, save_dir, X_target, X_target_denoised, Y_target, r_params_target = _init(args, 'targeted')
    assert X_target_denoised is not None, 'Denoised image must be provided to use targeted keypoints.'

    res = find_keypoints(
        X_target=X_target,
        X_target_denoised=X_target_denoised,
        manager=manager,
        oversize_input=args.keypoints_oversize_input,
        max_img_size=args.keypoints_max_img_size,
        batch_size=args.keypoints_batch_size,
        min_distance=args.keypoints_min_distance,
        threshold=args.keypoints_threshold,
        exclude_border=args.keypoints_exclude_border,
        blur_kernel_relative_size=args.keypoints_blur_kernel_relative_size,
        n_patches=args.keypoints_n_patches,
        patch_size=args.keypoints_patch_size,
        patch_search_res=args.keypoints_patch_search_res,
        attenuation_sigma=args.keypoints_attenuation_sigma,
        max_attenuation_factor=args.keypoints_max_attenuation_factor,
        low_res_catchment_distance=args.keypoints_low_res_catchment_distance,
        return_everything=True,
    )

    # Check that the relative and absolute keypoints match
    Y_rel = res['Y_candidates_final_rel']
    Y_abs = to_absolute_coordinates(Y_rel, X_target.shape[-1])
    assert torch.allclose(Y_abs, res['Y_candidates_final']), 'Relative and absolute keypoints do not match.'

    kp_img_args = dict(save_dir=save_dir, marker_type='o', suffix='')

    # Save the images and heatmaps
    save_img(X_target, 'original', save_dir)
    save_img(X_target_denoised, 'denoised', save_dir)
    save_img(res['X_lr'], 'denoised_lowres', save_dir)
    save_img(res['X_lr_kp'], 'kp_low_res', save_dir)
    save_img_with_keypoint_markers2(X_target, res['Y_lr'], lbl='original_lowres_markers', **kp_img_args)
    save_img_with_keypoint_markers2(res['X_lr'], res['Y_lr'], lbl='denoised_lowres_markers', **kp_img_args)

    # Save the patches as a grid
    save_img_grid(res['X_patches'], 'patches_og', save_dir, coords=res['Y_patches'])
    save_img_grid(res['X_patches_kp'], 'patches_og_kp', save_dir)
    save_img_grid(res['X_patches_dn'], 'patches_dn', save_dir, coords=res['Y_patches_dn'])
    save_img_grid(res['X_patches_dn_kp'], 'patches_dn_kp', save_dir)

    # Plot the combined keypoints
    save_img_with_keypoint_markers2(X_target, res['Y_candidates_all'], lbl='Y_candidates_0_all', **kp_img_args)
    save_img_with_keypoint_markers2(X_target, res['Y_candidates_merged'], lbl='Y_candidates_1_merged', **kp_img_args)
    save_img_with_keypoint_markers2(X_target, res['Y_candidates_final'], lbl='Y_candidates_2_final', **kp_img_args)


if __name__ == '__main__':
    start_time = time.time()
    # plot_keypoints()
    # plot_keypoints_noise_batch()
    plot_keypoints_targeted()

    # Print how long this took
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logger.info(f'Finished in {int(minutes):02d}:{int(seconds):02d}.')
