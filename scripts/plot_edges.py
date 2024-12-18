import time
from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import center_crop, to_tensor

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.refiner.denoising import denoise_image, stitch_image, tile_image
from crystalsizer3d.refiner.edge_detection import detect_edges_batch, find_edges
from crystalsizer3d.util.image_helpers import save_img, save_img_grid, save_img_with_edge_overlay, \
    threshold_img
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
            edges_model_path: Optional[Path] = None,
            edges_oversize_input: bool = False,
            edges_max_img_size: int = 1024,
            edges_batch_size: int = 4,
            edges_threshold: float = 0.5,
            edges_exclude_border: float = 0.05,

            # Tiling method
            edges_pred_from: str = 'denoised',
            edges_n_tiles: int = 4,
            edges_tile_overlap: float = 0.1,

            # Multi-crop targeted method
            edges_n_patches: int = 16,
            edges_patch_size: int = 700,
            edges_patch_search_res: int = 256,
            edges_attenuation_sigma: float = 0.5,
            edges_max_attenuation_factor: float = 1.5,
            edges_low_res_catchment_distance: int = 100,

            **kwargs
    ):
        # Target settings
        if image_path is not None:
            assert image_path.exists(), f'Image path does not exist: {image_path}'
        self.image_path = image_path
        self.ds_idx = ds_idx
        self.batch_size = batch_size

        # Denoising settings
        if edges_pred_from == 'denoised':
            assert denoiser_model_path is not None, 'Denoiser model path must be set to predict edges from denoised images.'
        if denoiser_model_path is not None:
            assert denoiser_model_path.exists(), f'DN model path does not exist: {denoiser_model_path}'
        self.denoiser_model_path = denoiser_model_path
        self.denoiser_n_tiles = denoiser_n_tiles
        self.denoiser_tile_overlap = denoiser_tile_overlap
        self.denoiser_oversize_input = denoiser_oversize_input
        self.denoiser_max_img_size = denoiser_max_img_size
        self.denoiser_batch_size = denoiser_batch_size

        # Edge detection settings
        assert edges_model_path.exists(), f'Edges model path does not exist: {edges_model_path}'
        self.edges_model_path = edges_model_path
        self.edges_oversize_input = edges_oversize_input
        self.edges_max_img_size = edges_max_img_size
        self.edges_batch_size = edges_batch_size
        self.edges_threshold = edges_threshold
        self.edges_exclude_border = edges_exclude_border

        # Tiling method
        self.edges_pred_from = edges_pred_from
        self.edges_n_tiles = edges_n_tiles
        self.edges_tile_overlap = edges_tile_overlap

        # Multi-crop targeted method
        self.edges_n_patches = edges_n_patches
        self.edges_patch_size = edges_patch_size
        self.edges_patch_search_res = edges_patch_search_res
        self.edges_attenuation_sigma = edges_attenuation_sigma
        self.edges_max_attenuation_factor = edges_max_attenuation_factor
        self.edges_low_res_catchment_distance = edges_low_res_catchment_distance

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

        # Edge detection settings
        group.add_argument('--edges-model-path', type=Path,
                           help='Path to the edges model checkpoint.')
        group.add_argument('--edges-oversize-input', type=str2bool, default=False,
                           help='Whether to resize the input images to the maximum image size for edge detection.')
        group.add_argument('--edges-max-img-size', type=int, default=1024,
                           help='Maximum image size for edge detection.')
        group.add_argument('--edges-batch-size', type=int, default=3,
                           help='Number of tiles to detect edges at a time.')
        group.add_argument('--edges-threshold', type=float, default=0.5,
                           help='Threshold for edge detection.')
        group.add_argument('--edges-exclude-border', type=float, default=0.05,
                           help='Exclude edges within this ratio of the border.')

        # Tiling method
        group.add_argument('--edges-pred-from', type=str, default='denoised', choices=['denoised', 'original'],
                           help='Calculate the edges from either the original or denoised image.')
        group.add_argument('--edges-n-tiles', type=int, default=9,
                           help='Number of tiles to split the image into for edges detection.')
        group.add_argument('--edges-tile-overlap', type=float, default=0.05,
                           help='Ratio of overlap between tiles for edges detection.')

        # Multi-crop targeted method
        group.add_argument('--edges-n-patches', type=int, default=16,
                           help='Number of patches to crop from the image for high res edge detection.')
        group.add_argument('--edges-patch-size', type=int, default=700,
                           help='Size of the crop patches.')
        group.add_argument('--edges-patch-search-res', type=int, default=256,
                           help='Resolution of the low-res edge heatmap to use for determining where to crop the patches.')
        group.add_argument('--edges-attenuation-sigma', type=float, default=0.5,
                           help='Sigma parameter for the Gaussian blob used to iteratively attenuate the edge heatmap.')
        group.add_argument('--edges-max-attenuation-factor', type=float, default=1.5,
                           help='Maximum Gaussian peak height used for the attenuation function.')

        return group


def parse_arguments(printout: bool = True) -> RuntimeArgs:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Detect edges in crystal images.')
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
    dir_name = f'{START_TIMESTAMP}_{args.edges_model_path.stem[:4]}_{target_str}'
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
        model_path=args.edges_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': USE_CUDA,
                'batch_size': 1
            },
        },
        save_dir=save_dir
    )
    assert manager.keypoint_detector is not None, 'No edge detector found.'
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
    if args.edges_pred_from == 'denoised':
        assert args.denoiser_model_path is not None, 'Denoiser model path must be set to detect edges from denoised images.'
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


def plot_edges(args: Optional[RuntimeArgs] = None):
    """
    Plot the edges for a given image.
    """
    args, manager, save_dir, X_target, X_target_denoised, Y_target, r_params_target = _init(args, 'edges')
    resize_args = dict(mode='bilinear', align_corners=False)
    manager.keypoint_detector_args.kd_resize_input = False  # We'll manually resize the images
    img_size = args.edges_max_img_size if args.edges_oversize_input else manager.image_shape[-1]
    X = {
        'original': to_numpy(X_target),
        'denoised': to_numpy(X_target_denoised)
    }

    # Detect edges for original and denoised complete images
    logger.info('Detecting edges in complete image.')
    X_batch = X_target[None, ...]
    if X_target_denoised is not None:
        X_batch = torch.cat([X_batch, X_target_denoised[None, ...]])
    restore_size = False
    if args.edges_oversize_input and X_batch.shape[-1] > img_size \
            or not args.edges_oversize_input and X_batch.shape[-1] != img_size:
        restore_size = X_batch.shape[-1]
        X_batch = F.interpolate(X_batch, size=img_size, **resize_args)
    _, X_wf = manager.detect_keypoints(X_batch)
    X_wf = X_wf.sum(dim=1).clamp(0, 1)  # Combine the anterior and posterior heatmaps
    if restore_size:
        X_wf = F.interpolate(X_wf[:, None], size=restore_size, **resize_args)[:, 0]
    X_wf = {
        'original': to_numpy(X_wf[0]),
        'denoised': to_numpy(X_wf[1]) if X_target_denoised is not None else None
    }
    X_wf_thresh = {
        'original': threshold_img(X_wf['original'], args.edges_threshold),
        'denoised': threshold_img(X_wf['denoised'], args.edges_threshold)
        if X_target_denoised is not None else None
    }

    # Tile the image and denoised image into overlapping patches
    tile_args = dict(n_tiles=args.edges_n_tiles, overlap=args.edges_tile_overlap)
    X_og_patches, patch_positions = tile_image(X_target, **tile_args)
    if X_target_denoised is not None:
        X_dn_patches, _ = tile_image(X_target_denoised, **tile_args)
    X_patches = {
        'original': to_numpy(X_og_patches),
        'denoised': to_numpy(X_dn_patches) if X_target_denoised is not None else None
    }

    # Detect edges in the image patches
    logger.info('Detecting edges in image patches.')
    X_batch = X_patches['original']
    if X_target_denoised is not None:
        X_batch = torch.cat([X_og_patches, X_dn_patches])
    restore_size = False
    if args.edges_oversize_input and X_batch.shape[-1] > img_size \
            or not args.edges_oversize_input and X_batch.shape[-1] != img_size:
        restore_size = X_batch.shape[-1]
        X_batch = F.interpolate(X_batch, size=img_size, **resize_args)
    X_patches_wf = detect_edges_batch(manager, X_batch, batch_size=args.edges_batch_size)
    if restore_size:
        X_patches_wf = F.interpolate(X_patches_wf[:, None], size=restore_size, **resize_args)[:, 0]
    X_og_patches_wf = X_patches_wf[:len(X_og_patches)]
    if X_target_denoised is not None:
        X_dn_patches_wf = X_patches_wf[len(X_og_patches):]
    X_patches_wf = {
        'original': to_numpy(X_og_patches_wf),
        'denoised': to_numpy(X_dn_patches_wf) if X_target_denoised is not None else None
    }

    # Stitch the patches back together
    stitch_args = dict(patch_positions=patch_positions, overlap=args.edges_tile_overlap)
    X_wf_stitched = {
        'original': to_numpy(stitch_image(X_og_patches_wf[:, None], **stitch_args).squeeze()),
        'denoised': to_numpy(stitch_image(X_dn_patches_wf[:, None], **stitch_args).squeeze())
        if X_target_denoised is not None else None
    }

    # Plot the edge images generated from the original and denoised images
    for lbl in ['original', 'denoised']:
        if X_wf[lbl] is None:
            continue

        # Save the edge heatmaps
        logger.info(f'Saving edge heatmaps generated from {lbl} image.')
        save_img(X[lbl], lbl, save_dir)
        save_img(X_wf[lbl], f'wf_from_{lbl}', save_dir)
        save_img(X_wf_thresh[lbl], f'wf_from_{lbl}_T={args.edges_threshold}', save_dir)

        # Save the edge heatmaps overlaid onto the original images
        logger.info(f'Saving edge heatmaps overlaid onto the original images.')
        for lbl2 in ['original', 'denoised']:
            if X_wf[lbl2] is not None:
                save_img_with_edge_overlay(X[lbl], X_wf[lbl2], lbl, lbl2, save_dir)
            if X_wf_stitched[lbl2] is not None:
                save_img_with_edge_overlay(X[lbl], X_wf_stitched[lbl2], lbl, lbl2 + '_stitched', save_dir)

        # Loop over patches
        for i in range(len(X_patches['original'])):
            lbl_patch = f'p{i:02d}_{lbl}'
            lbl_wf_patch = f'p{i:02d}_kp_from_{lbl}'

            # Save the patches and edge heatmaps
            logger.info(f'Saving patches and edge heatmaps for {lbl} image.')
            save_img(X_patches[lbl][i], lbl_patch, save_dir)
            save_img(X_patches_wf[lbl][i], lbl_wf_patch, save_dir)

            # Save the edge heatmaps overlaid onto the original images
            logger.info(f'Saving edge heatmaps overlaid onto the original images.')
            for lbl2 in ['original', 'denoised']:
                if X_patches_wf[lbl2] is not None:
                    save_img_with_edge_overlay(X_patches[lbl][i], X_patches_wf[lbl2][i], lbl_patch, lbl2, save_dir)

        # Save the edge heatmaps
        logger.info(f'Saving edge heatmaps generated from {lbl} image patches.')
        save_img(X_wf_stitched[lbl], f'wf_from_{lbl}_stitched', save_dir)
        save_img(
            threshold_img(X_wf_stitched[lbl], threshold=args.edges_threshold),
            f'wf_from_{lbl}_stitched_T={args.edges_threshold}', save_dir
        )

        # Save the edge heatmaps overlaid onto the original images
        logger.info(f'Saving edge heatmaps overlaid onto the original images.')
        for lbl2 in ['original', 'denoised']:
            if X_wf_stitched[lbl2] is None:
                continue
            save_img_with_edge_overlay(X[lbl], X_wf_stitched[lbl2], lbl, lbl2 + '_stitched', save_dir)


def plot_edges_targeted(args: Optional[RuntimeArgs] = None):
    """
    Plot the edges using a targeted approach.
    """
    args, manager, save_dir, X_target, X_target_denoised, Y_target, r_params_target = _init(args, 'targeted')
    assert X_target_denoised is not None, 'Denoised image must be provided to use targeted edge detection.'

    res = find_edges(
        X_target=X_target.cpu(),
        X_target_denoised=X_target_denoised,
        manager=manager,
        oversize_input=args.edges_oversize_input,
        max_img_size=args.edges_max_img_size,
        batch_size=args.edges_batch_size,
        threshold=args.edges_threshold,
        exclude_border=args.edges_exclude_border,
        n_patches=args.edges_n_patches,
        patch_size=args.edges_patch_size,
        patch_search_res=args.edges_patch_search_res,
        attenuation_sigma=args.edges_attenuation_sigma,
        max_attenuation_factor=args.edges_max_attenuation_factor,
        return_everything=True,
    )

    # Save the images and heatmaps
    save_img(X_target, 'original', save_dir)
    save_img(X_target_denoised, 'denoised', save_dir)
    save_img(res['X_lr_wf'], 'wf_low_res', save_dir)

    # Save the patches as a grid
    save_img_grid(res['X_patches'], 'patches_og', save_dir)
    save_img_grid(res['X_patches_wf'], 'patches_og_wf', save_dir)
    save_img_grid(res['X_patches_dn'], 'patches_dn', save_dir)
    save_img_grid(res['X_patches_dn_wf'], 'patches_dn_wf', save_dir)

    # Plot the combined edges
    save_img(res['X_combined_wf'], 'combined_wf', save_dir)


if __name__ == '__main__':
    start_time = time.time()
    # plot_edges()
    plot_edges_targeted()

    # Print how long this took
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logger.info(f'Finished in {int(minutes):02d}:{int(seconds):02d}.')
