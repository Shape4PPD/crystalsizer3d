import time
from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mayavi import mlab
from torch import Tensor
from torch.utils.data import default_collate
from torchvision.transforms.functional import center_crop, crop, gaussian_blur, to_tensor

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.projector import Projector
from crystalsizer3d.refiner.denoising import denoise_batch, denoise_image, stitch_image, tile_image
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.util.image_helpers import save_img_grid, save_img_with_keypoint_markers2
from crystalsizer3d.util.plots import make_3d_digital_crystal_image, make_error_image, plot_distances, plot_light, \
    plot_material, plot_transformation
from crystalsizer3d.util.utils import print_args, str2bool, to_dict, to_numpy

# Off-screen rendering
mlab.options.offscreen = True


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            model_path: Path,
            dn_model_path: Optional[Path] = None,
            image_path: Optional[Path] = None,
            ds_idx: int = 0,
            noise_batch_size: int = 1,
            pred_batch_size: int = 1,
            pred_oversize_input: bool = True,
            pred_max_img_size: int = 512,
            dn_oversize_input: bool = True,
            dn_max_img_size: int = 512,

            img_size_3d: int = 400,
            wireframe_r_factor: float = 0.2,
            surface_colour_target: str = 'orange',
            wireframe_colour_target: str = 'darkorange',
            surface_colour_pred: str = 'skyblue',
            wireframe_colour_pred: str = 'cornflowerblue',
            azim: float = 0,
            elev: float = 0,
            roll: float = 0,
            distance: float = 7,

            plot_colour_target: str = 'red',
            plot_colour_pred: str = 'darkblue',

            **kwargs
    ):
        assert model_path.exists(), f'Dataset path does not exist: {model_path}'
        assert model_path.suffix == '.json', f'Model path must be a json file: {model_path}'
        self.model_path = model_path
        if dn_model_path is not None:
            assert dn_model_path.exists(), f'DN model path does not exist: {dn_model_path}'
            assert dn_model_path.suffix == '.json', f'DN model path must be a json file: {dn_model_path}'
        self.dn_model_path = dn_model_path
        if image_path is not None:
            assert image_path.exists(), f'Image path does not exist: {image_path}'
        self.image_path = image_path
        self.ds_idx = ds_idx
        self.noise_batch_size = noise_batch_size
        self.pred_batch_size = pred_batch_size
        self.pred_oversize_input = pred_oversize_input
        self.pred_max_img_size = pred_max_img_size
        self.dn_oversize_input = dn_oversize_input
        self.dn_max_img_size = dn_max_img_size

        # Digital crystal image
        self.img_size_3d = img_size_3d
        self.wireframe_r_factor = wireframe_r_factor
        self.surface_colour_target = surface_colour_target
        self.wireframe_colour_target = wireframe_colour_target
        self.surface_colour_pred = surface_colour_pred
        self.wireframe_colour_pred = wireframe_colour_pred
        self.azim = azim
        self.elev = elev
        self.roll = roll
        self.distance = distance

        # Parameter plots
        self.plot_colour_target = plot_colour_target
        self.plot_colour_pred = plot_colour_pred

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Runtime Args')
        group.add_argument('--model-path', type=Path, required=True,
                           help='Path to the model\'s json file.')
        group.add_argument('--dn-model-path', type=Path,
                           help='Path to the denoising model\'s json file.')
        group.add_argument('--image-path', type=Path,
                           help='Path to the image to process. If set, will override the dataset entry.')
        group.add_argument('--ds-idx', type=int, default=0,
                           help='Index of the dataset entry to use.')
        group.add_argument('--noise-batch-size', type=int, default=16,
                           help='Batch size for a noisy prediction batch.')
        group.add_argument('--pred-batch-size', type=int, default=8,
                           help='Batch size for generating predictions.')
        group.add_argument('--pred-oversize-input', type=str2bool, default=False,
                           help='Send images to the predictor at full (or reduced to max) resolution. '
                                'Otherwise, downsample to the resolution the predictor was trained at.')
        group.add_argument('--pred-max-img-size', type=int, default=512,
                           help='Maximum image size for the predictor.')
        group.add_argument('--dn-oversize-input', type=str2bool, default=True,
                           help='Send images to the denoiser at full (or reduced to max) resolution. '
                                'Otherwise, downsample to the resolution the denoiser was trained at.')
        group.add_argument('--dn-max-img-size', type=int, default=512,
                           help='Maximum image size for the denoiser.')

        # Digital crystal image
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
        group.add_argument('--azim', type=float, default=50,
                           help='Azimuthal angle of the camera.')
        group.add_argument('--elev', type=float, default=50,
                           help='Elevation angle of the camera.')
        group.add_argument('--roll', type=float, default=-120,
                           help='Roll angle of the camera.')
        group.add_argument('--distance', type=float, default=7,
                           help='Camera distance.')

        # Parameter plots
        group.add_argument('--plot-colour-target', type=str, default='darkorange',
                           help='Target parameters plot colour.')
        group.add_argument('--plot-colour-pred', type=str, default='cornflowerblue',
                           help='Predicted parameters plot colour.')

        return group


def parse_arguments(printout: bool = True) -> RuntimeArgs:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Predict 3D morphology using a neural network.')
    RuntimeArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    runtime_args = RuntimeArgs.from_args(args)

    return runtime_args


def _init(args: Optional[RuntimeArgs], method: str):
    if args is None:
        args = parse_arguments()

    # Create an output directory
    if args.image_path is None:
        target_str = str(args.ds_idx)
    else:
        target_str = args.image_path.stem
    dir_name = f'{START_TIMESTAMP}_{args.model_path.stem[:4]}_{target_str}'
    if method == 'prediction_noise_batch':
        dir_name += f'_bs={args.noise_batch_size}'
    save_dir = LOGS_PATH / method / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments to json file
    with open(save_dir / 'args.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    # Instantiate the manager from the checkpoint json path
    manager = Manager.load(
        model_path=args.model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': USE_CUDA,
                'batch_size': 1
            },
        },
        save_dir=save_dir
    )
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

        if manager.dataset_args.use_clean_images:
            # X_target = image_clean   # todo: fix the broken clean images!

            # The clean images are not correct, so use this to re-render the target image
            img_target2, scene_target = manager.crystal_renderer.render_from_parameters(
                r_params_target, return_scene=True
            )
            scene_target.clear_interference()
            img_target2_clean = scene_target.render()
            Image.fromarray(img_target2_clean).save(save_dir / f'X_target_v2.png')
            img = Image.open(save_dir / f'X_target_v2.png')
            X_target = to_tensor(img)
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
        X_target = center_crop(X_target, min(X_target.shape[-2:]))
        # d = min(X_target.shape[-2:])
        # X_target = crop(X_target, top=0, left=X_target.shape[-1] - d, height=d, width=d)
        # X_target = F.interpolate(
        #     X_target[None, ...],
        #     size=manager.image_shape[-1],
        #     mode='bilinear',
        #     align_corners=False
        # )[0]
    X_target = default_collate([X_target, ])
    X_target = X_target.to(manager.device)

    return args, manager, save_dir, X_target, Y_target, r_params_target


def _plot_parameters(
        manager: Manager,
        args: RuntimeArgs,
        Y_pred: Dict[str, Tensor],
        Y_target: Optional[Dict[str, Tensor]] = None,
        idx: int = 0,
) -> Figure:
    """
    Plot the image and parameter predictions.
    """
    fig = plt.figure(figsize=(5, 4))
    gs = GridSpec(
        nrows=2,
        ncols=2,
        wspace=0.2,
        hspace=0.4,
        top=0.93,
        bottom=0.08,
        left=0.07,
        right=0.98
    )

    # Plot the parameters
    shared_args = dict(
        manager=manager,
        Y_pred=Y_pred,
        Y_target=Y_target,
        idx=idx,
        colour_pred=args.plot_colour_pred,
        colour_target=args.plot_colour_target,
        colour_pred2=args.plot_colour_pred,
    )
    if manager.dataset_args.train_distances:
        plot_distances(fig.add_subplot(gs[0, 0]), **shared_args)
    if manager.dataset_args.train_transformation:
        plot_transformation(fig.add_subplot(gs[0, 1]), **shared_args)
    if manager.dataset_args.train_material and len(manager.ds.labels_material_active) > 0:
        plot_material(fig.add_subplot(gs[1, 0]), **shared_args)
    if manager.dataset_args.train_light:
        plot_light(fig.add_subplot(gs[1, 1]), **shared_args)

    return fig
    # plt.show()
    # exit()


def _make_plots(
        args: RuntimeArgs,
        save_dir: Path,
        manager: Manager,
        X_target: Tensor,
        Y_pred: Dict[str, Tensor],
        Y_target: Optional[Dict[str, Tensor]] = None,
        r_params_target: Optional[Dict[str, Tensor]] = None,
        idx: int = 0,
        resize_res: int = None
):
    """
    Make a selection of plots and renderings.
    """
    r_params_pred = manager.ds.denormalise_rendering_params(Y_pred, idx)
    X_target_i = X_target[idx]

    # Plot the digital crystals
    logger.info('Plotting digital crystals.')
    crystal_pred = manager.ds.load_crystal(r_params=r_params_pred, zero_origin=True)
    dc_shared_args = dict(
        res=args.img_size_3d,
        wireframe_radius_factor=args.wireframe_r_factor,
        azim=args.azim,
        elev=args.elev,
        roll=args.roll,
        distance=crystal_pred.scale.item() * args.distance,
    )
    try:
        dig_pred = make_3d_digital_crystal_image(
            crystal=crystal_pred,
            surface_colour=args.surface_colour_pred,
            wireframe_colour=args.wireframe_colour_pred,
            **dc_shared_args
        )
        dig_pred.save(save_dir / f'digital_predicted_{idx:02d}.png')
    except Exception as e:
        logger.warning(f'Failed to plot predicted digital crystal: {e}')
    if Y_target is not None:
        crystal_target = manager.ds.load_crystal(r_params=r_params_target, zero_origin=True)
        dig_target = make_3d_digital_crystal_image(
            crystal=crystal_target,
            surface_colour=args.surface_colour_target,
            wireframe_colour=args.wireframe_colour_target,
            **dc_shared_args
        )
        dig_target.save(save_dir / f'digital_target_{idx:02d}.png')
        try:
            dig_combined = make_3d_digital_crystal_image(
                crystal=crystal_pred,
                crystal_comp=crystal_target,
                surface_colour=args.surface_colour_pred,
                wireframe_colour=args.wireframe_colour_pred,
                surface_colour_comp=args.surface_colour_target,
                wireframe_colour_comp=args.wireframe_colour_target,
                **dc_shared_args
            )
            dig_combined.save(save_dir / f'digital_combined_{idx:02d}.png')
        except Exception as e:
            logger.warning(f'Failed to plot combined digital crystals: {e}')

    # Save the original image
    logger.info('Saving rendered images.')
    img_target = to_numpy(X_target_i * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    Image.fromarray(img_target).save(save_dir / f'target_{idx:02d}.png')
    if resize_res is not None and img_target.shape[-1] != resize_res:
        img_target = np.array(Image.fromarray(img_target).resize((resize_res, resize_res)))

    # Re-render with the rendering pipeline
    if Y_target is not None:
        img_target2, scene_target = manager.crystal_renderer.render_from_parameters(r_params_target, return_scene=True)
        Image.fromarray(img_target2).save(save_dir / f'target_rerendered_{idx:02d}.png')

    # Render the predicted crystal
    img_pred, scene_pred = manager.crystal_renderer.render_from_parameters(r_params_pred, return_scene=True)
    if img_pred.shape[:2] != img_target.shape[:2]:
        img_pred = np.array(Image.fromarray(img_pred).resize(img_target.shape[:2][::-1]))
    Image.fromarray(img_pred).save(save_dir / f'predicted_{idx:02d}.png')

    # Project the wireframes and keypoints onto the images
    projector_args = dict(
        crystal=scene_pred.crystal,
        image_size=(img_pred.shape[1], img_pred.shape[0]),
        zoom=orthographic_scale_factor(scene_pred)
    )
    projector_pred = Projector(background_image=img_pred, **projector_args)
    img_pred_overlay = to_numpy(projector_pred.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    Image.fromarray(img_pred_overlay).save(save_dir / f'predicted_overlay_pred_{idx:02d}.png')
    projector_target = Projector(background_image=img_target, **projector_args)
    img_target_overlay_pred = to_numpy(projector_target.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    Image.fromarray(img_target_overlay_pred).save(save_dir / f'target_overlay_pred_{idx:02d}.png')
    if Y_target is not None:
        projector_target.crystal = scene_target.crystal
        projector_target.project()
        img_target_overlay_target = to_numpy(projector_target.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
        Image.fromarray(img_target_overlay_target).save(save_dir / f'target_overlay_target_{idx:02d}.png')

    # Add keypoint images
    for X, projector, pred_or_target \
            in zip([img_pred, img_target], [projector_pred, projector_target], ['pred', 'target']):
        save_img_with_keypoint_markers2(
            X=X,
            coords=to_numpy(projector.keypoints),
            fill_colour='lightgreen' if pred_or_target == 'target' else 'coral',
            outline_colour='darkgreen' if pred_or_target == 'target' else 'darkred',
            lbl=f'keypoints_{pred_or_target}',
            suffix=f'_{idx:02d}',
            save_dir=save_dir,
        )

    # Create error images
    img_l2 = make_error_image(img_target, img_pred, loss_type='l2')
    img_l2.save(save_dir / f'error_l2_{idx:02d}.png')
    img_l1 = make_error_image(img_target, img_pred, loss_type='l1')
    img_l1.save(save_dir / f'error_l1_{idx:02d}.png')

    # Plot the parameter values
    fig = _plot_parameters(manager, args, Y_pred, Y_target, idx)
    fig.savefig(save_dir / f'parameters_{idx:02d}.svg', transparent=True)
    plt.close(fig)


def _make_image_grid_plots(
        save_dir: Path,
        manager: Manager,
        X_target: Tensor,
        Y_pred: Dict[str, Tensor],
):
    """
    Make a selection of plots and renderings.
    """
    images = []

    for idx, X_target_i in enumerate(X_target):
        r_params_pred = manager.ds.denormalise_rendering_params(Y_pred, idx)
        img_target = to_numpy(X_target_i * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)

        # Render the predicted crystal
        img_pred, scene_pred = manager.crystal_renderer.render_from_parameters(r_params_pred, return_scene=True)
        if img_pred.shape[:2] != img_target.shape[:2]:
            img_pred = np.array(Image.fromarray(img_pred).resize(img_target.shape[:2][::-1]))

        # Project the wireframes onto the images
        projector_args = dict(
            crystal=scene_pred.crystal,
            image_size=(img_pred.shape[1], img_pred.shape[0]),
            zoom=orthographic_scale_factor(scene_pred)
        )
        projector = Projector(background_image=img_target, **projector_args)
        img_target_overlay_pred = to_numpy(projector.image * 255).astype(np.uint8).squeeze()

        images.append(img_target_overlay_pred)

    images = np.stack(images)

    save_img_grid(
        X=images,
        lbl='target_overlay_pred',
        save_dir=save_dir,
    )


def plot_prediction(args: Optional[RuntimeArgs] = None):
    """
    Plot the predicted parameters for a given image.
    """
    args, manager, save_dir, X_target, Y_target, r_params_target = _init(args, 'prediction')

    # Predict parameters and plot
    logger.info('Predicting parameters at full resolution.')
    full_res = X_target.shape[-1]
    Y_pred = manager.predict(X_target)
    (save_dir / f'{full_res}').mkdir(parents=True, exist_ok=True)
    _make_plots(args, save_dir / f'{full_res}', manager, X_target, Y_pred, Y_target, r_params_target)

    # Resize the input to match the predictor's training resolution
    resize_args = dict(mode='bilinear', align_corners=False)
    img_size = manager.image_shape[-1]
    X_target2 = F.interpolate(X_target, size=img_size, **resize_args)
    Y_pred2 = manager.predict(X_target2)
    (save_dir / f'{img_size}').mkdir(parents=True, exist_ok=True)
    _make_plots(args, save_dir / f'{img_size}', manager, X_target2, Y_pred2, Y_target, r_params_target,
                resize_res=full_res)


def plot_prediction_noise_batch(args: Optional[RuntimeArgs] = None):
    """
    Plot a batch of predicted parameters for a given image with some noise added.
    """
    args, manager, save_dir, X_target, Y_target, r_params_target = _init(args, 'prediction_noise_batch')

    def make_noisy_batch(X, n):
        # Progressively add more noise to the second half of the batch
        X = X.repeat(n, 1, 1, 1)
        noise = torch.randn_like(X)[:n // 2] \
                * torch.linspace(0, 0.2, n // 2, device=manager.device)[:, None, None, None]
        X[n // 2:] = X[n // 2:] + noise

        # Progressively blur the first half of the batch
        for j in range(n // 2):
            X[n // 2 - j - 1] = gaussian_blur(X[n // 2 - j], kernel_size=[9, 9])

        return X

    # Make the initial batch from the original image
    X_batch = make_noisy_batch(X_target, args.noise_batch_size)

    # Load the denoising model
    if args.dn_model_path.exists():
        logger.info('Denoising the image.')
        manager.load_network(args.dn_model_path, 'denoiser')
        X_denoised = denoise_image(
            manager=manager,
            X=X_target[0],
            n_tiles=9,
            overlap=0.08,
            oversize_input=args.dn_oversize_input,
            max_img_size=args.dn_max_img_size,
            batch_size=3
        )
        X_denoised = make_noisy_batch(X_denoised, args.noise_batch_size)
        X_batch = torch.cat([X_batch, X_denoised])

    # Resize the images if necessary
    resize_args = dict(mode='bilinear', align_corners=False)
    img_size = args.pred_max_img_size if args.pred_oversize_input else manager.image_shape[-1]
    if args.pred_oversize_input and X_batch.shape[-1] > img_size \
            or not args.pred_oversize_input and X_batch.shape[-1] != img_size:
        X_batch = F.interpolate(X_batch, size=img_size, **resize_args)

    # Predict parameters
    logger.info('Predicting parameters.')
    bs = args.pred_batch_size
    n_batches = (X_batch.shape[0] + bs - 1) // bs
    Y_pred = {}
    for i in range(n_batches):
        X_ = X_batch[i * bs:(i + 1) * bs] if bs > 0 else X_batch
        Y_pred_i = manager.predict(X_)
        for k, v in Y_pred_i.items():
            if k not in Y_pred:
                Y_pred[k] = v
            else:
                Y_pred[k] = torch.concatenate([Y_pred[k], v], dim=0)

    # Plot
    _make_image_grid_plots(save_dir, manager, X_batch, Y_pred)
    # for i in range(args.noise_batch_size):
    #     _make_plots(args, save_dir, manager, X_batch, Y_pred, Y_target, r_params_target, i)


def plot_denoised_prediction(args: Optional[RuntimeArgs] = None):
    """
    Plot the predicted parameters for a denoised image (with comparisons).
    """
    args, manager, save_dir, X_target, Y_target, r_params_target = _init(args, 'denoised_prediction')

    # Load the denoising model
    assert args.dn_model_path.exists(), f'DN model path does not exist: {args.dn_model_path}.'
    manager.load_network(args.dn_model_path, 'denoiser')

    # Denoise the image
    logger.info('Denoising the image.')
    X_denoised = manager.denoise(X_target)

    # Progressively add more noise to the remainder of the batch
    X_denoised = X_denoised.repeat(args.noise_batch_size - 1, 1, 1, 1)
    noise = torch.randn_like(X_denoised) \
            * torch.linspace(0, 0.2, args.noise_batch_size - 1, device=manager.device)[:, None, None, None]
    X_denoised = X_denoised + noise

    X_target = torch.cat([X_target, X_denoised])
    X_target.clip_(0, 1)

    # Predict parameters and plot
    logger.info('Predicting parameters.')
    Y_pred = manager.predict(X_target)
    for i in range(len(X_target)):
        _make_plots(args, save_dir, manager, X_target, Y_pred, Y_target, r_params_target, idx=i)


def plot_denoised_patches(args: Optional[RuntimeArgs] = None):
    """
    Try denoising the full-size image in patches... is it better?
    """
    args, manager, save_dir, X_target, Y_target, r_params_target = _init(args, 'denoised_patches')
    resize_args = dict(mode='bilinear', align_corners=False)

    # Patch settings
    n_tiles = 9
    overlap = 0.05
    img_size = args.dn_max_img_size if args.dn_oversize_input else manager.image_shape[-1]

    # Load the denoising model
    assert args.dn_model_path.exists(), f'DN model path does not exist: {args.dn_model_path}.'
    manager.load_network(args.dn_model_path, 'denoiser')
    manager.denoiser_args.dn_resize_input = False  # We'll manually resize the images

    # Load the full-size image and make square
    X_target_og = to_tensor(Image.open(args.image_path))
    d = min(X_target_og.shape[-2:])
    X_target_og = crop(X_target_og, top=0, left=X_target_og.shape[-1] - d, height=d, width=d)

    # Tile it up into overlapping patches
    X_patches, patch_positions = tile_image(X_target_og, n_tiles=n_tiles, overlap=overlap)

    # Denoise the full size image
    logger.info('Denoising the complete image.')
    restore_size = False
    if args.dn_oversize_input and X_target_og.shape[-1] > img_size \
            or not args.dn_oversize_input and X_target_og.shape[-1] != img_size:
        restore_size = X_target_og.shape[-1]
        X_target_og = F.interpolate(X_target_og[None, ...], size=img_size, **resize_args)[0]
    X_target_denoised = manager.denoise(X_target_og[None, ...].to(manager.device))[0]
    if restore_size:
        X_target_og = F.interpolate(X_target_og[None, ...], size=restore_size, **resize_args)[0]
        X_target_denoised = F.interpolate(X_target_denoised[None, ...], size=restore_size, **resize_args)[0]

    # Denoise the patches
    logger.info('Denoising the patches.')
    restore_size = False
    if args.dn_oversize_input and X_patches.shape[-1] > img_size \
            or not args.dn_oversize_input and X_patches.shape[-1] != img_size:
        restore_size = X_patches.shape[-1]
        X_patches = F.interpolate(X_patches, size=img_size, **resize_args)
    X_patches_denoised = denoise_batch(manager, X_patches.to(manager.device), batch_size=3)
    if restore_size:
        X_patches = F.interpolate(X_patches, size=restore_size, **resize_args)
        X_patches_denoised = F.interpolate(X_patches_denoised, size=restore_size, **resize_args)

    # Plot the full denoising attempt
    logger.info('Plotting.')
    img = to_numpy(X_target_og * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    Image.fromarray(img).save(save_dir / f'target.png')
    img = to_numpy(X_target_denoised * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    Image.fromarray(img).save(save_dir / f'target_denoised.png')

    # Plot patches
    for i in range(len(X_patches_denoised)):
        patch = to_numpy(X_patches[i] * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
        Image.fromarray(patch).save(save_dir / f'patch_{i:02d}.png')
        patch_dn = to_numpy(X_patches_denoised[i] * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
        Image.fromarray(patch_dn).save(save_dir / f'patch_{i:02d}_denoised.png')

    # Plot the reconstituted image from the patches
    X_stitched = stitch_image(X_patches, patch_positions, overlap=overlap)
    img = to_numpy(X_stitched * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    Image.fromarray(img).save(save_dir / f'target_restitched.png')

    # Plot the reconstituted denoised image from the patches
    X_denoised_stitched = stitch_image(X_patches_denoised, patch_positions, overlap=overlap)
    img = to_numpy(X_denoised_stitched * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    Image.fromarray(img).save(save_dir / f'target_denoised_stitched.png')


def benchmark_prediction_time(args: Optional[RuntimeArgs] = None):
    """
    Estimate the prediction time for a given image.
    """
    args, manager, save_dir, X_target, Y_target, r_params_target = _init(args, 'prediction')
    resize_args = dict(mode='bilinear', align_corners=False)
    img_size = manager.image_shape[-1]

    start_time = time.time()

    n_tests = 100
    batch_size = 32
    for _ in range(n_tests):
        X_target2 = F.interpolate(X_target, size=img_size, **resize_args)
        X_batch = X_target2.repeat(batch_size, 1, 1, 1)
        X_batch = X_batch + torch.randn_like(X_batch) * 0.1
        manager.predict(X_batch)

    elapsed_time = time.time() - start_time
    time_per_image = elapsed_time / n_tests / batch_size

    logger.info(f'Elapsed time: {elapsed_time:.2f} s\n'
                f'Time per image: {time_per_image:.2f} s\n'
                f'Images per second: {1 / time_per_image:.2f}')


if __name__ == '__main__':
    start_time = time.time()

    # plot_prediction()
    # plot_prediction_noise_batch()
    # plot_denoised_prediction()
    # plot_denoised_patches()
    benchmark_prediction_time()

    # # Iterate over all images in the image_path directory
    # args_ = parse_arguments()
    # img_dir = args_.image_path.parent
    # for img_path in img_dir.iterdir():
    #     args_.image_path = img_path
    #     try:
    #         plot_prediction(args_)
    #     except Exception as e:
    #         logger.error(f'Failed to plot prediction for image {img_path}: {e}')
    #         continue

    # Plot the first 20 images in the dataset
    # for i in range(20):
    #     args_.ds_idx = i
    #     try:
    #         plot_prediction(args_)
    #     except Exception as e:
    #         logger.error(f'Failed to plot prediction for index {i}: {e}')
    #         continue

    # Print how long this took
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logger.info(f'Finished in {int(minutes):02d}:{int(seconds):02d}.')
