import time
from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.projector import Projector
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.util.plots import make_3d_digital_crystal_image, make_error_image, plot_distances, plot_light, \
    plot_material, plot_transformation
from crystalsizer3d.util.utils import print_args, to_dict, to_numpy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mayavi import mlab
from torch.utils.data import default_collate
from torchvision.transforms.functional import crop, gaussian_blur, to_tensor

# Off-screen rendering
mlab.options.offscreen = True


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            model_path: Path,
            dn_model_path: Optional[Path] = None,
            image_path: Optional[Path] = None,
            ds_idx: int = 0,
            batch_size: int = 1,

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
        self.batch_size = batch_size

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
        group.add_argument('--batch-size', type=int, default=16,
                           help='Batch size for a noisy prediction batch.')

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

    # Load the input image (and parameters if loading from the dataset)
    if args.image_path is None:
        metas, X_target, X_target_clean, Y_target = manager.ds.load_item(args.ds_idx)
        X_target = to_tensor(X_target)
        Y_target = {
            k: torch.from_numpy(v).to(torch.float32).to(manager.device)
            for k, v in Y_target.items()
        }
        r_params_target = metas['rendering_parameters']

    else:
        X_target = to_tensor(Image.open(args.image_path))
        if X_target.shape[0] == 4:
            assert torch.allclose(X_target[3], torch.ones_like(X_target[3])), 'Transparent images not supported.'
            X_target = X_target[:3]
        Y_target = None
        r_params_target = None

        # Crop and resize the image to the working image size
        # X_target = center_crop(X_target, min(X_target.shape[-2:]))
        d = min(X_target.shape[-2:])
        X_target = crop(X_target, top=0, left=X_target.shape[-1] - d, height=d, width=d)
        X_target = F.interpolate(
            X_target[None, ...],
            size=manager.image_shape[-1],
            mode='bilinear',
            align_corners=False
        )[0]
    X_target = default_collate([X_target, ])
    X_target = X_target.to(manager.device)

    return args, manager, save_dir, X_target, Y_target, r_params_target


def _plot_parameters(
        manager: Manager,
        args: RuntimeArgs,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Optional[Dict[str, torch.Tensor]] = None,
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
        X_target: torch.Tensor,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Optional[Dict[str, torch.Tensor]] = None,
        r_params_target: Optional[Dict[str, torch.Tensor]] = None,
        idx: int = 0,
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

    # Re-render with the rendering pipeline
    if Y_target is not None:
        img_target2, scene_target = manager.crystal_renderer.render_from_parameters(r_params_target, return_scene=True)
        # img_target2 = cv2.cvtColor(img_target2, cv2.COLOR_RGB2BGR)
        Image.fromarray(img_target2).save(save_dir / f'target_rerendered_{idx:02d}.png')

    # Render the predicted crystal
    img_pred, scene_pred = manager.crystal_renderer.render_from_parameters(r_params_pred, return_scene=True)
    # img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
    Image.fromarray(img_pred).save(save_dir / f'predicted_{idx:02d}.png')

    # Project the wireframes onto the images
    projector_args = dict(
        crystal=scene_pred.crystal,
        image_size=(img_pred.shape[1], img_pred.shape[0]),
        zoom=orthographic_scale_factor(scene_pred)
    )
    projector = Projector(background_image=img_pred, **projector_args)
    img_pred_overlay = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    Image.fromarray(img_pred_overlay).save(save_dir / f'predicted_overlay_pred_{idx:02d}.png')
    projector = Projector(background_image=img_target, **projector_args)
    img_target_overlay_pred = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    Image.fromarray(img_target_overlay_pred).save(save_dir / f'target_overlay_pred_{idx:02d}.png')
    if Y_target is not None:
        projector.crystal = scene_target.crystal
        projector.project()
        img_target_overlay_target = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
        Image.fromarray(img_target_overlay_target).save(save_dir / f'target_overlay_target_{idx:02d}.png')

    # Create error images
    img_l2 = make_error_image(img_target, img_pred, loss_type='l2')
    img_l2.save(save_dir / f'error_l2_{idx:02d}.png')
    img_l1 = make_error_image(img_target, img_pred, loss_type='l1')
    img_l1.save(save_dir / f'error_l1_{idx:02d}.png')

    # Plot the parameter values
    fig = _plot_parameters(manager, args, Y_pred, Y_target, idx)
    fig.savefig(save_dir / f'parameters_{idx:02d}.svg', transparent=True)
    plt.close(fig)


def plot_prediction(args: Optional[RuntimeArgs] = None):
    """
    Plot the predicted parameters for a given image.
    """
    args, manager, save_dir, X_target, Y_target, r_params_target = _init(args, 'prediction')

    # Predict parameters and plot
    logger.info('Predicting parameters.')
    Y_pred = manager.predict(X_target)
    _make_plots(args, save_dir, manager, X_target, Y_pred, Y_target, r_params_target)


def plot_prediction_noise_batch(args: Optional[RuntimeArgs] = None):
    """
    Plot a batch of predicted parameters for a given image with some noise added.
    """
    args, manager, save_dir, X_target, Y_target, r_params_target = _init(args, 'prediction_noise_batch')

    # Progressively add more noise to the second half of the batch
    X_target = X_target.repeat(args.batch_size, 1, 1, 1)
    noise = torch.randn_like(X_target)[:args.batch_size // 2] \
            * torch.linspace(0, 0.2, args.batch_size // 2, device=manager.device)[:, None, None, None]
    X_target[args.batch_size // 2:] = X_target[args.batch_size // 2:] + noise

    # Progressively blur the first half of the batch
    for j in range(args.batch_size // 2):
        X_target[args.batch_size // 2 - j - 1] = gaussian_blur(X_target[args.batch_size // 2 - j], kernel_size=[9, 9])

    # Predict parameters
    logger.info('Predicting parameters.')
    Y_pred = manager.predict(X_target)

    # Plot
    for i in range(args.batch_size):
        _make_plots(args, save_dir, manager, X_target, Y_pred, Y_target, r_params_target, i)


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
    X_denoised = X_denoised.repeat(args.batch_size - 1, 1, 1, 1)
    noise = torch.randn_like(X_target)[:args.batch_size // 2] \
            * torch.linspace(0, 0.2, args.batch_size - 1, device=manager.device)[:, None, None, None]
    X_denoised = X_denoised + noise

    X_target = torch.cat([X_target, X_denoised])
    X_target.clip_(0, 1)

    # Predict parameters and plot
    logger.info('Predicting parameters.')
    Y_pred = manager.predict(X_target)
    for i in range(len(X_target)):
        _make_plots(args, save_dir, manager, X_target, Y_pred, Y_target, r_params_target, idx=i)


if __name__ == '__main__':
    start_time = time.time()

    # plot_prediction()
    # plot_prediction_noise_batch()
    plot_denoised_prediction()

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
