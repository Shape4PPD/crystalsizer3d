import gc
import math
import re
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import drjit as dr
import mitsuba as mi
import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from geomloss import SamplesLoss
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from skimage.feature import peak_local_max
from omegaconf import OmegaConf
from taming.models.lfqgan import VQModel
from taming.util import get_ckpt_path
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.scheduler.scheduler import Scheduler
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, GaussianBlur
from torchvision.transforms.functional import center_crop, to_tensor

from crystalsizer3d import DATA_PATH, LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.nn.models.rcf import RCF
from crystalsizer3d.projector import ProjectedVertexKey, Projector
from crystalsizer3d.refiner.denoising import denoise_image
from crystalsizer3d.refiner.keypoint_detection import find_keypoints, generate_attention_patches, \
    to_absolute_coordinates
from crystalsizer3d.refiner.edge_matcher import EdgeMatcher
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.util.convergence_detector import ConvergenceDetector
from crystalsizer3d.util.image_helpers import save_img, save_img_grid, save_img_with_keypoint_markers
from crystalsizer3d.util.plots import _add_bars, plot_light, plot_material, plot_transformation
from crystalsizer3d.util.utils import get_seed, gumbel_sigmoid, hash_data, init_tensor, is_bad, set_seed, to_multiscale, \
    to_numpy


# torch.autograd.set_detect_anomaly(True)


class Refiner:
    manager: Manager
    projector: Projector
    optimiser: Optimizer
    lr_scheduler: Scheduler
    metric_keys: List[str]
    step: int = 0
    loss: float = 0.

    X_target: Union[Tensor, List[Tensor]]
    X_target_wis: Union[Tensor, List[Tensor]]
    X_target_denoised: Optional[Union[Tensor, List[Tensor]]]
    X_target_denoised_wis: Optional[Union[Tensor, List[Tensor]]]
    X_target_aug: Union[Tensor, List[Tensor]]
    X_target_patches: Optional[Tensor]

    X_pred: Tensor = None
    X_pred_patches: Tensor | None = None
    patch_centres: Tensor | None = None

    scene: Scene = None
    scene_params: mi.SceneParameters = None

    crystal: Crystal = None
    symmetry_idx: Tensor
    conj_pairs: List[Tuple[int, int]]
    conj_switch_probs: Tensor
    param_group_keys: List[str]

    keypoint_targets: Optional[Tensor] = None
    anchors: Dict[ProjectedVertexKey, Tensor] = {}
    distances_est: Optional[Tensor] = None
    distances_min: Optional[Tensor] = None

    rcf_feats_og: Optional[List[Tensor]]
    rcf_feats: Optional[List[Tensor]]

    convergence_detector: ConvergenceDetector
    convergence_detector_param_names: List[str]

    def __init__(
            self,
            args: RefinerArgs,
            output_dir: Path | None = None,
            output_dir_base: Path | None = None,
            destroy_denoiser: bool = True,
            destroy_keypoint_detector: bool = True,
            destroy_predictor: bool = True,
            do_init: bool = True
    ):
        self.args = args

        # Seed
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Whether to destroy the models after use
        self.destroy_denoiser = destroy_denoiser
        self.destroy_keypoint_detector = destroy_keypoint_detector
        self.destroy_predictor = destroy_predictor

        if do_init:
            # Set up the log directory
            self.init_save_dir(output_dir, output_dir_base)

            # Load the optimisation targets
            self.init_X_target()
            self.init_X_target_denoised()
            self._init_Y_target()

    def __getattr__(self, name: str):
        """
        Lazy loading of the components.
        """
        if name == 'save_dir':
            self.init_save_dir()
            return self.save_dir
        elif name == 'manager':
            self._init_manager()
            return self.manager
        elif name == 'projector':
            self._init_projector()
            return self.projector
        elif name == 'blur':
            self._init_blur()
            return self.blur
        elif name == 'rcf':
            self._init_rcf()
            return self.rcf
        elif name == 'rcf_feats_og':
            self._calculate_rcf_feats_og()
            return self.rcf_feats_og
        elif name == 'perceptual_model':
            self._init_perceptual_model()
            return self.perceptual_model
        elif name == 'latents_model':
            self._init_latents_model()
            return self.latents_model
        elif name == 'edge_matching':
            self._init_edge_matching()
            return self.edge_matching_model
        elif name in ['optimiser', 'lr_scheduler']:
            self._init_optimiser()
            return getattr(self, name)
        elif name == 'metric_keys':
            self.metric_keys = self._init_metrics()
            return self.metric_keys
        elif name == 'tb_logger':
            self.init_tb_logger()
            return self.tb_logger
        elif name == 'device':
            self.device = self.manager.device
            return self.device
        elif name == 'symmetry_idx':
            self._init_symmetry_idx()
            return self.symmetry_idx
        elif name == 'conj_pairs' or name == 'conj_switch_probs':
            self._init_conj_switch_probs()
            return getattr(self, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_parameter_vector(self, include_switch_probs: bool = True) -> Tensor:
        """
        Return a vector of the parameters.
        """
        params = torch.concatenate([
            self.crystal.distances.detach().cpu(),
            self.crystal.origin.detach().cpu(),
            self.crystal.scale[None, ...].detach().cpu(),
            self.crystal.rotation.detach().cpu(),
            self.crystal.material_roughness[None, ...].detach().cpu(),
            self.crystal.material_ior[None, ...].detach().cpu(),
            self.scene.light_radiance.detach().cpu(),
        ])
        if include_switch_probs:
            params = torch.concatenate([params, self.conj_switch_probs.detach().cpu()])

        return params

    def init_save_dir(self, output_dir: Path | None = None, output_dir_base: Path | None = None):
        """
        Set up the output directory.
        """
        logger.info('Initialising output directory.')
        assert not (output_dir is not None and output_dir_base is not None), \
            'Can\'t set both the output_dir and the output_dir_base.'
        copy_dirs = ['cache', 'initial_prediction', 'denoise_patches', 'keypoints']

        if output_dir is None:
            if self.args.image_path is None:
                target_str = str(self.args.ds_idx)
            else:
                target_str = self.args.image_path.stem
            model_str = self.args.predictor_model_path.stem[:4] + \
                        (f'_{self.args.denoiser_model_path.stem[:4]}'
                         if self.args.denoiser_model_path is not None else '') + \
                        (f'_{self.args.keypoints_model_path.stem[:4]}'
                         if self.args.keypoints_model_path is not None and self.args.use_keypoints else '')
            base_params_str = (f'spp{self.args.spp}' +
                               f'_res{self.args.rendering_size}' +
                               (f'_ms' if self.args.multiscale else '') +
                               f'_{self.args.opt_algorithm}')
            if output_dir_base is None:
                base_dir = LOGS_PATH
            else:
                base_dir = output_dir_base
            base_dir = base_dir / model_str / target_str / base_params_str
            dir_name = hash_data(self.args.to_dict())

            # Check for existing directory
            if base_dir.exists():
                pattern = re.compile(rf'{dir_name}.*')
                existing_dirs = [d for d in base_dir.iterdir() if pattern.match(d.name)]
                if len(existing_dirs) > 0:
                    logger.warning(f'Found existing directory: {existing_dirs[0]}. Overwriting.')
                    for d in copy_dirs:
                        if (existing_dirs[0] / d).exists():
                            shutil.move(existing_dirs[0] / d, base_dir / f'{d}_tmp')
                    shutil.rmtree(existing_dirs[0])

            # Make the new save directory
            self.save_dir = base_dir / (dir_name + f'_{START_TIMESTAMP}')
            self.save_dir.mkdir(parents=True, exist_ok=True)
            for d in copy_dirs:
                if (base_dir / f'{d}_tmp').exists():
                    shutil.move(base_dir / f'{d}_tmp', self.save_dir / d)
                else:
                    (self.save_dir / d).mkdir()

        else:
            self.save_dir = output_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
            for d in copy_dirs:
                (self.save_dir / d).mkdir(exist_ok=True)

        # Save arguments to yml file
        with open(self.save_dir / 'args.yml', 'w') as f:
            spec = self.args.to_dict()
            spec['created'] = START_TIMESTAMP
            yaml.dump(spec, f)

    def _init_manager(self):
        """
        Initialise the manager used for making initial predictions and denoising the input images.
        """
        manager = Manager.load(
            model_path=self.args.predictor_model_path,
            args_changes={
                'runtime_args': {
                    'use_gpu': USE_CUDA,
                    'batch_size': 1
                },
            },
            save_dir=self.save_dir
        )
        self.manager = manager

    def _init_projector(self):
        """
        Initialise the projector.
        """
        self.projector = Projector(
            crystal=self.crystal,
            image_size=(400, 400),
            zoom=orthographic_scale_factor(self.scene),
            multi_line=True,
            rtol=1e-2
        )

    def _init_blur(self):
        """
        Initialise the Gaussian blur kernel
        """
        blur = GaussianBlur(kernel_size=5, sigma=1.0)
        blur = torch.jit.script(blur)
        blur.to(self.device)
        self.blur = blur

    def _init_perceptual_model(self):
        """
        Initialise the perceptual loss network.
        """
        if self.args.use_perceptual_model:
            assert self.args.perceptual_model is not None, 'Perceptual model not set.'
        else:
            self.perceptual_model = None
            return

        percept = timm.create_model(
            model_name=self.args.perceptual_model,
            pretrained=True,
            num_classes=0,
            features_only=True
        )
        percept.eval()

        n_params = sum([p.data.nelement() for p in percept.parameters()])
        logger.info(f'Instantiated perception network with {n_params / 1e6:.4f}M parameters '
                    f'from {self.args.perceptual_model}.')
        percept.to(self.device)

        data_config = timm.data.resolve_model_data_config(percept)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        transforms = Compose([
            transforms.transforms[0],
            transforms.transforms[1],
            transforms.transforms[3],
        ])

        def model(x):
            x = transforms(x)
            return percept(x)

        # model = torch.jit.script(model)

        self.perceptual_model = model

    def _init_latents_model(self):
        """
        Initialise the latent encoder network.
        """
        if self.args.use_latents_model:
            assert self.args.latents_model == 'MAGVIT2', 'Only MAGVIT2 encoder is supported.'
        else:
            self.latents_model = None
            return

        # Monkey-patch the LPIPS class so that it loads from a sensible place
        from taming.modules.losses.lpips import LPIPS

        def load_pips(self, name='vgg_lpips'):
            ckpt = get_ckpt_path(name, DATA_PATH / 'vgg_lpips')
            self.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'), weights_only=True), strict=False)
            logger.info(f'Loaded pretrained LPIPS loss from {ckpt}.')

        LPIPS.load_from_pretrained = load_pips

        # Load the model checkpoint
        config = OmegaConf.load(self.args.mv2_config_path)
        self.dn_config = config
        model = VQModel(**config.model.init_args)
        sd = torch.load(self.args.mv2_checkpoint_path, map_location='cpu', weights_only=True)['state_dict']
        model.load_state_dict(sd, strict=False)
        model.eval()

        # Instantiate the network
        n_params = sum([p.data.nelement() for p in model.parameters()])
        logger.info(f'Instantiated latent encoder network with {n_params / 1e6:.4f}M parameters.')
        model.to(self.device)

        self.latents_model = model

    def _init_rcf(self):
        """
        Initialise the Richer Convolutional Features model for edge detection.
        """
        rcf_path = self.args.rcf_model_path
        if self.args.use_rcf_model:
            assert rcf_path.exists(), 'RCF model path does not exist!'
        else:
            self.rcf = None
            return

        rcf = RCF()
        checkpoint = torch.load(rcf_path, weights_only=True)
        rcf.load_state_dict(checkpoint, strict=False)
        rcf.eval()

        n_params = sum([p.data.nelement() for p in rcf.parameters()])
        logger.info(f'Instantiated RCF network with {n_params / 1e6:.4f}M parameters from {rcf_path}.')
        rcf = torch.jit.script(rcf)
        rcf.to(self.device)

        self.rcf = rcf

    def _calculate_rcf_feats_og(self):
        """
        Calculate the RCF features on the original target image.
        """
        if self.rcf is None:
            self.rcf_feats_og = None
        else:
            model_input = self.X_target_wis[None, ...].permute(0, 3, 1, 2)
            self.rcf_feats_og = self.rcf(model_input, apply_sigmoid=False)

    def _init_edge_matching(self):
        edge_matching = EdgeMatcher()
        edge_matching.to(self.device)
        self.edge_matching_model = edge_matching

    def _init_convergence_detector(self):
        """
        Initialise the convergence detector.
        """
        self.convergence_detector_param_names = [
            # 'loss',
            *[f'd_{i:02d}' for i in range(self.crystal.distances.shape[0])],
            'roughness',
            'ior',
            *[f'light_{"rgb"[i]}' for i in range(3)],
        ]
        self.convergence_detector = ConvergenceDetector(
            shape=(len(self.convergence_detector_param_names),),
            tau_fast=self.args.convergence_tau_fast,
            tau_slow=self.args.convergence_tau_slow,
            threshold=self.args.convergence_threshold,
            patience=self.args.convergence_patience,
            min_absolute_threshold=1e-4
        )

    def _resize(self, X: Tensor, size: int | None = None) -> Tensor:
        """
        Resize the image to the working image size.
        """
        if size is None:
            size = self.args.rendering_size
        trim_channel_dim = False
        trim_batch_dim = False
        if X.ndim == 2:
            X = X[None, ...]
            trim_channel_dim = True
        if X.ndim == 3:
            X = X[None, ...]
            trim_batch_dim = True
        if X.ndim != 4:
            raise ValueError(f'Invalid input shape: {X.shape}')
        out_hwc = False
        if X.shape[-1] == 3:  # HWC
            X = X.permute(0, 3, 1, 2)
            out_hwc = True
        if X.shape[-1] != size:
            wis = (size, size)
            X = F.interpolate(
                X,
                size=wis,
                mode='bilinear',
                align_corners=False
            )
        if out_hwc:
            X = X.permute(0, 2, 3, 1)
        if trim_batch_dim:
            X = X[0]
        if trim_channel_dim:
            X = X[0]
        return X

    def init_X_target(self):
        """
        Load the input image.
        """
        cache_path = self.save_dir / 'cache' / 'X_target.pt'
        if cache_path.exists():
            try:
                X_target = torch.load(cache_path, weights_only=True)
                self.X_target = X_target.to(self.device)
                self.X_target_wis = self._resize(self.X_target)
                logger.info('Loaded target image.')
                return
            except Exception as e:
                logger.warning(f'Failed to load cached target image: {e}')
                cache_path.unlink()

        if self.args.image_path is None:
            metas, X_target, Y_target = self.manager.ds.load_item(self.args.ds_idx)
            X_target = to_tensor(X_target)

        else:
            X_target = to_tensor(Image.open(self.args.image_path))
            if X_target.shape[0] == 4:
                assert torch.allclose(X_target[3], torch.ones_like(X_target[3])), 'Transparent images not supported.'
                X_target = X_target[:3]
            X_target = center_crop(X_target, min(X_target.shape[-2:]))

        # Resize target image to the working image size for inverse rendering
        X_target = X_target.permute(1, 2, 0)  # HWC
        self.X_target = X_target.to(self.device)
        self.X_target_wis = self._resize(self.X_target)

        # Multiscale
        if self.args.multiscale:
            raise NotImplementedError('Multiscale not implemented.')
            X_target = to_multiscale(X_target, self.blur)
            resolution_pyramid = [t.shape[0] for t in X_target[::-1]]
            logger.info(f'Resolution pyramid has {len(X_target)} levels: '
                        f'{", ".join([str(res) for res in resolution_pyramid])}')

        # Save the target image
        torch.save(self.X_target, cache_path)

    def init_X_target_denoised(self):
        """
        Generate or load the denoised target.
        """
        cache_path = self.save_dir / 'cache' / 'X_target_denoised.pt'
        if cache_path.exists():
            try:
                X_target_dn = torch.load(cache_path, weights_only=True)
                self.X_target_denoised = X_target_dn.to(self.device)
                self.X_target_denoised_wis = self._resize(self.X_target_denoised)
                logger.info('Loaded denoised target image.')
                return
            except Exception as e:
                logger.warning(f'Failed to load cached denoised target image: {e}')
                cache_path.unlink()
        save_dir = self.save_dir / 'denoise_patches'
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load the denoiser model if set
        if self.args.denoiser_model_path is not None:
            assert self.args.denoiser_model_path.exists(), f'Denoiser model path does not exist: {self.args.denoiser_model_path}.'
            self.manager.load_network(self.args.denoiser_model_path, 'denoiser')

        # If no denoiser is set, return here
        if self.manager.denoiser is None:
            return None

        # Denoise the input image if a denoiser is available
        logger.info('Denoising input image.')
        with torch.no_grad():
            X_denoised, X_patches, X_patches_denoised, patch_positions = denoise_image(
                manager=self.manager,
                X=self.X_target.permute(2, 0, 1),
                n_tiles=self.args.denoiser_n_tiles,
                overlap=self.args.denoiser_tile_overlap,
                oversize_input=self.args.denoiser_oversize_input,
                max_img_size=self.args.denoiser_max_img_size,
                batch_size=self.args.denoiser_batch_size,
                return_patches=True
            )

            # Resize target image to the working image size for inverse rendering
            X_denoised = X_denoised.permute(1, 2, 0)
            X_denoised_wis = self._resize(X_denoised)

        # Save the denoiser output (and input)
        save_img(self.X_target, 'X_original', save_dir)
        save_img(X_denoised, 'X_denoised', save_dir)
        save_img_grid(X_patches, 'X_patches_original', save_dir)
        save_img_grid(X_patches_denoised, 'X_patches_denoised', save_dir)

        # Destroy the denoiser to free up space
        if self.destroy_denoiser:
            logger.info('Destroying denoiser to free up space.')
            self.manager.denoiser = None
            torch.cuda.empty_cache()
            gc.collect()

        # Multiscale
        if self.args.multiscale:
            X_denoised = to_multiscale(X_denoised, self.blur)

        # Save the denoised target
        torch.save(X_denoised, cache_path)
        self.X_target_denoised = X_denoised
        self.X_target_denoised_wis = X_denoised_wis

    def _init_Y_target(self):
        """
        Load the parameters if loading from the dataset.
        """
        if self.args.image_path is None:
            metas, X_target, Y_target = self.manager.ds.load_item(self.args.ds_idx)
            Y_target = {
                k: torch.from_numpy(v).to(torch.float32).to(self.device)
                for k, v in Y_target.items()
            }
            r_params_target = metas['rendering_parameters']

        else:
            Y_target = None
            r_params_target = None

        self.Y_target = Y_target
        self.r_params_target = r_params_target

    def _init_symmetry_idx(self):
        """
        Load the symmetry index from the manager's crystal.
        """
        sym_crystal = Crystal(
            lattice_unit_cell=self.crystal.lattice_unit_cell,
            lattice_angles=self.crystal.lattice_angles,
            miller_indices=self.manager.ds.dataset_args.miller_indices,
            point_group_symbol=self.crystal.point_group_symbol,
        )
        self.symmetry_idx = sym_crystal.symmetry_idx

    def _init_conj_switch_probs(self):
        """
        Initialise the conjugate switching probabilities.
        """
        miller_idxs = self.crystal.all_miller_indices
        pairs = []
        for i, mi1 in enumerate(miller_idxs):
            for j, mi2 in enumerate(miller_idxs):
                if i >= j or mi1[-1] == 0 or mi2[-1] == 0 or torch.any(mi1[:2] != mi2[:2]):
                    continue
                pairs.append((i, j))
        self.conj_pairs = pairs
        self.conj_switch_probs = nn.Parameter(
            self.args.conj_switch_prob_init * torch.ones(len(pairs)),
            requires_grad=True
        )

    def init_keypoint_targets(self):
        """
        Initialise the keypoint targets.
        """
        if not self.args.use_keypoints:
            self.keypoint_targets = None
            return

        # Try to load the keypoints data from cache
        res = None
        cache_path = self.save_dir / 'cache' / 'keypoints.pt'
        if cache_path.exists():
            try:
                res = torch.load(cache_path, weights_only=True)
                logger.info(f'Loaded target keypoints data from {cache_path}')
            except Exception as e:
                logger.warning(f'Failed to load cached keypoints data: {e}')
                cache_path.unlink()

        if res is None:
            logger.info('No cached keypoints data found. Finding keypoints.')

            # Load the keypoint detector
            self.manager.load_network(self.args.keypoints_model_path, 'keypointdetector')
            assert self.manager.keypoint_detector is not None, 'No keypoints model loaded, so can\'t predict keypoints.'

            # Predict the keypoints from the input image
            res = find_keypoints(
                X_target=self.X_target,
                X_target_denoised=self.X_target_denoised,
                manager=self.manager,
                oversize_input=self.args.keypoints_oversize_input,
                max_img_size=self.args.keypoints_max_img_size,
                batch_size=self.args.keypoints_batch_size,
                min_distance=self.args.keypoints_min_distance,
                threshold=self.args.keypoints_threshold,
                exclude_border=self.args.keypoints_exclude_border,
                blur_kernel_relative_size=self.args.keypoints_blur_kernel_relative_size,
                n_patches=self.args.keypoints_n_patches,
                patch_size=self.args.keypoints_patch_size,
                patch_search_res=self.args.keypoints_patch_search_res,
                attenuation_sigma=self.args.keypoints_attenuation_sigma,
                max_attenuation_factor=self.args.keypoints_max_attenuation_factor,
                low_res_catchment_distance=self.args.keypoints_low_res_catchment_distance,
                return_everything=True,
            )

            # Save the keypoints data to cache
            torch.save(res, cache_path)

            # Destroy the keypoint detector to free up space
            if self.destroy_keypoint_detector:
                logger.info('Destroying keypoint detector to free up space.')
                self.manager.keypoint_detector = None
                torch.cuda.empty_cache()
                gc.collect()

        # Save the images and heatmaps
        save_dir = self.save_dir / 'keypoints'
        kp_img_args = dict(save_dir=save_dir, marker_type='o', suffix='')
        save_img(res['X_lr_kp'], 'kp_low_res', save_dir)
        save_img_with_keypoint_markers(self.X_target, res['Y_lr'], 'original_lowres_markers', **kp_img_args)
        save_img_with_keypoint_markers(res['X_lr'], res['Y_lr'], 'denoised_lowres_markers', **kp_img_args)

        # Save the patches as a grid
        save_img_grid(res['X_patches'], 'patches_og', save_dir, coords=res['Y_patches'])
        save_img_grid(res['X_patches_kp'], 'patches_og_kp', save_dir)
        save_img_grid(res['X_patches_dn'], 'patches_dn', save_dir, coords=res['Y_patches_dn'])
        save_img_grid(res['X_patches_dn_kp'], 'patches_dn_kp', save_dir)

        # Plot the combined keypoints
        save_img_with_keypoint_markers(self.X_target, res['Y_candidates_all'], 'Y_candidates_0_all', **kp_img_args)
        save_img_with_keypoint_markers(self.X_target, res['Y_candidates_merged'], 'Y_candidates_1_merged',
                                       **kp_img_args)
        save_img_with_keypoint_markers(self.X_target, res['Y_candidates_final'], 'Y_candidates_2_final', **kp_img_args)

        # Save the keypoint targets
        self.keypoint_targets = res['Y_candidates_final_rel']
        logger.info(f'Found {len(self.keypoint_targets)} keypoints in image.')

    def _init_optimiser(self):
        """
        Set up the optimiser and learning rate scheduler.
        """
        logger.info('Initialising optimiser.')

        param_groups = []
        self.param_group_keys = []
        for k in ['distances', 'origin', 'rotation', 'material', 'light', 'switches']:
            lr = getattr(self.args, f'lr_{k}')
            if lr == 0:
                continue
            if k in ['distances', 'origin', 'rotation']:
                params = [getattr(self.crystal, k)]
            elif k == 'material':
                params = [self.crystal.material_roughness, self.crystal.material_ior]
            elif k == 'light':
                params = [self.scene.light_radiance]
            elif k == 'switches':
                if not self.args.use_conj_switching:
                    continue
                params = [self.conj_switch_probs]
            param_groups.append({'params': params, 'lr': lr})
            self.param_group_keys.append(k)

        optimiser = create_optimizer_v2(
            opt=self.args.opt_algorithm,
            weight_decay=0,
            model_or_params=param_groups,
        )

        # For cycle based schedulers (cosine, tanh, poly) adjust total steps for cycles and cooldown
        if self.args.lr_scheduler in ['cosine', 'tanh', 'poly']:
            cycles = max(1, self.args.lr_cycle_limit)
            if self.args.lr_cycle_mul == 1.0:
                n_steps_with_cycles = self.args.max_steps * cycles
            else:
                n_steps_with_cycles = int(math.floor(-self.args.max_steps * (self.args.lr_cycle_mul**cycles - 1)
                                                     / (1 - self.args.lr_cycle_mul)))
            n_steps_adj = math.ceil(
                self.args.max_steps * (self.args.max_steps - self.args.lr_cooldown_steps) / n_steps_with_cycles)
        else:
            n_steps_adj = self.args.max_steps

        # Create the learning rate scheduler
        lr_scheduler, _ = create_scheduler_v2(
            optimizer=optimiser,
            sched=self.args.lr_scheduler,
            num_epochs=n_steps_adj,
            decay_epochs=self.args.lr_decay_steps,
            decay_milestones=self.args.lr_decay_milestones,
            cooldown_epochs=self.args.lr_cooldown_steps,
            patience_epochs=self.args.lr_patience_steps,
            decay_rate=self.args.lr_decay_rate,
            min_lr=self.args.lr_min,
            warmup_lr=self.args.lr_warmup,
            warmup_epochs=self.args.lr_warmup_steps,
            cycle_mul=self.args.lr_cycle_mul,
            cycle_decay=self.args.lr_cycle_decay,
            cycle_limit=self.args.lr_cycle_limit,
            k_decay=self.args.lr_k_decay,
            plateau_mode='min'
        )

        self.optimiser = optimiser
        self.lr_scheduler = lr_scheduler

    def _init_metrics(self) -> List[str]:
        """
        Set up the metrics to track.
        """
        metric_keys = ['losses/anchors', ]
        if self.args.use_inverse_rendering:
            metric_keys += ['losses/l1', 'losses/l2']
            if self.args.use_perceptual_model:
                metric_keys.append('losses/perceptual')
            if self.args.use_latents_model:
                metric_keys.append('losses/latent')
            if self.args.use_rcf_model:
                metric_keys.append('losses/rcf')
        if self.args.use_keypoints:
            metric_keys.append('losses/keypoints')

        return metric_keys

    def init_tb_logger(self):
        """Initialise the tensorboard writer."""
        self.tb_logger = SummaryWriter(self.save_dir, flush_secs=5)

    def set_anchors(self, anchors: Dict[ProjectedVertexKey, Tensor]):
        """
        Set the manually-defined anchor points.
        """
        logger.info('Setting anchor points.')
        self.anchors = {k: v.to(self.device) for k, v in anchors.items()}

    @torch.no_grad()
    def set_initial_scene(self, scene: Scene):
        """
        Set the initial scene data directly.
        """
        logger.info('Setting initial scene parameters.')

        with torch.no_grad():
            if self.crystal is None:
                self.crystal = scene.crystal
            else:
                self.crystal.copy_parameters_from(scene.crystal)
                scene.crystal = self.crystal
            self.crystal.to('cpu')
            scene.light_radiance = nn.Parameter(init_tensor(scene.light_radiance, device=self.device),
                                                requires_grad=True)
            scene.build_mi_scene()
            self.scene = scene
            self.scene_params = mi.traverse(scene.mi_scene)

            # Render the scene to get the initial X_pred
            X_pred = scene.render(seed=get_seed())
            X_pred = X_pred.astype(np.float32) / 255.
            X_pred = torch.from_numpy(X_pred).permute(2, 0, 1).to(self.device)
            X_pred = F.interpolate(
                X_pred[None, ...],
                size=self.args.rendering_size,
                mode='bilinear',
                align_corners=False
            )[0].permute(1, 2, 0)
            if self.args.multiscale:
                X_pred = to_multiscale(X_pred, self.blur)
            self.X_pred = X_pred

        # Reinitialise the optimiser to include the new parameters
        self._init_optimiser()

    @torch.no_grad()
    def make_initial_prediction(self):
        """
        Make initial prediction
        """
        save_dir = self.save_dir / 'initial_prediction'
        save_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = self.save_dir / 'cache'
        scene_path = cache_dir / 'scene.yml'
        X_pred_path = cache_dir / 'X_pred.pt'

        def update_scene_parameters(scene_: Scene):
            # Update the scene rendering parameters for optimisation
            scene_.res = self.args.rendering_size
            scene_.spp = self.args.spp
            scene_.camera_type = 'perspective'  # thinlens doesn't work for inverse rendering
            scene_.integrator_max_depth = self.args.integrator_max_depth
            scene_.integrator_rr_depth = self.args.integrator_rr_depth
            scene_.light_radiance = nn.Parameter(init_tensor(scene.light_radiance, device=scene.device),
                                                 requires_grad=True)
            scene_.build_mi_scene()
            scene_.crystal.to('cpu')
            return scene_

        if scene_path.exists():
            try:
                self.X_pred = torch.load(X_pred_path, weights_only=True)
                scene = Scene.from_yml(scene_path)
                self.scene = update_scene_parameters(scene)
                self.scene_params = mi.traverse(self.scene.mi_scene)
                self.crystal = self.scene.crystal
                logger.info('Loaded initial prediction from cache.')
                return
            except Exception as e:
                logger.warning(f'Failed to load cached initial prediction: {e}')
                scene_path.unlink()
                X_pred_path.unlink()
        logger.info('Predicting parameters.')

        # Use both the original and denoised image as inputs
        X_target = torch.stack([self.X_target, self.X_target_denoised]).permute(0, 3, 1, 2)

        # Set up the resizing
        resize_args = dict(mode='bilinear', align_corners=False)
        resize_input_old = self.manager.predictor.resize_input
        self.manager.predictor.resize_input = False
        oversize_input = self.args.initial_pred_oversize_input
        img_size = self.args.initial_pred_max_img_size if oversize_input else self.manager.image_shape[-1]

        # Resize the input if needed
        if oversize_input and X_target.shape[-1] > img_size \
                or not oversize_input and X_target.shape[-1] != img_size:
            X_target = F.interpolate(X_target, size=img_size, **resize_args)

        # Create a batch
        bs = self.args.initial_pred_batch_size // 2
        X_target_batch = X_target[None, ...].repeat(bs, 1, 1, 1, 1)

        # Add some noise to the batch
        noise_scale = torch.linspace(
            self.args.initial_pred_noise_min,
            self.args.initial_pred_noise_max,
            bs,
            device=self.device
        )
        X_target_batch += torch.randn_like(X_target_batch) * noise_scale[:, None, None, None, None]

        # Reshape so the first half of the batch is the original images and the second half is the denoised images
        X_target_batch = torch.cat([X_target_batch[:, 0], X_target_batch[:, 1]])

        # Predict the parameters
        Y_pred_batch = self.manager.predict(X_target_batch)

        # Restore the resize input setting
        self.manager.predictor.resize_input = resize_input_old

        # Destroy the predictor to free up space
        if self.destroy_predictor:
            logger.info('Destroying predictor to free up space.')
            self.manager.predictor = None
            torch.cuda.empty_cache()
            gc.collect()

        # Set the target image for loss calculations
        render_size = self.manager.crystal_renderer.dataset_args.image_size  # Use the rendering size the predictor was trained on
        self.X_target_aug = self._resize(self.X_target_denoised, render_size)

        # Ensure the inverse rendering losses are turned on
        use_inverse_rendering = self.args.use_inverse_rendering
        self.args.use_inverse_rendering = True
        use_latents_model = self.args.use_latents_model
        self.args.use_latents_model = self.args.w_latent > 0
        use_perceptual_model = self.args.use_perceptual_model
        self.args.use_perceptual_model = self.args.w_perceptual > 0

        # Generate some images from the parameters
        X_pred_batch = []
        scene_batch = []
        losses = []
        for i in range(bs):
            # Render the image
            r_params = self.manager.ds.denormalise_rendering_params(Y_pred_batch, idx=i)
            X_pred_i, scene_i = self.manager.crystal_renderer.render_from_parameters(r_params, return_scene=True)
            scene_batch.append(scene_i)

            # Add the image to the batch
            X_pred_i = X_pred_i.astype(np.float32) / 255.
            X_pred_i = torch.from_numpy(X_pred_i).to(self.device)
            X_pred_batch.append(X_pred_i)

            # Multiscale
            if self.args.multiscale:
                X_pred_i = to_multiscale(X_pred_i, self.blur)

            # Set the variables temporarily
            self.X_pred = X_pred_i
            self.scene = scene_i
            self.crystal = scene_i.crystal
            self.crystal.to('cpu')

            # Project the crystal mesh - reinitialise the projector for the new scene
            if self.args.use_keypoints and len(self.keypoint_targets) > 0:
                self._init_projector()
                self.projector.project(generate_image=False)

            # Calculate losses
            loss_i, _ = self._calculate_losses()
            losses.append(loss_i.item())

            # Save image
            X = X_pred_i[0] if isinstance(X_pred_i, list) else X_pred_i
            img = Image.fromarray(to_numpy(X * 255).astype(np.uint8))
            img.save(save_dir / f'{i:02d}_noise={noise_scale[i]:.3f}_loss={loss_i:.4E}.png')

        # Restore the inverse rendering losses setting
        del self.projector
        self.args.use_inverse_rendering = use_inverse_rendering
        self.args.use_latents_model = use_latents_model
        self.args.use_perceptual_model = use_perceptual_model

        # Pick the best image
        best_idx = np.argmin(losses)
        X_pred = X_pred_batch[best_idx]
        scene = scene_batch[best_idx]

        # Save the best initial prediction scene image
        img = Image.fromarray(to_numpy(X_pred * 255).astype(np.uint8))
        img.save(save_dir / f'best_idx={best_idx}.png')

        # Update the scene rendering parameters for optimisation
        scene = update_scene_parameters(scene)
        scene_params = mi.traverse(scene.mi_scene)

        # Render the new scene
        img = scene.render()
        Image.fromarray(img).save(save_dir / f'best_idx={best_idx}_new_scene.png')

        # Move the crystal on to the CPU - faster mesh building
        scene.crystal.to('cpu')

        # Save variables
        scene.to_yml(scene_path, overwrite=True)
        torch.save(X_pred, X_pred_path)

        # Set variables
        self.X_pred = X_pred
        self.scene = scene
        self.scene_params = scene_params
        self.crystal = scene.crystal
        self._init_projector()

    def train(
            self,
            callback: Callable | None = None,
            distances_est: Tensor | None = None,
            distances_min: Tensor | None = None,
    ):
        """
        Train the parameters for a number of steps.
        """
        if self.args.use_keypoints and self.keypoint_targets is None:
            self.init_keypoint_targets()
        if self.scene is None:
            self.make_initial_prediction()
        n_steps = self.args.max_steps
        start_step = self.step
        end_step = start_step + n_steps
        logger.info(f'Training for max {n_steps} steps. Starting at step {start_step}.')
        logger.info(f'Logs path: {self.save_dir}.')
        log_freq = self.args.log_every_n_steps
        running_loss = 0.
        running_metrics = {k: 0. for k in self.metric_keys}
        running_tps = 0
        use_inverse_rendering = self.args.use_inverse_rendering  # Save the inverse rendering setting
        self.distances_est = distances_est
        self.distances_min = distances_min

        # (Re-)initialise the optimiser, learning rate scheduler and convergence detector
        self._init_optimiser()
        self._init_convergence_detector()

        # Plot initial prediction
        self.step = start_step - 1
        self._make_plots(force=True)

        for step in range(start_step, end_step):
            start_time = time.time()
            self.step = step

            # Conditionally enable the inverse rendering
            self.args.use_inverse_rendering = use_inverse_rendering and step >= self.args.ir_wait_n_steps

            # Train for a single step
            loss, stats = self._train_step()

            # Adjust tracking loss to include the IR loss placeholder
            loss_track = loss.detach().cpu()
            if use_inverse_rendering and not self.args.use_inverse_rendering and self.args.ir_loss_placeholder > 0:
                loss_track += self.args.ir_loss_placeholder
            self.tb_logger.add_scalar('losses/total_tracked', loss_track.item(), step)

            # Log the parameter values
            self.tb_logger.add_scalar('params/roughness', self.crystal.material_roughness.item(), step)
            self.tb_logger.add_scalar('params/ior', self.crystal.material_ior.item(), step)
            for i, val in enumerate(self.scene.light_radiance):
                self.tb_logger.add_scalar(f'params/light_{"rgb"[i]}', val.item(), step)
            if self.args.use_conj_switching:
                for i, (pair, prob) in enumerate(zip(self.conj_pairs, self.conj_switch_probs)):
                    ab = ','.join([f'{k}' for k in self.crystal.all_miller_indices[pair[0]][:2].tolist()])
                    self.tb_logger.add_scalar(f'params/conj_switch_probs/{ab}', prob.item(), step)

            # Log learning rates and update them
            for i, param_group in enumerate(self.param_group_keys):
                self.tb_logger.add_scalar(f'lr/{param_group}', self.optimiser.param_groups[i]['lr'], step)
            if self.args.lr_scheduler != 'none':
                self.lr_scheduler.step(step, loss_track)

            # Log convergence statistics
            for i, val in enumerate(self.convergence_detector.convergence_count):
                param_name = self.convergence_detector_param_names[i]
                self.tb_logger.add_scalar(f'convergence/{i:02d}_{param_name}', val, step)
            self.tb_logger.add_scalar(f'convergence/bad_epochs', self.lr_scheduler.lr_scheduler.num_bad_epochs, step)

            # Track running loss and metrics
            running_loss += loss
            for k in self.metric_keys:
                if k in stats:
                    running_metrics[k] += stats[k]
            time_per_step = time.time() - start_time
            running_tps += time_per_step

            # Log statistics every X steps
            if (step + 1) % log_freq == 0:
                log_msg = f'[{step + 1}/{n_steps}]\tLoss: {running_loss / log_freq:.4E}'
                for k, v in running_metrics.items():
                    k = k.replace('losses/', '')
                    log_msg += f'\t{k}: {v / log_freq:.4E}'
                logger.info(log_msg)
                running_loss = 0.
                running_metrics = {k: 0. for k in self.metric_keys}
                average_tps = running_tps / log_freq
                running_tps = 0
                seconds_left = float((self.args.max_steps - step) * average_tps)
                logger.info('Time per step: {}, Est. complete in: {}'.format(
                    str(timedelta(seconds=average_tps)),
                    str(timedelta(seconds=seconds_left))))

            # Plots
            self._make_plots()

            # Callback
            if callback is not None:
                continue_signal = callback(step, loss, stats)
                if continue_signal is False:
                    logger.info('Received stop signal. Stopping training.')
                    break

            # Check for convergence
            check_vals = torch.concatenate([
                # loss_track[None, ...],
                self.crystal.distances.detach().cpu(),
                self.crystal.material_roughness[None, ...].detach().cpu(),
                self.crystal.material_ior[None, ...].detach().cpu(),
                self.scene.light_radiance.detach().cpu(),
            ])
            self.convergence_detector.forward(check_vals, first_val=step == 0)
            if self.convergence_detector.converged.all():
                logger.info(f'Converged after {step + 1} iterations.')
                break

        # Final plots
        self._make_plots(force=True)

        # Restore the inverse rendering losses setting
        self.args.use_inverse_rendering = use_inverse_rendering

        logger.info('Training complete.')

    def _train_step(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Train for a single step.
        """
        loss, stats = self._process_step(add_noise=True)

        # Backpropagate errors
        (loss / self.args.acc_grad_steps).backward()

        # Take optimisation step
        if (self.step + 1) % self.args.acc_grad_steps == 0:
            # Clip gradients
            if self.args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_([self.crystal.distances], max_norm=self.args.clip_grad_norm)
                nn.utils.clip_grad_norm_([self.crystal.rotation], max_norm=self.args.clip_grad_norm)

            # Check for bad gradients
            for group in self.optimiser.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        if is_bad(param.grad):
                            logger.warning('Bad gradients detected!')
                            param.grad.zero_()

            # Optimisation step
            self.optimiser.step()
            self.optimiser.zero_grad()

            # Clamp parameters
            self.crystal.clamp_parameters(rescale=False)
            self.conj_switch_probs.data = torch.clamp(
                self.conj_switch_probs,
                min=self.args.conj_switch_prob_min,
                max=self.args.conj_switch_prob_max
            )
            if self.distances_min is not None:
                self.crystal.distances.data = self.crystal.distances.clamp(min=self.distances_min)

            # Actually switch the distances where the switch prob is high enough
            if self.args.use_conj_switching:
                with torch.no_grad():
                    for i, p in enumerate(self.conj_pairs):
                        if self.conj_switch_probs[i] > 0.7:
                            self.crystal.distances.data[p[0]], self.crystal.distances.data[p[1]] = \
                                self.crystal.distances[p[1]], self.crystal.distances[p[0]]
                            self.conj_switch_probs.data[i] = 1 - self.conj_switch_probs[i]

        # Log losses
        for key, val in stats.items():
            self.tb_logger.add_scalar(key, float(val), self.step)

        return loss, stats

    def _process_step(self, add_noise: bool = True) -> Tuple[Tensor, Dict[str, float]]:
        """
        Process a single step.
        """
        rotation = self.crystal.rotation
        distances = self.crystal.distances
        roughness = self.crystal.material_roughness
        ior = self.crystal.material_ior
        radiance = self.scene.light_radiance

        # Add parameter noise
        if add_noise:
            rotation = rotation + torch.randn_like(rotation) * self.args.rotation_noise
            distances = distances + torch.randn_like(distances) * self.args.distances_noise
            roughness = roughness + torch.randn_like(roughness) * self.args.material_roughness_noise
            ior = ior + torch.randn_like(ior) * self.args.material_ior_noise
            radiance = radiance + torch.randn_like(radiance) * self.args.radiance_noise

        # Randomly switch distances with their "conjugates" - the faces flipped in the c-axis
        if self.args.use_conj_switching:
            d = distances.clone()

            # Sample from the Bernoulli distribution
            if add_noise:
                for i, p in enumerate(self.conj_pairs):
                    prob = self.conj_switch_probs[i]
                    # Use Gumbel-Sigmoid for a smooth, differentiable approximation
                    switch_prob = gumbel_sigmoid(torch.log(prob / (1 - prob)))

                    # Straight-through estimator
                    switch_hard = (switch_prob > 0.5).float()
                    switch = switch_hard.detach() + switch_prob - switch_prob.detach()

                    # If the distances are switched then detach the values
                    d0 = distances[p[0]]
                    d1 = distances[p[1]]
                    if switch_hard == 1:
                        d0 = d0.detach()
                        d1 = d1.detach()
                    d[p[0]] = switch * d1 + (1 - switch) * d0
                    d[p[1]] = switch * d0 + (1 - switch) * d1

            # Switch distances according to the probabilities
            else:
                for i, p in enumerate(self.conj_pairs):
                    switch_prob = self.conj_switch_probs[i]
                    if switch_prob > 0.5:
                        d[p[0]] = distances[p[1]]
                        d[p[1]] = distances[p[0]]

            distances = d

        # Rebuild the mesh
        v, f = self.crystal.build_mesh(distances=distances, rotation=rotation, update_uv_map=False)

        # Render new image
        if self.args.use_inverse_rendering:
            device = self.scene.device
            v, f = v.to(device), f.to(device)
            eta = ior.to(device) / self.scene.crystal_material_bsdf['ext_ior']
            roughness = roughness.to(device).clone()
            radiance = radiance.to(device).clone()
            X_pred = self._render_image(v, f, eta, roughness, radiance, seed=self.step)
            X_pred = torch.clip(X_pred, 0, 1)
            if self.args.multiscale:
                X_pred = to_multiscale(X_pred, self.blur)
        else:
            X_pred = torch.zeros_like(self.X_target_wis)
        self.X_pred = X_pred

        # Use denoised target if available
        X_target = self.X_target_denoised_wis if self.X_target_denoised is not None else self.X_target_wis

        # Add some noise to the target image
        if isinstance(X_target, list):
            X_target_aug = [X + torch.randn_like(X) * self.args.image_noise_std for X in X_target]
            X_target_aug = [X.clip(0, 1) for X in X_target_aug]
        else:
            X_target_aug = X_target + torch.randn_like(X_target) * self.args.image_noise_std
            X_target_aug.clip_(0, 1)
        self.X_target_aug = X_target_aug

        # Project the crystal mesh
        if (self.args.use_keypoints and len(self.keypoint_targets) > 0) or len(self.anchors) > 0:
            self.projector.project(generate_image=False)

        # Calculate losses
        loss, stats = self._calculate_losses(distances=distances)
        self.loss = loss.item()

        return loss, stats

    @dr.wrap_ad(source='torch', target='drjit')
    def _render_image(
            self,
            vertices: mi.TensorXf,
            faces: mi.TensorXi64,
            eta: mi.TensorXf,
            roughness: mi.TensorXf,
            radiance: mi.TensorXf,
            seed: int = 1
    ):
        if seed < 0:
            seed = 0
        self.scene_params[Scene.VERTEX_KEY] = dr.ravel(vertices)
        self.scene_params[Scene.FACES_KEY] = dr.ravel(faces)
        self.scene_params[Scene.ETA_KEY] = dr.ravel(eta)
        self.scene_params[Scene.ROUGHNESS_KEY] = dr.ravel(roughness)
        self.scene_params[Scene.RADIANCE_KEY] = dr.unravel(mi.Color3f, radiance)
        self.scene_params.update()
        return mi.render(self.scene.mi_scene, self.scene_params, seed=max(0, seed))

    def _calculate_losses(
            self,
            is_patch: bool = False,
            distances: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate losses.
        """

        # Inverse rendering losses
        if self.args.use_inverse_rendering:
            l1_loss, l1_stats = self._img_loss(X_target=self.X_target_aug, X_pred=self.X_pred, loss_type='l1',
                                               decay_factor=self.args.l_decay_l1)
            l2_loss, l2_stats = self._img_loss(X_target=self.X_target_aug, X_pred=self.X_pred, loss_type='l2',
                                               decay_factor=self.args.l_decay_l2)
            percept_loss, percept_stats = self._perceptual_loss()
            latent_loss, latent_stats = self._latents_loss()
            rcf_loss, rcf_stats = self._rcf_loss(is_patch=is_patch)
        else:
            l1_loss, l1_stats = torch.tensor(0.), {}
            l2_loss, l2_stats = torch.tensor(0.), {}
            percept_loss, percept_stats = torch.tensor(0.), {}
            latent_loss, latent_stats = torch.tensor(0.), {}
            rcf_loss, rcf_stats = torch.tensor(0.), {}

        # Combine image losses
        image_loss = l1_loss.cpu() * self.args.w_img_l1 \
                     + l2_loss.cpu() * self.args.w_img_l2 \
                     + percept_loss.cpu() * self.args.w_perceptual \
                     + latent_loss.cpu() * self.args.w_latent \
                     + rcf_loss.cpu() * self.args.w_rcf

        if not is_patch:
            # Regularisations
            overshoot_loss, overshoot_stats = self._overshoot_loss(distances=distances)
            symmetry_loss, symmetry_stats = self._symmetry_loss()
            z_pos_loss, z_pos_stats = self._z_pos_loss()
            rxy_loss, rxy_stats = self._rotation_xy_loss()
            switch_loss, switch_stats = self._switch_loss()
            temporal_loss, temporal_stats = self._temporal_loss()

            # Keypoints and anchors
            keypoints_loss, keypoints_stats = self._keypoints_loss()
            anchors_loss, anchors_stats = self._anchors_loss()

            # Edge matching
            edge_matching_loss, edge_matching_stats = self._edge_matching_loss()

            # Combine losses
            loss = image_loss \
                   + overshoot_loss * self.args.w_overshoot \
                   + symmetry_loss * self.args.w_symmetry \
                   + z_pos_loss * self.args.w_z_pos \
                   + rxy_loss * self.args.w_rotation_xy \
                   + switch_loss * self.args.w_switch_probs \
                   + temporal_loss * self.args.w_temporal \
                   + keypoints_loss * self.args.w_keypoints \
                   + anchors_loss * self.args.w_anchors \
                   + edge_matching_loss * self.args.w_edge_matching

            # Patches - recalculate the image losses for each patch
            patch_loss, patch_stats = self._patches_loss()
            loss = loss * self.args.w_fullsize + patch_loss * self.args.w_patches

            # Combine stats
            stats = {
                'losses/total': loss.item(),
                **l1_stats, **l2_stats, **percept_stats, **latent_stats, **rcf_stats, **overshoot_stats,
                **symmetry_stats,
                **z_pos_stats, **rxy_stats, **switch_stats, **temporal_stats, **keypoints_stats, **anchors_stats,
                **patch_stats
            }

        else:
            loss = image_loss
            stats = {**l1_stats, **l2_stats, **percept_stats, **latent_stats, **rcf_stats}

        assert not is_bad(loss), 'Bad loss!'

        return loss, stats

    def _img_loss(
            self,
            X_target: Union[Tensor, List[Tensor]],
            X_pred: Union[Tensor, List[Tensor], List[List[Tensor]]],
            loss_type: str,
            decay_factor: float = 1.
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the image loss.
        """
        loss = torch.tensor(0., device=self.device)
        stats = {}

        # If the input is a list of lists then it is a multiscale probabilistic samples
        if isinstance(X_pred, list) and isinstance(X_pred[0], list):
            X_pred = [torch.stack([b[i] for b in X_pred]) for i in range(len(X_pred[0]))]
            X_target = [a[None, ...].expand_as(b) for a, b in zip(X_target, X_pred)]
        elif isinstance(X_target, Tensor) and isinstance(X_pred, Tensor) and X_target.shape != X_pred.shape:
            if X_target.ndim == 3 and X_pred.ndim == 3:
                raise RuntimeError(f'Image shapes do not match: {X_target.shape} != {X_pred.shape}')
            if X_target.ndim == 3:
                assert X_pred.ndim == 4, f'Image shapes do not match: {X_target.shape} != {X_pred.shape}'
                X_target = X_target[None, ...].expand_as(X_pred)
            else:
                assert X_target.ndim == 4, f'Image shapes do not match: {X_target.shape} != {X_pred.shape}'
                assert len(X_target) == 1, \
                    f'Target image must be a single image, not a batch. {len(X_target)} received.'
                X_target = X_target.expand_as(X_pred[0])

        # Multiscale losses
        if isinstance(X_target, list):
            assert isinstance(X_pred, list) and len(X_target) == len(X_pred)
            for i in range(len(X_target)):
                l, _ = self._img_loss(X_target[i], X_pred[i], loss_type=loss_type)
                ld = l * decay_factor**i
                stats[f'ms/{loss_type}/raw/{i}'] = l.item()
                stats[f'ms/{loss_type}/dec/{i}'] = ld.item()
                loss = loss + ld
            stats[f'losses/{loss_type}'] = loss.item()
            return loss, stats

        # Single scale loss
        if loss_type == 'l.5':
            loss = torch.mean(((X_target - X_pred).abs() + 1e-6).sqrt())
        elif loss_type == 'l1':
            loss = torch.mean((X_target - X_pred).abs())
        elif loss_type == 'l2':
            loss = torch.mean((X_target - X_pred)**2)
        elif loss_type == 'l4':
            loss = torch.mean((X_target - X_pred)**4)
        else:
            raise ValueError(f'Unknown loss type: {loss_type}.')
        stats[f'losses/{loss_type}'] = loss.item()

        return loss, stats

    def _perceptual_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the perceptual loss.
        """
        loss = torch.tensor(0., device=self.device)
        stats = {}
        if not self.args.use_perceptual_model or self.perceptual_model is None:
            return loss, stats

        # If multiscale, just use the first (largest) image
        X_target = self.X_target_aug
        X_pred = self.X_pred
        if self.args.multiscale:
            assert isinstance(X_target, list) and isinstance(X_pred, list)
            X_target = X_target[0]
            X_pred = X_pred[0]

        # Calculate the perceptual loss
        if X_target.ndim == 3:
            X_target = X_target[None, ...]
        if X_pred.ndim == 3:
            X_pred = X_pred[None, ...]
        assert len(X_target) == len(X_pred)
        model_input = torch.cat([X_target, X_pred]).permute(0, 3, 1, 2)
        img_feats = self.perceptual_model(model_input)
        for i, f in enumerate(img_feats):
            f_target, f_pred = f.chunk(2)
            l, _ = self._img_loss(f_target, f_pred, loss_type='l2')
            ld = l * self.args.l_decay_perceptual**i
            stats[f'perceptual/raw/{i}'] = l.item()
            stats[f'perceptual/dec/{i}'] = ld.item()
            loss = loss + ld
        stats['losses/perceptual'] = loss.item()

        return loss, stats

    def _latents_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the latent-encoding loss.
        """
        loss = torch.tensor(0., device=self.device)
        stats = {}
        if not self.args.use_latents_model or self.latents_model is None:
            return loss, stats

        # If multiscale, just use the first (largest) image
        X_target = self.X_target_aug
        X_pred = self.X_pred
        if self.args.multiscale:
            assert isinstance(X_target, list) and isinstance(X_pred, list)
            X_target = X_target[0]
            X_pred = X_pred[0]

        # Encode the images
        if X_target.ndim == 3:
            X_target = X_target[None, ...]
        if X_pred.ndim == 3:
            X_pred = X_pred[None, ...]
        assert len(X_target) == len(X_pred)
        model_input = torch.cat([X_target, X_pred]).permute(0, 3, 1, 2)

        # Resize images for input
        if self.args.latents_input_size == 0:
            input_size = self.dn_config.data.init_args.train.params.config.size
        elif self.args.latents_input_size == -1:
            input_size = model_input.shape[-1]
        else:
            input_size = self.args.latents_input_size
        if model_input.shape[-1] != input_size:
            model_input = F.interpolate(model_input, size=(input_size, input_size), mode='bilinear',
                                        align_corners=False)

        # Calculate encodings
        quant, diff, _, loss_break = self.latents_model.encode(model_input)
        img_feats = quant.permute(1, 0, 2, 3)

        # Calculate the loss
        for i, f in enumerate(img_feats):
            f_target, f_pred = f.chunk(2)
            l, _ = self._img_loss(f_target, f_pred, loss_type='l1')
            ld = l * self.args.l_decay_latent**i
            stats[f'latent/raw/{i}'] = l.item()
            stats[f'latent/dec/{i}'] = ld.item()
            loss = loss + ld
        stats['losses/latent'] = loss.item()

        return loss, stats

    def _rcf_loss(self, is_patch: bool = False) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the RCF edge detection loss.
        """
        loss = torch.tensor(0., device=self.device)
        stats = {}
        if not self.args.use_rcf_model or self.rcf is None:
            return loss, stats

        # If multiscale, just use the first (largest) image
        X_target = self.X_target_aug
        X_pred = self.X_pred
        if self.args.multiscale:
            assert isinstance(X_target, list) and isinstance(X_pred, list)
            X_target = X_target[0]
            X_pred = X_pred[0]

        # Calculate the RCF loss
        if X_target.ndim == 3:
            X_target = X_target[None, ...]
        if X_pred.ndim == 3:
            X_pred = X_pred[None, ...]
        assert len(X_target) == len(X_pred)
        model_input = torch.cat([X_target, X_pred]).permute(0, 3, 1, 2)
        rcf_feats = self.rcf(model_input, apply_sigmoid=False)
        for i, f in enumerate(rcf_feats):
            f_target, f_pred = f.chunk(2)
            l, _ = self._img_loss(f_target, f_pred, loss_type='l2')
            ld = l * self.args.l_decay_perceptual**i
            stats[f'rcf/raw/{i}'] = l.item()
            stats[f'rcf/dec/{i}'] = ld.item()
            loss = loss + ld
        stats['losses/rcf'] = loss.item()
        if not is_patch:
            self.rcf_feats = rcf_feats

        return loss, stats

    def _overshoot_loss(self, distances: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the distances overshoot loss - to regularise the distances to touch the polyhedron.
        """
        if distances is None:
            distances = self.crystal.distances
        d_min = (self.crystal.N @ self.crystal.vertices_og.T).amax(dim=1)[:len(self.crystal.miller_indices)]
        overshoot = distances - d_min
        loss = torch.where(
            overshoot > 0,
            overshoot**2,
            torch.zeros_like(overshoot)
        ).mean()
        stats = {'losses/overshoot': loss.item()}
        return loss, stats

    def _symmetry_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the symmetry loss - how close are the face distances within each group.
        """
        loss = torch.tensor(0., device=self.crystal.origin.device)
        if self.symmetry_idx is None:
            return loss, {}
        for i, hkl in enumerate(self.manager.ds.dataset_args.miller_indices):
            group_idxs = (self.symmetry_idx == i).nonzero().squeeze().tolist()
            if len(group_idxs) == 1:
                continue
            d_group = self.crystal.all_distances[group_idxs]
            l = ((d_group - d_group.mean())**2).mean()
            loss = loss + l
        stats = {'losses/symmetry': loss.item()}
        return loss, stats

    def _z_pos_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the z position loss - how close is the bottom vertex to the z=0 plane.
        """
        loss = self.crystal.vertices.amin(dim=0)[2]**2
        stats = {'losses/z_pos': loss.item()}
        return loss, stats

    def _rotation_xy_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the rotation loss - assuming the crystal should by lying flat on the xy plane.
        """
        loss = self.crystal.rotation[:2].abs().mean()
        stats = {'losses/rotation_xy': loss.item()}
        return loss, stats

    def _patches_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the super-resolution patches loss.
        """
        loss = torch.tensor(0., device=self.device)
        stats = {}
        if not self.args.use_inverse_rendering or self.args.n_patches is None or self.args.n_patches <= 0 or self.scene_params is None:
            return loss, stats

        ps = self.args.patch_size
        ps2 = ps // 2
        wis = self.args.rendering_size
        device = self.scene.device

        # If multiscale, just use the first (largest) image
        X_target_og = self.X_target_aug
        X_pred_og = self.X_pred
        if self.args.multiscale:
            assert isinstance(X_target_og, list) and isinstance(X_pred_og, list)
            X_target = X_target_og[0].clone()
            X_pred = X_pred_og[0].clone()
        else:
            X_target = X_target_og.clone()
            X_pred = X_pred_og.clone()

        # Use the L2 error map to select the best patches
        with torch.no_grad():
            l2 = ((X_pred - X_target)**2).mean(dim=-1).detach()
            error_map = F.interpolate(
                l2[None, None, ...], size=self.X_target.shape[0],
                mode='bilinear', align_corners=False
            )[0, 0]
            X_target_patches, patch_centres = generate_attention_patches(
                X=self.X_target_denoised.permute(2, 0, 1),
                X_kp=error_map,
                patch_search_res=256,
                n_patches=self.args.n_patches,
                patch_size=ps,
                attenuation_sigma=0.5,
                max_attenuation_factor=1.5,
            )
            X_target_patches = F.interpolate(
                X_target_patches[0],
                size=(wis, wis),
                mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)

        # Update the film size parameters
        sf = wis / ps
        self.scene_params[Scene.FILM_SIZE_KEY] = round(self.X_target.shape[0] * sf)
        self.scene_params[Scene.FILM_CROP_SIZE_KEY] = self.args.rendering_size

        # Render the patches
        X_pred_patches = []
        for i, patch_centre in enumerate(patch_centres):
            vertices = self.crystal.mesh_vertices.to(device)
            faces = self.crystal.mesh_faces.to(device)
            eta = self.crystal.material_ior.to(device) / self.scene.crystal_material_bsdf['ext_ior']
            roughness = self.crystal.material_roughness.to(device).clone()
            radiance = self.scene.light_radiance.to(device).clone()
            self.scene_params[Scene.FILM_CROP_OFFSET_KEY] = (patch_centre * sf - wis / 2).round().int().tolist()
            X_pred_patch = self._render_image(
                vertices, faces, eta, roughness, radiance,
                seed=self.step + i
            )
            X_pred_patches.append(X_pred_patch)
        X_pred_patches = torch.stack(X_pred_patches)

        # Restore full-size rendering parameters
        self.scene_params[Scene.FILM_SIZE_KEY] = self.args.rendering_size
        self.scene_params[Scene.FILM_CROP_OFFSET_KEY] = [0, 0]

        # Calculate losses for each pair of patches
        self.X_pred = X_pred_patches
        self.X_target_aug = X_target_patches
        patch_loss, patch_stats = self._calculate_losses(is_patch=True)
        patch_stats = {f'patches/{k.replace("losses/", "")}/{i}': v for k, v in patch_stats.items() if 'losses/' in k}
        stats.update(patch_stats)
        loss = loss + patch_loss

        stats['losses/patches'] = loss.item()

        # Restore the original image
        self.X_pred = X_pred
        self.X_target_aug = X_target

        # Save the patches
        self.patch_centres = patch_centres
        self.X_pred_patches = X_pred_patches
        self.X_target_patches = X_target_patches

        return loss, stats

    def _switch_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the conjugate switching loss.
        """
        loss = torch.tensor(0., device=self.conj_switch_probs.device)
        stats = {}
        if not self.args.use_conj_switching or self.conj_switch_probs is None:
            return loss, stats

        # Add a term to encourage selecting 0 or 1
        loss = torch.where(
            self.conj_switch_probs < 0.5,
            self.conj_switch_probs**2,
            (1 - self.conj_switch_probs)**2
        ).mean()
        stats['losses/switch'] = loss.item()

        return loss, stats

    def _temporal_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the temporal regularisation loss.
        """
        loss = torch.tensor(0., device=self.crystal.distances.device)
        stats = {}
        if self.distances_est is None:
            return loss, stats

        # L2 loss between the current distances and estimated from the sequence history
        loss = torch.mean((self.crystal.distances - self.distances_est)**2)
        stats['losses/temporal_dists'] = loss.item()

        return loss, stats

    def _keypoints_loss(self):
        """
        Calculate the keypoints loss.
        """
        loss = torch.tensor(0., device=self.keypoint_targets.device)
        stats = {}
        if not self.args.use_keypoints or self.keypoint_targets is None or len(self.keypoint_targets) == 0:
            return loss, stats

        # Calculate the closest distances between the projected keypoints and the detected targets
        k_pred = self.projector.keypoints_rel
        k_target = self.keypoint_targets
        if self.args.keypoints_loss_type == 'mindists':
            distances = torch.cdist(k_pred, k_target)
            d_target_to_pred = distances.amin(dim=0)
            d_pred_to_target = distances.amin(dim=1)
            loss = d_target_to_pred.mean() + d_pred_to_target.mean()

        # Calculate the sinkhorn loss
        elif self.args.keypoints_loss_type == 'sinkhorn':
            loss_mod = SamplesLoss('sinkhorn', p=2, blur=0.0001, reach=0.1)
            loss = loss_mod(k_pred, k_target)

        # Calculate the hausdorff loss
        elif self.args.keypoints_loss_type == 'hausdorff':
            loss_mod = SamplesLoss('hausdorff', p=2, blur=0.01)
            loss = loss_mod(k_pred, k_target)

        else:
            raise RuntimeError(f'Unknown keypoints loss type: {self.args.keypoints_loss_type}.')

        stats[f'losses/keypoints'] = loss.item()

        return loss, stats

    def _anchors_loss(self):
        """
        Calculate the loss for manual constraints.
        """
        loss = torch.tensor(0.)
        stats = {}
        if len(self.anchors) == 0:
            return loss, stats

        # Calculate the loss for each anchor
        losses = []
        for vertex_key, anchor_coords in self.anchors.items():
            vertex_id, face_idx = vertex_key
            cluster_idx = self.projector.cluster_idxs[self.projector.vertex_ids == vertex_id]
            is_visible = False
            if len(cluster_idx) > 0:
                cluster_key = (cluster_idx.item(), face_idx)
                is_visible = cluster_key in self.projector.projected_vertex_keys

            # Vertex is visible, so calculate the distance between the vertex and the target anchor location
            if is_visible:
                idx = self.projector.projected_vertex_keys.index(cluster_key)
                v_coords = self.projector.projected_vertices_rel[idx]
                l = (v_coords - anchor_coords).norm()
                losses.append(l)

            # Vertex is not visible, so set the distance to 0
            else:
                l = torch.tensor(0., device=self.device)

            # Log the loss per-anchor
            v_id, face_idx = vertex_key
            stats[f'anchors/{v_id}_{face_idx}'] = l.item()

        # Take the mean of the visible anchors
        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        stats[f'losses/anchors'] = loss.item()

        return loss, stats
    
    def _edge_matching_loss(self):
        """
        Calculate the edge matching loss.
        """
        loss = torch.tensor(0., device=self.crystal.distances.device)
        stats = {}
        if not self.args.edge_matching:
            return loss, stats
        
        # Calculate the distance between a point on the edge
        # of the crystal and a found edge
        edge_points = self.projector.edge_points
        edge_normals = self.projector.edge_normals   
        
        # These need to be calulated first     
        
        
        loss, distances = self.edge_matching_model(edge_points,edge_normals, self.rcf_feats[5])
        
        stats[f'losses/edgematching'] = loss.item()
        
        return loss, stats

    @torch.no_grad()
    def _make_plots(
            self,
            force: bool = False
    ):
        """
        Generate some example plots.
        """
        if self.args.plot_every_n_steps > -1 and (force or (self.step + 1) % self.args.plot_every_n_steps == 0):
            logger.info('Plotting.')
            # Re-process the step with no noise for plotting
            self._process_step(add_noise=False)
            with torch.no_grad():
                fig = self._plot_comparison()
                self._save_plot(fig, 'optimisation')
            if self.patch_centres is not None:
                fig = self._plot_patches()
                self._save_plot(fig, 'patches')

    @torch.no_grad()
    def _plot_comparison(self) -> Figure:
        """
        Plot the target and optimised images side by side.
        """
        X_target_og = self.X_target if isinstance(self.X_target, Tensor) else self.X_target[0]
        X_target_dn = self.X_target_denoised if isinstance(self.X_target_denoised, Tensor) \
            else self.X_target_denoised[0]
        Xs = [X_target_og, X_target_dn]
        if self.args.use_inverse_rendering:
            X_pred = [self.X_pred] if isinstance(self.X_pred, Tensor) else self.X_pred
            Xs.extend(X_pred)
        if self.args.use_keypoints:
            Xs.append(X_target_dn)
        n_cols = len(Xs)

        n_rows = 3
        rcf_feats = None
        if self.rcf_feats_og is not None:
            assert self.rcf_feats is not None, 'RCF features not available.'
            rcf_feats = [torch.cat([f0[:, 0], f1[:, 0]]) for f0, f1 in zip(self.rcf_feats_og, self.rcf_feats)]
            assert len(self.args.plot_rcf_feats) <= len(rcf_feats), \
                'Number of RCF features to plot must be less than or equal to the number of features available.'
            for f in rcf_feats:
                assert len(f) >= 3, 'Number of RCF features must match the number of images.'
            n_rows += len(self.args.plot_rcf_feats)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.4), squeeze=False)
        for i in range(n_cols):
            img = Xs[i]
            if isinstance(img, list):
                img = img[0]
            img = np.clip(to_numpy(img), 0, 1)
            ax = axes[0, i]
            ax.imshow(img)
            ax.axis('off')

            # Show the patches
            if i == 2 and self.args.use_inverse_rendering and self.patch_centres is not None:
                sf = X_target_og.shape[0] / img.shape[0]
                patch_centres = to_numpy(self.patch_centres) / sf
                ps = self.args.patch_size / X_target_og.shape[0] * img.shape[0]
                for x, y in patch_centres:
                    ax.scatter(x, y, c='darkblue', s=20, marker='x', alpha=0.7)
                    rect = Rectangle((x - ps // 2, y - ps // 2), ps, ps, linewidth=0.7, edgecolor='darkblue',
                                     facecolor='none', linestyle='--', alpha=0.7)
                    ax.add_patch(rect)

            # Show the target and predicted keypoints
            if self.args.use_keypoints and i == len(Xs) - 1 and self.keypoint_targets is not None:
                kp_target_abs = to_numpy(to_absolute_coordinates(self.keypoint_targets, img.shape[0]))
                kp_pred_abs = to_numpy(to_absolute_coordinates(self.projector.keypoints_rel, img.shape[0]))
                ax.scatter(*kp_target_abs.T,
                           facecolors=(0, 0.7, 0, 0.2),
                           edgecolors=(0, 0.7, 0, 0.5),
                           linewidths=0.5, s=25)
                ax.scatter(*kp_target_abs.T, marker='x', c='g', s=0.2, alpha=0.5)
                ax.scatter(*kp_pred_abs.T,
                           facecolors=(1, 0, 0, 0.2),
                           edgecolors=(1, 0, 0, 0.5),
                           linewidths=0.5, s=25)
                ax.scatter(*kp_pred_abs.T, marker='x', c='r', s=0.2, alpha=0.5)

            # Add wireframe overlay
            ax = axes[1, i]
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            self.projector.set_background(img)
            img_overlay = to_numpy(self.projector.generate_image() * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
            ax.imshow(img_overlay)
            ax.axis('off')

            # Show the target keypoints
            if self.args.use_keypoints and i == len(Xs) - 1 and self.keypoint_targets is not None:
                kp_target_abs = to_numpy(to_absolute_coordinates(self.keypoint_targets, img_overlay.shape[0]))
                ax.scatter(*kp_target_abs.T,
                           facecolors=(0, 0.7, 0, 0.2),
                           edgecolors=(0, 0.7, 0, 0.5),
                           linewidths=0.5, s=25)
                ax.scatter(*kp_target_abs.T, marker='x', c='g', s=0.2, alpha=0.5)

            # RCF features
            if rcf_feats is not None:
                for j in range(len(self.args.plot_rcf_feats)):
                    ax = axes[2 + j, i]
                    f_idx = self.args.plot_rcf_feats[j]
                    if i < len(rcf_feats[f_idx]):
                        f = rcf_feats[f_idx][i]
                        f = (f - f.min()) / (f.max() - f.min())
                        img = np.clip(to_numpy(f), 0, 1)
                        ax.imshow(img, cmap='Blues')
                    ax.axis('off')

        # Plot the distance parameters using custom method as the shared method gives a wierd performance hit
        self._plot_distances(axes[n_rows - 1, 0])

        # Plot the transformation and material parameters
        shared_args = dict(
            manager=self.manager,
            Y_pred={
                'transformation': torch.cat([
                    self.crystal.origin, self.crystal.scale[None, ...], self.crystal.rotation
                ]),
                'material': torch.cat([
                    self.crystal.material_ior[None, ...], self.crystal.material_roughness[None, ...]
                ]),
                'light': self.scene.light_radiance
            },
            colour_pred=self.args.plot_colour_pred,
            show_legend=False
        )
        plot_transformation(axes[n_rows - 1, 1], **shared_args)
        plot_material(axes[n_rows - 1, 2], **shared_args)
        if n_cols > 3:
            ax = axes[n_rows - 1, 3]
            plot_light(ax, **shared_args)
            ax.set_ylim(0, 1)

        fig.suptitle(f'Step {self.step + 1} Loss: {self.loss:.4E}')
        fig.tight_layout()
        return fig

    def _plot_distances(self, ax: Axes):
        """
        Plot the distances on the axis.
        """
        d_pred = to_numpy(self.crystal.distances)

        # Group asymmetric distances by face group
        distance_groups = {}
        grouped_order = []
        for i, hkl in enumerate(self.manager.ds.dataset_args.miller_indices):
            group_idxs = (self.symmetry_idx == i).nonzero().squeeze().tolist()
            distance_groups[hkl] = group_idxs
            grouped_order.extend(group_idxs)
        d_pred = d_pred[grouped_order]

        # Add bar chart data
        locs, bar_width, offset = _add_bars(
            ax=ax,
            pred=d_pred,
            colour_pred=self.args.plot_colour_pred,
        )

        # Reorder labels for asymmetric distances
        xlabels = []
        for i, (hkl, g) in enumerate(distance_groups.items()):
            group_labels = [''] * len(g)
            group_labels[len(g) // 2] = '(' + ''.join(map(str, hkl)) + ')'
            xlabels.extend(group_labels)

            # Add vertical separator lines between face groups
            if i < len(distance_groups) - 1:
                ax.axvline(locs[len(xlabels) - 1] + 0.5, color='black', linestyle='--', linewidth=1)

        # Replace -X with \bar{X} in labels
        xlabels = [re.sub(r'-(\d)', r'$\\bar{\1}$', label) for label in xlabels]

        ax.set_title('Distances')
        ax.set_xticks(locs)
        ax.set_xticklabels(xlabels)
        if len(xlabels) > 5:
            ax.tick_params(axis='x', labelsize='small')
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '', '1'])

    def _plot_patches(self) -> Figure:
        """
        Plot the target and optimised patches side by side.
        """
        X_target = self.X_target if isinstance(self.X_target, Tensor) else self.X_target[0]
        X_target = np.clip(to_numpy(X_target), 0, 1)
        X_pred = self.X_pred if isinstance(self.X_pred, Tensor) else self.X_pred[0]
        X_pred = np.clip(to_numpy(X_pred), 0, 1)

        n_cols = 5
        if self.args.n_patches > 0 and self.args.plot_n_patches == -1:
            n_rows = self.args.n_patches
        else:
            n_rows = min(self.args.n_patches, self.args.plot_n_patches)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.4), squeeze=False)
        for i in range(n_rows):
            x, y = self.patch_centres[i]
            ps = self.args.patch_size

            # Show the original target image with a patch overlay
            ax = axes[i, 0]
            ax.imshow(X_target)
            ax.scatter(x, y, c='darkblue', s=20, marker='x', alpha=0.7)
            rect = Rectangle((x - ps // 2, y - ps // 2), ps, ps, linewidth=0.7, edgecolor='darkblue',
                             facecolor='none', linestyle='--', alpha=0.7)
            ax.add_patch(rect)
            ax.axis('off')

            # Show the rendered image with a patch overlay
            sf = X_target.shape[0] / X_pred.shape[0]
            x, y = x / sf, y / sf
            ps = ps / sf
            ax = axes[i, 1]
            ax.imshow(X_pred)
            ax.scatter(x, y, c='darkblue', s=20, marker='x', alpha=0.7)
            rect = Rectangle((x - ps // 2, y - ps // 2), ps, ps, linewidth=0.7, edgecolor='darkblue',
                             facecolor='none', linestyle='--', alpha=0.7)
            ax.add_patch(rect)
            ax.axis('off')

            # Show the target patch
            ax = axes[i, 2]
            X_target_patch = np.clip(to_numpy(self.X_target_patches[i]), 0, 1)
            ax.imshow(X_target_patch)
            ax.axis('off')

            # Show the optimised patch
            ax = axes[i, 3]
            X_pred_patch = np.clip(to_numpy(self.X_pred_patches[i]), 0, 1)
            ax.imshow(X_pred_patch)
            ax.axis('off')

            # Show an L2 error map
            ax = axes[i, 4]
            error_map = ((X_pred_patch - X_target_patch)**2).sum(axis=-1)
            ax.imshow(error_map)
            ax.axis('off')

        fig.suptitle(f'Step {self.step + 1} Loss: {self.loss:.4E}')
        fig.tight_layout()
        return fig

    def _save_plot(self, fig: Figure, plot_type: str):
        """
        Log the figure to the tensorboard logger and optionally save it to disk.
        """
        # Save to disk
        save_dir = self.save_dir / plot_type
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f'{self.step + 1:08d}.png'
        plt.savefig(path, bbox_inches='tight')

        # Log to tensorboard
        if self.args.plot_to_tensorboard:
            self.tb_logger.add_figure(plot_type, fig, self.step)
            self.tb_logger.flush()

        plt.close(fig)
