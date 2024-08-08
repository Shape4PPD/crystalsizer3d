import gc
import math
import re
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import drjit as dr
import mitsuba as mi
import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from diffusers import LDMSuperResolutionPipeline
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
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
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.util.plots import _add_bars, plot_material, plot_transformation
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
    X_target_denoised: Optional[Union[Tensor, List[Tensor]]]
    X_target_aug: Union[Tensor, List[Tensor]]
    X_target_patches: Optional[Tensor]
    X_pred: Tensor = None
    X_pred_patches: Optional[Tensor] = None
    patch_centres: Optional[List[Tuple[int, int]]] = None

    scene: Scene = None
    scene_params: mi.SceneParameters = None

    crystal: Crystal = None
    symmetry_idx: Tensor
    conj_pairs: List[Tuple[int, int]]
    conj_switch_probs: Tensor

    anchors: Dict[ProjectedVertexKey, Tensor] = {}

    rcf_feats_og: Optional[List[Tensor]]
    rcf_feats: Optional[List[Tensor]]

    def __init__(
            self,
            args: RefinerArgs,
            output_dir: Optional[Path] = None,
    ):
        self.args = args

        # Seed
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Set up the log directory
        self._init_save_dir(output_dir)

        # Load the optimisation targets
        self._init_X_target()
        self._init_X_target_denoised()
        self._init_Y_target()

    def __getattr__(self, name: str):
        """
        Lazy loading of the components.
        """
        if name == 'save_dir':
            self._init_save_dir()
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
        elif name in ['optimiser', 'lr_scheduler']:
            self._init_optimiser()
            return getattr(self, name)
        elif name == 'metric_keys':
            self.metric_keys = self._init_metrics()
            return self.metric_keys
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

    def _init_save_dir(self, output_dir: Optional[Path] = None):
        """
        Set up the output directory.
        """
        logger.info('Initialising output directory.')

        if self.args.image_path is None:
            target_str = str(self.args.ds_idx)
        else:
            target_str = self.args.image_path.stem
        model_str = self.args.predictor_model_path.stem[:4] \
                    + (
                        f'_{self.args.denoiser_model_path.stem[:4]}' if self.args.denoiser_model_path is not None else '')
        base_params_str = (f'spp{self.args.spp}' +
                           f'_res{self.args.working_image_size}' +
                           (f'_ms' if self.args.multiscale else '') +
                           f'_{self.args.opt_algorithm}')
        if output_dir is None:
            base_dir = LOGS_PATH
        else:
            base_dir = output_dir
        base_dir = base_dir / model_str / target_str / base_params_str
        dir_name = hash_data(self.args.to_dict())

        # Check for existing directory
        if base_dir.exists():
            pattern = re.compile(rf'{dir_name}.*')
            existing_dirs = [d for d in base_dir.iterdir() if pattern.match(d.name)]
            if len(existing_dirs) > 0:
                logger.warning(f'Found existing directory: {existing_dirs[0]}. Overwriting.')
                cache_dir = existing_dirs[0] / 'cache'
                if cache_dir.exists():
                    shutil.move(cache_dir, base_dir / 'cache_tmp')
                shutil.rmtree(existing_dirs[0])

        # Make the new save directory
        self.save_dir = base_dir / (dir_name + f'_{START_TIMESTAMP}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if (base_dir / 'cache_tmp').exists():
            shutil.move(base_dir / 'cache_tmp', self.save_dir / 'cache')
        else:
            (self.save_dir / 'cache').mkdir()

        # Save arguments to yml file
        with open(self.save_dir / 'args.yml', 'w') as f:
            spec = self.args.to_dict()
            spec['created'] = START_TIMESTAMP
            yaml.dump(spec, f)

    def _init_manager(self):
        """
        Initialise the predictor and denoiser networks.
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
        manager.enable_eval()  # Should be on as default, but set it just in case

        # Load the denoiser model if set
        if self.args.denoiser_model_path is not None:
            assert self.args.denoiser_model_path.exists(), f'Denoiser model path does not exist: {self.args.denoiser_model_path}.'
            manager.load_network(self.args.denoiser_model_path, 'denoiser')

        self.manager = manager

    def _init_projector(self):
        """
        Initialise the projector.
        """
        self.projector = Projector(
            crystal=self.crystal,
            image_size=(self.args.working_image_size, self.args.working_image_size),
            zoom=orthographic_scale_factor(self.scene)
        )

    def _init_blur(self):
        """
        Initialise the Gaussian blur kernel
        """
        blur = GaussianBlur(kernel_size=5, sigma=1.0)
        blur = torch.jit.script(blur)
        blur.to(self.device)
        self.blur = blur

    def _init_superres_model(self):
        """
        Initialise the super-resolution model.
        """
        if self.args.superres_n_patches <= 0:
            self.superres_model = None
            return
        model = LDMSuperResolutionPipeline.from_pretrained(self.args.superres_model)
        model = model.to(self.device)
        self.superres_model = model

    def _init_perceptual_model(self):
        """
        Initialise the perceptual loss network.
        """
        if self.args.perceptual_model is None:
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
        if self.args.latents_model is None:
            self.latents_model = None
            return
        assert self.args.latents_model == 'MAGVIT2', 'Only MAGVIT2 encoder is supported.'

        # Monkey-patch the LPIPS class so that it loads from a sensible place
        from taming.modules.losses.lpips import LPIPS

        def load_pips(self, name='vgg_lpips'):
            ckpt = get_ckpt_path(name, DATA_PATH / 'vgg_lpips')
            self.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
            logger.info(f'Loaded pretrained LPIPS loss from {ckpt}.')

        LPIPS.load_from_pretrained = load_pips

        # Load the model checkpoint
        config = OmegaConf.load(self.args.mv2_config_path)
        self.dn_config = config
        model = VQModel(**config.model.init_args)
        sd = torch.load(self.args.mv2_checkpoint_path, map_location='cpu')['state_dict']
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
        if self.args.rcf_model_path is None:
            self.rcf = None
            return None
        rcf_path = self.args.rcf_model_path
        rcf = RCF()
        checkpoint = torch.load(rcf_path)
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
        model_input = self.X_target[None, ...].permute(0, 3, 1, 2)
        self.rcf_feats_og = self.rcf(model_input, apply_sigmoid=False)

    def _init_X_target(self):
        """
        Load the input image.
        """
        cache_path = self.save_dir / 'cache' / 'X_target.pt'
        if cache_path.exists():
            try:
                X_target = torch.load(cache_path)
                self.X_target = X_target.to(self.device)
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

            # Crop and resize the image to the working image size
            X_target = center_crop(X_target, min(X_target.shape[-2:]))
            X_target = F.interpolate(
                X_target[None, ...],
                size=self.manager.image_shape[-1],
                mode='bilinear',
                align_corners=False
            )[0]
        X_target = default_collate([X_target, ])
        X_target = X_target.to(self.device)

        # Resize target image to the working image size for inverse rendering
        wis = (self.args.working_image_size, self.args.working_image_size)
        X_target = F.interpolate(X_target, size=wis, mode='bilinear', align_corners=False)[0]
        X_target = X_target.permute(1, 2, 0)  # HWC

        # Multiscale
        if self.args.multiscale:
            X_target = to_multiscale(X_target, self.blur)
            resolution_pyramid = [t.shape[0] for t in X_target[::-1]]
            logger.info(f'Resolution pyramid has {len(X_target)} levels: '
                        f'{", ".join([str(res) for res in resolution_pyramid])}')

        # Save the target image
        torch.save(X_target, cache_path)
        self.X_target = X_target

    def _init_X_target_denoised(self):
        """
        Generate or load the denoised target.
        """
        cache_path = self.save_dir / 'cache' / 'X_target_denoised.pt'
        if cache_path.exists():
            try:
                X_target_denoised = torch.load(cache_path)
                self.X_target_denoised = X_target_denoised.to(self.device)
                logger.info('Loaded denoised target image.')
                return
            except Exception as e:
                logger.warning(f'Failed to load cached denoised target image: {e}')
                cache_path.unlink()

        # Denoise the input image if a denoiser is available
        if self.manager.denoiser is not None:
            logger.info('Denoising input image.')
            with torch.no_grad():
                X_denoised = denoise_image(
                    manager=self.manager,
                    X=self.X_target.permute(2, 0, 1),
                    n_tiles=self.args.denoiser_n_tiles,
                    overlap=self.args.denoiser_tile_overlap,
                    batch_size=self.args.denoiser_batch_size
                )[None, ...]

            # Destroy the denoiser to free up space
            logger.info('Destroying denoiser to free up space.')
            self.manager.denoiser = None
            del self.manager.denoiser
            torch.cuda.empty_cache()
            gc.collect()
        else:
            assert self.args.initial_pred_from != 'denoised', 'No denoiser set, so can\'t predict using a denoised image'
            return None

        # Resize target image to the working image size for inverse rendering
        wis = (self.args.working_image_size, self.args.working_image_size)
        X_denoised = F.interpolate(X_denoised, size=wis, mode='bilinear', align_corners=False)[0]
        X_denoised = X_denoised.permute(1, 2, 0)

        # Multiscale
        if self.args.multiscale:
            X_denoised = to_multiscale(X_denoised, self.blur)

        # Save the denoised target
        torch.save(X_denoised, cache_path)
        self.X_target_denoised = X_denoised

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

    def _init_optimiser(self):
        """
        Set up the optimiser and learning rate scheduler.
        """
        logger.info('Initialising optimiser.')
        optimiser = create_optimizer_v2(
            opt=self.args.opt_algorithm,
            weight_decay=0,
            model_or_params=[
                # {'params': [self.crystal.scale], 'lr': self.args.lr_scale},
                {'params': [self.crystal.distances], 'lr': self.args.lr_distances},
                {'params': [self.crystal.origin], 'lr': self.args.lr_origin},
                {'params': [self.crystal.rotation], 'lr': self.args.lr_rotation},
                {'params': [self.crystal.material_roughness, self.crystal.material_ior], 'lr': self.args.lr_material},
                # {'params': [self.crystal.material_roughness], 'lr': self.args.lr_material / 10},
                # {'params': [self.crystal.material_ior], 'lr': self.args.lr_material},
                {'params': [self.scene.light_radiance], 'lr': self.args.lr_light},
                {'params': [self.conj_switch_probs], 'lr': self.args.lr_switches},
            ],
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
            k_decay=self.args.lr_k_decay
        )

        self.optimiser = optimiser
        self.lr_scheduler = lr_scheduler

    def _init_metrics(self) -> List[str]:
        """
        Set up the metrics to track.
        """
        metric_keys = [
            'losses/l1',
            'losses/l2',
            'losses/perceptual',
            'losses/latent',
            'losses/rcf',
            'losses/anchors',
        ]
        return metric_keys

    def _init_tb_logger(self):
        """Initialise the tensorboard writer."""
        self.tb_logger = SummaryWriter(self.save_dir, flush_secs=5)

    def set_anchors(self, anchors: Dict[ProjectedVertexKey, Tensor]):
        """
        Set the manually-defined anchor points.
        """
        self.anchors = anchors

    def set_initial_scene(self, scene: Scene):
        """
        Set the initial scene data directly.
        """
        logger.info('Setting initial scene parameters.')
        self.scene = scene
        self.scene.crystal.to('cpu')
        self.scene_params = mi.traverse(self.scene.mi_scene)
        self.crystal = self.scene.crystal
        self._init_optimiser()  # Reinitialise the optimiser to include the new parameters

        # Render the scene to get the initial X_pred
        X_pred = scene.render(seed=get_seed())
        X_pred = cv2.cvtColor(X_pred, cv2.COLOR_RGB2BGR)  # todo: do we need this?
        X_pred = X_pred.astype(np.float32) / 255.
        X_pred = torch.from_numpy(X_pred).permute(2, 0, 1).to(self.device)
        X_pred = F.interpolate(
            X_pred[None, ...],
            size=self.args.working_image_size,
            mode='bilinear',
            align_corners=False
        )[0].permute(1, 2, 0)
        if self.args.multiscale:
            X_pred = to_multiscale(X_pred, self.blur)
        self.X_pred = X_pred

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
        if scene_path.exists():
            try:
                self.X_pred = torch.load(X_pred_path)
                self.scene = Scene.from_yml(scene_path)
                self.scene.crystal.to('cpu')
                self.scene_params = mi.traverse(self.scene.mi_scene)
                self.crystal = self.scene.crystal
                logger.info('Loaded initial prediction from cache.')
                return
            except Exception as e:
                logger.warning(f'Failed to load cached initial prediction: {e}')
                scene_path.unlink()
                X_pred_path.unlink()

        # Use either the original or denoised image as input
        logger.info('Predicting parameters.')
        if self.args.initial_pred_from == 'original':
            X_target = self.X_target
        elif self.args.initial_pred_from == 'denoised':
            assert self.X_target_denoised is not None, 'Denoised image not available.'
            X_target = self.X_target_denoised
        else:
            raise ValueError(f'Invalid initial_pred_from: {self.args.initial_pred_from}')
        self.X_target_aug = X_target
        if isinstance(X_target, list):
            X_target = X_target[0]

        # Create a batch
        bs = self.args.initial_pred_batch_size
        X_target_batch = X_target.permute(2, 0, 1).repeat(bs, 1, 1, 1)

        # Add some noise to the batch
        noise_scale = torch.linspace(
            self.args.initial_pred_noise_min,
            self.args.initial_pred_noise_max,
            bs,
            device=self.device
        )
        X_target_batch += torch.randn_like(X_target_batch) * noise_scale[:, None, None, None]

        # Predict the parameters
        Y_pred_batch = self.manager.predict(X_target_batch)

        # Destroy the predictor to free up space
        logger.info('Destroying predictor to free up space.')
        self.manager.predictor = None
        del self.manager.predictor
        torch.cuda.empty_cache()
        gc.collect()

        # Generate some images from the parameters
        X_pred_batch = []
        scene_batch = []
        losses = []
        for i in range(bs):
            # Render the image
            r_params = self.manager.ds.denormalise_rendering_params(Y_pred_batch, idx=i)
            X_pred_i, scene_i = self.manager.crystal_renderer.render_from_parameters(r_params, return_scene=True)
            scene_batch.append(scene_i)

            # Resize the image to the working image size
            X_pred_i = X_pred_i.astype(np.float32) / 255.
            X_pred_i = torch.from_numpy(X_pred_i).permute(2, 0, 1).to(self.device)
            X_pred_i = F.interpolate(
                X_pred_i[None, ...],
                size=self.args.working_image_size,
                mode='bilinear',
                align_corners=False
            )[0].permute(1, 2, 0)
            X_pred_batch.append(X_pred_i)

            # Multiscale
            if self.args.multiscale:
                X_pred_i = to_multiscale(X_pred_i, self.blur)

            # Calculate losses
            self.X_pred = X_pred_i
            self.crystal = scene_i.crystal
            loss_i, _ = self._calculate_losses()
            losses.append(loss_i.item())

            # Save image
            X = X_pred_i[0] if isinstance(X_pred_i, list) else X_pred_i
            img = Image.fromarray(to_numpy(X * 255).astype(np.uint8))
            img.save(save_dir / f'{i:02d}_noise={noise_scale[i]:.3f}_loss={loss_i:.4E}.png')

        # Pick the best image
        best_idx = np.argmin(losses)
        X_pred = X_pred_batch[best_idx]
        scene = scene_batch[best_idx]

        # Save the best initial prediction scene image
        img = Image.fromarray(to_numpy(X_pred * 255).astype(np.uint8))
        img.save(save_dir / f'best_idx={best_idx}.png')

        # Update the scene rendering parameters for optimisation
        wis = (self.args.working_image_size, self.args.working_image_size)
        scene.res = wis[0]
        scene.spp = self.args.spp
        scene.integrator_max_depth = self.args.integrator_max_depth
        scene.integrator_rr_depth = self.args.integrator_rr_depth
        scene.camera_type = 'perspective'  # thinlens doesn't work for inverse rendering
        scene.light_radiance = nn.Parameter(init_tensor(scene.light_radiance, device=scene.device), requires_grad=True)
        scene.build_mi_scene()
        scene_params = mi.traverse(scene.mi_scene)

        # Render the new scene
        img = scene.render()
        Image.fromarray(img).save(save_dir / f'best_idx={best_idx}_new_scene.png')

        # Move the crystal on to the CPU - faster mesh building
        scene.crystal.to('cpu')

        # Save variables
        scene.to_yml(scene_path)
        torch.save(X_pred, X_pred_path)

        # Set variables
        self.X_pred = X_pred
        self.scene = scene
        self.scene_params = scene_params
        self.crystal = scene.crystal

    def train(self, callback: Optional[callable] = None):
        """
        Train the parameters for a number of steps.
        """
        self._init_tb_logger()
        if self.scene is None:
            self.make_initial_prediction()

        logger.info(f'Training for {self.args.max_steps} steps.')
        logger.info(f'Logs path: {self.save_dir}.')
        n_steps = self.args.max_steps
        log_freq = self.args.log_every_n_steps
        running_loss = 0.
        running_metrics = {k: 0. for k in self.metric_keys}
        running_tps = 0

        for step in range(n_steps):
            start_time = time.time()
            self.step = step
            loss, stats = self._train_step()

            # Log the material parameter values
            self.tb_logger.add_scalar('params/roughness', self.crystal.material_roughness.item(), step)
            self.tb_logger.add_scalar('params/ior', self.crystal.material_ior.item(), step)

            # Log the conjugate switching probabilities
            if self.args.use_conj_switching:
                for i, (pair, prob) in enumerate(zip(self.conj_pairs, self.conj_switch_probs)):
                    ab = ','.join([f'{k}' for k in self.crystal.all_miller_indices[pair[0]][:2].tolist()])
                    self.tb_logger.add_scalar(f'params/conj_switch_probs/{ab}', prob.item(), step)

            # Log learning rates and update them
            for i, param_group in enumerate(['distances', 'origin', 'rotation', 'material', 'light']):
                self.tb_logger.add_scalar(f'lr/{param_group}', self.optimiser.param_groups[i]['lr'], step)
            if self.args.lr_scheduler != 'none':
                self.lr_scheduler.step(step, loss)

            # Track running loss and metrics
            running_loss += loss
            for k in self.metric_keys:
                if k in stats:
                    running_metrics[k] += stats[k]
            time_per_step = time.time() - start_time
            running_tps += time_per_step

            # Log statistics every X steps
            if (step + 1) % log_freq == 0:
                log_msg = f'[{step + 1}/{n_steps}]' \
                          f'\tLoss: {running_loss / log_freq:.4E}'
                for k, v in running_metrics.items():
                    k = k.replace('losses/', '')
                    log_msg += f'\t{k}: {v / log_freq:.4E}'
                logger.info(log_msg)
                running_loss = 0.
                running_metrics = {k: 0. for k in self.metric_keys}
                seconds_left = float((self.args.max_steps - step) * time_per_step / log_freq)
                logger.info('Time per step: {}, Est. complete in: {}'.format(
                    str(timedelta(seconds=time_per_step)),
                    str(timedelta(seconds=seconds_left))))

            # Plots
            self._make_plots(force=step == 0)

            # Callback
            if callback is not None:
                continue_signal = callback(step)
                if continue_signal is False:
                    logger.info('Received stop signal. Stopping training.')
                    break

        # Final plots
        self._make_plots(force=True)

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
            if self.args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_([self.crystal.distances], max_norm=self.args.clip_grad_norm)
            self.optimiser.step()
            self.optimiser.zero_grad()
            self.crystal.clamp_parameters(rescale=False)
            self.conj_switch_probs.data = torch.clamp(
                self.conj_switch_probs,
                min=self.args.conj_switch_prob_min,
                max=self.args.conj_switch_prob_max
            )

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
        device = self.scene.device
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
        v, f = v.to(device), f.to(device)
        eta = ior.to(device) / self.scene.crystal_material_bsdf['ext_ior']
        roughness = roughness.to(device).clone()
        radiance = radiance.to(device).clone()

        # Render new image
        X_pred = self._render_image(v, f, eta, roughness, radiance, seed=self.step)
        X_pred = torch.clip(X_pred, 0, 1)
        if self.args.multiscale:
            X_pred = to_multiscale(X_pred, self.blur)
        self.X_pred = X_pred

        # Use denoised target if available
        X_target = self.X_target_denoised if self.X_target_denoised is not None else self.X_target

        # Add some noise to the target image
        if isinstance(X_target, list):
            X_target_aug = [X + torch.randn_like(X) * self.args.image_noise_std for X in X_target]
            X_target_aug = [X.clip(0, 1) for X in X_target_aug]
        else:
            X_target_aug = X_target + torch.randn_like(X_target) * self.args.image_noise_std
            X_target_aug.clip_(0, 1)
        self.X_target_aug = X_target_aug

        # Calculate losses
        loss, stats = self._calculate_losses(distances=distances)
        self.loss = loss.item()

        return loss, stats

    @dr.wrap_ad(source='torch', target='drjit')
    def _render_image(self, vertices, faces, eta, roughness, radiance, seed=1):
        self.scene_params[Scene.VERTEX_KEY] = dr.ravel(vertices)
        self.scene_params[Scene.FACES_KEY] = dr.ravel(faces)
        self.scene_params[Scene.ETA_KEY] = dr.ravel(eta)
        self.scene_params[Scene.ROUGHNESS_KEY] = dr.ravel(roughness)
        self.scene_params[Scene.RADIANCE_KEY] = dr.unravel(mi.Color3f, radiance)
        self.scene_params.update()
        return mi.render(self.scene.mi_scene, self.scene_params, seed=seed)

    def _calculate_losses(
            self,
            is_patch: bool = False,
            distances: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate losses.
        """
        l1_loss, l1_stats = self._img_loss(X_target=self.X_target_aug, X_pred=self.X_pred, loss_type='l1',
                                           decay_factor=self.args.l_decay_l1)
        l2_loss, l2_stats = self._img_loss(X_target=self.X_target_aug, X_pred=self.X_pred, loss_type='l2',
                                           decay_factor=self.args.l_decay_l2)
        percept_loss, percept_stats = self._perceptual_loss()
        latent_loss, latent_stats = self._latents_loss()
        rcf_loss, rcf_stats = self._rcf_loss(is_patch=is_patch)
        overshoot_loss, overshoot_stats = self._overshoot_loss(distances=distances)
        symmetry_loss, symmetry_stats = self._symmetry_loss()
        z_pos_loss, z_pos_stats = self._z_pos_loss()
        rxy_loss, rxy_stats = self._rotation_xy_loss()
        if not is_patch:
            patch_loss, patch_stats = self._patches_loss()
        switch_loss, switch_stats = self._switch_loss()
        anchors_loss, anchors_stats = self._anchors_loss()

        # Combine losses
        loss = l1_loss * self.args.w_img_l1 \
               + l2_loss * self.args.w_img_l2 \
               + percept_loss * self.args.w_perceptual \
               + latent_loss * self.args.w_latent \
               + rcf_loss * self.args.w_rcf \
               + overshoot_loss * self.args.w_overshoot \
               + symmetry_loss * self.args.w_symmetry \
               + z_pos_loss * self.args.w_z_pos \
               + rxy_loss * self.args.w_rotation_xy \
               + switch_loss * self.args.w_switch_probs \
               + anchors_loss * self.args.w_anchors

        if not is_patch:
            loss = loss * self.args.w_fullsize + patch_loss * self.args.w_patches
        assert not is_bad(loss), 'Bad loss!'

        stats = {
            'losses/total': loss.item(),
            **l1_stats, **l2_stats, **percept_stats, **latent_stats, **rcf_stats, **overshoot_stats, **symmetry_stats,
            **z_pos_stats, **rxy_stats, **switch_stats, **anchors_stats
        }
        if not is_patch:
            stats.update(patch_stats)

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
                assert len(
                    X_target) == 1, f'Target image must be a single image, not a batch. {len(X_target)} received.'
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
        if self.perceptual_model is None:
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
        if self.latents_model is None:
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
        input_size = self.dn_config.data.init_args.train.params.config.size
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
        if self.rcf is None:
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
        loss = torch.tensor(0., device=self.device)
        if self.symmetry_idx is None:
            return loss, {}
        for i, hkl in enumerate(self.manager.ds.dataset_args.miller_indices):
            group_idxs = (self.symmetry_idx == i).nonzero().squeeze().tolist()
            if len(group_idxs) == 1:
                continue
            d_group = self.crystal.distances[group_idxs]
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
        if self.args.superres_n_patches is None or self.args.superres_n_patches <= 0:
            return loss, stats

        ps = self.args.superres_patch_size
        ps2 = ps // 2
        wis = self.args.working_image_size

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

        # Use the L1 error map to select the best patches
        l1 = (X_pred - X_target).abs().mean(dim=-1).detach()
        patch_centres = []
        X_pred_patches = []
        X_target_patches = []
        for i in range(self.args.superres_n_patches):
            with torch.no_grad():
                # Calculate the sum of errors for each possible patch position
                l1_patch_sums = F.conv2d(l1[None, ...], torch.ones(1, 1, ps, ps).to(self.device), padding=ps2)[0]
                if l1_patch_sums.shape[0] != wis:
                    l1_patch_sums = F.interpolate(
                        l1_patch_sums[None, None, ...],
                        size=wis,
                        mode='bilinear',
                        align_corners=False
                    )[0, 0]

                # Find the index of the maximum error sum
                max_idx = torch.argmax(l1_patch_sums).item()
                y = max_idx // wis
                x = max_idx % wis
                y = min(max(ps2, y), wis - ps2)
                x = min(max(ps2, x), wis - ps2)
                patch_centres.append((x, y))

            # Crop patches
            X_pred_patch = X_pred[y - ps2:y + ps2, x - ps2:x + ps2]
            X_target_patch = X_target[y - ps2:y + ps2, x - ps2:x + ps2]
            X_pred_patches.append(X_pred_patch.detach())
            X_target_patches.append(X_target_patch.detach())

            # Reduce the error map around the selected patch
            l1[y - ps2:y + ps2, x - ps2:x + ps2] /= 1.5

        # Upscale the patches to the working image size
        patches = torch.stack(X_pred_patches + X_target_patches).permute(0, 3, 1, 2)
        upscaled_patches = F.interpolate(patches, size=wis, mode='bilinear', align_corners=False)

        # Calculate losses for each pair of patches
        self.X_pred = upscaled_patches[:self.args.superres_n_patches].permute(0, 2, 3, 1)
        self.X_target_aug = upscaled_patches[self.args.superres_n_patches:].permute(0, 2, 3, 1)
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
        self.X_pred_patches = torch.stack(X_pred_patches)
        self.X_target_patches = torch.stack(X_target_patches)

        return loss, stats

    def _switch_loss(self) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the conjugate switching loss.
        """
        loss = torch.tensor(0., device=self.device)
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

    def _anchors_loss(self):
        """
        Calculate the loss for manual constraints.
        """
        loss = torch.tensor(0., device=self.device)
        stats = {}
        if len(self.anchors) == 0:
            return loss, stats

        # Project the vertices
        self.projector.project(generate_image=False)

        # Calculate the loss for each anchor
        losses = []
        for vertex_key, anchor_coords in self.anchors.items():
            # Vertex is visible, so calculate the distance between the vertex and the target anchor location
            if vertex_key in self.projector.projected_vertex_keys:
                idx = self.projector.projected_vertex_keys.index(vertex_key)
                v_coords = self.projector.projected_vertex_coords[idx]
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
            # if self.patch_centres is not None:
            #     fig = self._plot_patches()
            #     self._save_plot(fig, 'patches')

    @torch.no_grad()
    def _plot_comparison(self) -> Figure:
        """
        Plot the target and optimised images side by side.
        """
        X_target_og = self.X_target if isinstance(self.X_target, Tensor) else self.X_target[0]
        X_target_aug = self.X_target_aug if isinstance(self.X_target_aug, Tensor) else self.X_target_aug[0]
        X_pred = [self.X_pred] if isinstance(self.X_pred, Tensor) else self.X_pred
        Xs = [X_target_og, X_target_aug, *X_pred]

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

        n_cols = 3
        if self.args.plot_n_samples == -1:
            n_cols += len(X_pred) - 1
        else:
            n_cols += min(len(X_pred) - 1, self.args.plot_n_samples)

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
            if i == 2 and self.patch_centres is not None:
                ps = self.args.superres_patch_size
                ax.scatter(*np.array(self.patch_centres).T, c='r', s=25, marker='x')
                for x, y in self.patch_centres:
                    ax.scatter(x, y, c='r', s=25, marker='x')
                    rect = Rectangle((x - ps // 2, y - ps // 2), ps, ps, linewidth=1, edgecolor='r', facecolor='none',
                                     linestyle='--')
                    ax.add_patch(rect)

            # Add wireframe overlay
            ax = axes[1, i]
            self.projector.set_background(img)
            img_overlay = to_numpy(self.projector.project() * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
            ax.imshow(img_overlay)
            ax.axis('off')

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
                'transformation': torch.cat(
                    [self.crystal.origin, self.crystal.scale[None, ...], self.crystal.rotation]),
                'material': torch.cat(
                    [self.crystal.material_ior[None, ...], self.crystal.material_roughness[None, ...]])
            },
            colour_pred=self.args.plot_colour_pred,
            show_legend=False
        )
        plot_transformation(axes[n_rows - 1, 1], **shared_args)
        plot_material(axes[n_rows - 1, 2], **shared_args)

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
        X_target = self.X_target_aug if isinstance(self.X_target_aug, Tensor) else self.X_target_aug[0]
        X_target = np.clip(to_numpy(X_target), 0, 1)
        X_pred = self.X_pred if isinstance(self.X_pred, Tensor) else self.X_pred[0]
        X_pred = np.clip(to_numpy(X_pred), 0, 1)
        ps = self.args.superres_patch_size

        n_cols = 4
        if self.args.superres_n_patches > 0 and self.args.plot_n_patches == -1:
            n_rows = self.args.superres_n_patches
        else:
            n_rows = min(self.args.superres_n_patches, self.args.plot_n_patches)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.4), squeeze=False)
        for i in range(n_rows):
            x, y = self.patch_centres[i]

            # Show the augmented target image with a patch overlay
            ax = axes[i, 0]
            ax.imshow(X_target)
            ax.scatter(x, y, c='r', s=25, marker='x')
            rect = Rectangle((x - ps // 2, y - ps // 2), ps, ps, linewidth=1, edgecolor='r', facecolor='none',
                             linestyle='--')
            ax.add_patch(rect)
            ax.axis('off')

            # Show the optimised image with a patch overlay
            ax = axes[i, 1]
            ax.imshow(X_pred)
            ax.scatter(x, y, c='r', s=25, marker='x')
            rect = Rectangle((x - ps // 2, y - ps // 2), ps, ps, linewidth=1, edgecolor='r', facecolor='none',
                             linestyle='--')
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
        self.tb_logger.add_figure(plot_type, fig, self.step)
        self.tb_logger.flush()

        plt.close(fig)
