import gc
import json
import math
import shutil
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry import axis_angle_to_rotation_matrix
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from taming.models.lfqgan import VQModel
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from taming.util import get_ckpt_path
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.scheduler.scheduler import Scheduler
from torch import Tensor, nn
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crystalsizer3d import DATA_PATH, LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs
from crystalsizer3d.args.denoiser_args import DenoiserArgs
from crystalsizer3d.args.generator_args import GeneratorArgs
from crystalsizer3d.args.network_args import NetworkArgs
from crystalsizer3d.args.optimiser_args import OptimiserArgs
from crystalsizer3d.args.runtime_args import RuntimeArgs
from crystalsizer3d.args.transcoder_args import TranscoderArgs
from crystalsizer3d.crystal import Crystal, ROTATION_MODE_AXISANGLE, ROTATION_MODE_QUATERNION
from crystalsizer3d.crystal_generator import CrystalGenerator
from crystalsizer3d.crystal_renderer import CrystalRenderer
from crystalsizer3d.nn.checkpoint import Checkpoint
from crystalsizer3d.nn.data_loader import get_data_loader
from crystalsizer3d.nn.dataset import Dataset
from crystalsizer3d.nn.models.basenet import BaseNet
from crystalsizer3d.nn.models.densenet import DenseNet
from crystalsizer3d.nn.models.discriminator import Discriminator
from crystalsizer3d.nn.models.fcnet import FCNet
from crystalsizer3d.nn.models.generatornet import GeneratorNet
from crystalsizer3d.nn.models.gigagan import GigaGAN
from crystalsizer3d.nn.models.pyramidnet import PyramidNet
from crystalsizer3d.nn.models.rcf import RCF
from crystalsizer3d.nn.models.resnet import ResNet
from crystalsizer3d.nn.models.timmnet import TimmNet
from crystalsizer3d.nn.models.transcoder import Transcoder
from crystalsizer3d.nn.models.transcoder_mask_inv import TranscoderMaskInv
from crystalsizer3d.nn.models.transcoder_tvae import TranscoderTVAE
from crystalsizer3d.nn.models.transcoder_vae import TranscoderVAE
from crystalsizer3d.nn.models.vit_pretrained import ViTPretrainedNet
from crystalsizer3d.nn.models.vitvae import ViTVAE
from crystalsizer3d.util.ema import EMA
from crystalsizer3d.util.geometry import geodesic_distance
from crystalsizer3d.util.plots import plot_denoiser_samples, plot_generator_samples, plot_training_samples
from crystalsizer3d.util.polyhedron import calculate_polyhedral_vertices
from crystalsizer3d.util.utils import calculate_model_norm, is_bad, set_seed


# torch.autograd.set_detect_anomaly(True)


class Manager:
    ds: Dataset
    train_loader: DataLoader
    test_loader: DataLoader
    crystal_generator: CrystalGenerator
    crystal_renderer: CrystalRenderer
    predictor: Optional[BaseNet]
    generator: Optional[GeneratorNet]
    discriminator: Optional[Discriminator]
    transcoder: Optional[Transcoder]
    optimiser_p: Optimizer
    lr_scheduler_p: Scheduler
    optimiser_g: Optional[Optimizer]
    lr_scheduler_g: Optional[Scheduler]
    optimiser_d: Optional[Optimizer]
    lr_scheduler_d: Optional[Scheduler]
    optimiser_t: Optional[Optimizer]
    lr_scheduler_t: Optional[Scheduler]
    metric_keys: List[str]
    device: torch.device
    checkpoint: Checkpoint
    crystal: Crystal

    def __init__(
            self,
            runtime_args: RuntimeArgs,
            dataset_args: DatasetTrainingArgs,
            net_args: NetworkArgs,
            generator_args: GeneratorArgs,
            denoiser_args: DenoiserArgs,
            transcoder_args: TranscoderArgs,
            optimiser_args: OptimiserArgs,
            save_dir: Optional[Path] = None,
            print_networks: bool = True
    ):
        # Argument groups
        self.runtime_args = runtime_args
        self.dataset_args = dataset_args
        self.net_args = net_args
        self.generator_args = generator_args
        self.denoiser_args = denoiser_args
        self.transcoder_args = transcoder_args
        self.optimiser_args = optimiser_args

        # Save dir
        self.save_dir = save_dir

        # Debug
        self.print_networks = print_networks

        # Seed
        if self.runtime_args.seed is not None:
            set_seed(self.runtime_args.seed)

        # Load checkpoint
        self.checkpoint = self._init_checkpoint(save_dir=save_dir)

    @property
    def image_shape(self) -> Tuple[int, ...]:
        n_channels = 5 if self.dataset_args.add_coord_grid else 3
        return n_channels, self.ds.image_size, self.ds.image_size

    @property
    def parameters_shape(self) -> Tuple[int, ...]:
        return self.ds.label_size,

    @property
    def logs_path(self) -> Path:
        return self.checkpoint.save_dir / self.checkpoint.id

    def __getattr__(self, name: str):
        """
        Lazy loading of the components.
        """
        if name == 'ds':
            self.ds = Dataset(self.dataset_args)
            if self.ds.dataset_args.asymmetry is not None:
                assert self.dataset_args.check_symmetries == 0, \
                    f'Asymmetry is set, so check_symmetries must be 0 (received {self.dataset_args.check_symmetries})'
            return self.ds
        elif name in ['train_loader', 'test_loader']:
            self.train_loader, self.test_loader = self._init_data_loaders()
            return self.train_loader
        elif name == 'crystal_generator':
            self.crystal_generator = self._init_crystal_generator()
            return self.crystal_generator
        elif name == 'crystal_renderer':
            self.crystal_renderer = self._init_crystal_renderer()
            return self.crystal_renderer
        elif name == 'predictor':
            self.predictor = self._init_predictor()
            return self.predictor
        elif name == 'generator':
            self.generator = self._init_generator()
            return self.generator
        elif name == 'denoiser':
            self.denoiser = self._init_denoiser()
            return self.denoiser
        elif name == 'discriminator':
            self.discriminator = self._init_discriminator()
            return self.discriminator
        elif name == 'transcoder':
            self.transcoder = self._init_transcoder()
            return self.transcoder
        elif name == 'rcf':
            self.rcf = self._init_rcf()
            return self.rcf
        elif name in ['optimiser_p', 'lr_scheduler_p',
                      'optimiser_g', 'lr_scheduler_g',
                      'optimiser_d', 'lr_scheduler_d',
                      'optimiser_dn', 'lr_scheduler_dn',
                      'optimiser_t', 'lr_scheduler_t']:
            (self.optimiser_p, self.lr_scheduler_p,
             self.optimiser_g, self.lr_scheduler_g,
             self.optimiser_d, self.lr_scheduler_d,
             self.optimiser_dn, self.lr_scheduler_dn,
             self.optimiser_t, self.lr_scheduler_t) = self._init_optimisers()
            return getattr(self, name)
        elif name == 'metric_keys':
            self.metric_keys = self._init_metrics()
            return self.metric_keys
        elif name == 'device':
            self.device = self._init_devices()
            return self.device
        elif name == 'crystal':
            self.crystal = self._init_crystal()
            return self.crystal
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def load(
            cls,
            model_path: Path,
            args_changes: Dict[str, Dict[str, Any]],
            save_dir: Optional[Path] = None,
            print_networks: bool = False
    ) -> 'Manager':
        """
        Instantiate a manager from a checkpoint json file.
        """
        logger.info(f'Loading model from {model_path}.')
        with open(model_path, 'r') as f:
            data = json.load(f)

        # Ensure all the required arguments are present
        required_args = ['runtime_args', 'dataset_args', 'network_args',
                         'generator_args', 'denoiser_args', 'transcoder_args',
                         'optimiser_args']
        for arg in required_args:
            if arg not in data:
                data[arg] = {}

        # Always set resume to True, otherwise the model won't load
        data['runtime_args']['resume'] = True
        data['runtime_args']['resume_only'] = True
        data['runtime_args']['resume_from'] = model_path

        # Don't check the dataset paths as we probably aren't using them
        data['dataset_args']['check_paths'] = False

        args = dict(
            runtime_args=RuntimeArgs.from_args(data['runtime_args']),
            dataset_args=DatasetTrainingArgs.from_args(data['dataset_args']),
            net_args=NetworkArgs.from_args(data['network_args']),
            generator_args=GeneratorArgs.from_args(data['generator_args']),
            denoiser_args=DenoiserArgs.from_args(data['denoiser_args']),
            transcoder_args=TranscoderArgs.from_args(data['transcoder_args']),
            optimiser_args=OptimiserArgs.from_args(data['optimiser_args']),
            print_networks=print_networks
        )

        # Update the arguments with required changes
        for arg_group, arg_changes in args_changes.items():
            for k, v in arg_changes.items():
                setattr(args[arg_group], k, v)

        return cls(**args, save_dir=save_dir)

    def _init_crystal_generator(self) -> CrystalGenerator:
        """
        Initialise the crystal generator.
        """
        dsa = self.ds.dataset_args
        generator = CrystalGenerator(
            crystal_id=dsa.crystal_id,
            miller_indices=dsa.miller_indices,
            ratio_means=dsa.ratio_means,
            ratio_stds=dsa.ratio_stds,
            zingg_bbox=dsa.zingg_bbox,
            constraints=dsa.distance_constraints,
            asymmetry=dsa.asymmetry,
        )
        return generator

    def _init_crystal_renderer(self) -> CrystalRenderer:
        """
        Initialise the crystal renderer.
        """
        renderer = CrystalRenderer(
            param_path=self.ds.path / 'parameters.csv',
            dataset_args=self.ds.dataset_args,
            quiet_render=True
        )
        return renderer

    def _init_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get the data loaders.
        """
        logger.info('Initialising data loaders.')
        loaders = {}
        for tt in ['train', 'test']:
            loaders[tt] = get_data_loader(
                ds=self.ds,
                dst_args=self.dataset_args,
                train_or_test=tt,
                batch_size=self.runtime_args.batch_size,
                n_workers=self.runtime_args.n_dataloader_workers,
                prefetch_factor=self.runtime_args.prefetch_factor
            )

        return loaders['train'], loaders['test']

    def _init_predictor(self) -> Optional[BaseNet]:
        """
        Build the predictor network using the given parameters.
        """
        if not self.dataset_args.train_predictor:
            return None

        net_args = self.net_args
        output_shape = self.parameters_shape if not self.transcoder_args.use_transcoder \
            else (self.transcoder_args.tc_latent_size,)

        params = {**{
            'network_type': net_args.base_net,
            'input_shape': self.image_shape,
            'output_shape': output_shape,
        }, **net_args.hyperparameters}

        shared_args = dict(
            input_shape=self.image_shape,
            output_shape=output_shape,
            build_model=True
        )

        # Separate classes are used to validate the different available hyperparameters
        if net_args.base_net == 'fcnet':
            predictor = FCNet(
                layers_config=params['layers_config'],
                dropout_prob=params['dropout_prob'],
                **shared_args
            )
        elif net_args.base_net == 'resnet':
            predictor = ResNet(
                n_init_channels=params['n_init_channels'],
                block_config=params['blocks_config'],
                shortcut_type=params['shortcut_type'],
                use_bottlenecks=params['use_bottlenecks'],
                dropout_prob=params['dropout_prob'],
                **shared_args
            )
        elif net_args.base_net == 'densenet':
            predictor = DenseNet(
                n_init_channels=params['n_init_channels'],
                growth_rate=params['growth_rate'],
                block_config=params['blocks_config'],
                compression_factor=params['compression_factor'],
                dropout_prob=params['dropout_prob'],
                **shared_args
            )
        elif net_args.base_net == 'pyramidnet':
            predictor = PyramidNet(
                n_init_channels=params['n_init_channels'],
                block_config=params['blocks_config'],
                shortcut_type=params['shortcut_type'],
                use_bottlenecks=params['use_bottlenecks'],
                alpha=params['alpha'],
                dropout_prob=params['dropout_prob'],
                **shared_args
            )
        elif net_args.base_net == 'vitnet':
            predictor = ViTPretrainedNet(
                model_name=params['model_name'],
                use_cls_token=params['use_cls_token'],
                classifier_hidden_layers=params['classifier_hidden_layers'],
                classifier_dropout_prob=params['classifier_dropout_prob'],
                vit_dropout_prob=params['vit_dropout_prob'],
                **shared_args
            )
        elif net_args.base_net == 'timm':
            predictor = TimmNet(
                model_name=params['model_name'],
                classifier_hidden_layers=params['classifier_hidden_layers'],
                dropout_prob=params['dropout_prob'],
                droppath_prob=params['droppath_prob'],
                classifier_dropout_prob=params['classifier_dropout_prob'],
                **shared_args
            )
        else:
            raise ValueError(f'Unrecognised base net: {net_args.base_net}')

        # Instantiate the network
        logger.info(f'Instantiated predictor network with {predictor.get_n_params() / 1e6:.4f}M parameters.')
        if net_args.base_net in ['vitnet', 'timm']:
            logger.info(f'Classifier has {predictor.get_n_classifier_params() / 1e6:.4f}M parameters.')
        if self.print_networks:
            logger.debug(f'----------- Predictor Network --------------\n\n{predictor}\n\n')
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            predictor = nn.DataParallel(predictor)
        predictor.to(self.device)

        # Instantiate an exponential moving average tracker for the predictor loss
        self.p_loss_ema = EMA()

        return predictor

    def _init_generator(self) -> Optional[GeneratorNet]:
        """
        Initialise the generator network.
        """
        if not self.dataset_args.train_generator and not self.dataset_args.train_combined:
            return None

        if hasattr(self.predictor, 'img_mean'):
            img_mean = self.predictor.img_mean.mean().item()
            img_std = self.predictor.img_std.mean().item()
        else:
            img_mean = 0.5
            img_std = 0.5

        n_inputs = self.parameters_shape[0] \
                   + (len(self.ds.labels_distances_active) if self.generator_args.gen_include_face_areas else 0)
        input_shape = (n_inputs,) if not self.transcoder_args.use_transcoder \
            else (self.transcoder_args.tc_latent_size,)
        shared_args = dict(
            input_shape=input_shape,
            output_shape=self.image_shape,
            latent_size=self.generator_args.gen_latent_size,
            img_mean=img_mean,
            img_std=img_std,
            build_model=True
        )

        if self.generator_args.gen_model_name == 'dcgan':
            generator = GeneratorNet(
                n_blocks=self.generator_args.dcgan_n_blocks,
                n_base_filters=self.generator_args.dcgan_n_base_filters,
                **shared_args
            )
        elif self.generator_args.gen_model_name == 'gigagan':
            generator = GigaGAN(
                oversample=self.generator_args.ggan_oversample,
                dim_capacity=self.generator_args.ggan_dim_capacity,
                dim_max=self.generator_args.ggan_dim_max,
                self_attn_resolutions=self.generator_args.ggan_self_attn_resolutions,
                self_attn_dim_head=self.generator_args.ggan_self_attn_dim_head,
                self_attn_heads=self.generator_args.ggan_self_attn_heads,
                self_attn_ff_mult=self.generator_args.ggan_self_attn_ff_mult,
                cross_attn_resolutions=self.generator_args.ggan_cross_attn_resolutions,
                cross_attn_dim_head=self.generator_args.ggan_cross_attn_dim_head,
                cross_attn_heads=self.generator_args.ggan_cross_attn_heads,
                cross_attn_ff_mult=self.generator_args.ggan_cross_attn_ff_mult,
                **shared_args
            )
        elif self.generator_args.gen_model_name == 'vitvae':
            generator = ViTVAE(
                oversample=self.generator_args.vitvae_oversample,
                n_layers=self.generator_args.vitvae_n_layers,
                patch_size=self.generator_args.vitvae_patch_size,
                dim_head=self.generator_args.vitvae_dim_head,
                heads=self.generator_args.vitvae_heads,
                ff_mult=self.generator_args.vitvae_ff_mult,
                **shared_args
            )
        else:
            raise ValueError(f'Unrecognised generator net: {self.generator_args.gen_model_name}')

        # Instantiate the network
        logger.info(f'Instantiated generator network with {generator.get_n_params() / 1e6:.4f}M parameters.')
        if self.print_networks:
            logger.debug(f'----------- Generator Network --------------\n\n{generator}\n\n')
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            generator = nn.DataParallel(generator)
        generator.to(self.device)

        # Instantiate an exponential moving average tracker for the generator loss
        self.g_loss_ema = EMA()

        # Instantiate the RCF model here too
        self.rcf = self._init_rcf()

        return generator

    def _init_discriminator(self) -> Optional[Discriminator]:
        """
        Initialise the discriminator network.
        """
        if not self.dataset_args.train_generator or not self.generator_args.use_discriminator:
            return None

        discriminator = Discriminator(
            input_shape=self.image_shape,
            output_shape=(1,),
            n_base_filters=self.generator_args.disc_n_base_filters,
            n_layers=self.generator_args.disc_n_layers,
            loss_type=self.optimiser_args.gan_loss,
            build_model=True
        )

        # Instantiate the network
        logger.info(f'Instantiated discriminator network with {discriminator.get_n_params() / 1e6:.4f}M parameters.')
        if self.print_networks:
            logger.debug(f'----------- Discriminator Network --------------\n\n{discriminator}\n\n')
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            discriminator = nn.DataParallel(discriminator)
        discriminator.to(self.device)

        # Instantiate an exponential moving average tracker for the discriminator loss
        self.d_loss_ema = EMA()

        return discriminator

    def _init_denoiser(self) -> Optional[VQModel]:
        """
        Initialise the denoiser network.
        """
        if not self.dataset_args.train_denoiser:
            return None
        assert self.denoiser_args.dn_model_name == 'MAGVIT2', 'Only MAGVIT2 denoiser is supported.'

        # Monkey-patch the LPIPS class so it loads from a sensible place
        from taming.modules.losses.lpips import LPIPS

        def load_pips(self, name='vgg_lpips'):
            ckpt = get_ckpt_path(name, DATA_PATH / 'vgg_lpips')
            self.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
            logger.info(f'Loaded pretrained LPIPS loss from {ckpt}.')

        LPIPS.load_from_pretrained = load_pips

        # Load the model checkpoint
        config = OmegaConf.load(self.denoiser_args.dn_mv2_config_path)
        self.dn_config = config
        denoiser = VQModel(**config.model.init_args)
        sd = torch.load(self.denoiser_args.dn_mv2_checkpoint_path, map_location='cpu')['state_dict']
        missing, unexpected = denoiser.load_state_dict(sd, strict=False)
        denoiser.eval()

        # Instantiate the network
        n_params = sum([p.data.nelement() for p in denoiser.parameters()])
        logger.info(f'Instantiated denoiser network with {n_params / 1e6:.4f}M parameters.')
        if self.print_networks:
            logger.debug(f'----------- Denoiser Network --------------\n\n{denoiser}\n\n')
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            denoiser = nn.DataParallel(denoiser)
        denoiser.to(self.device)

        return denoiser

    def _init_transcoder(self) -> Optional[Transcoder]:
        """
        Initialise the transcoder network.
        """
        if not self.transcoder_args.use_transcoder:
            return None

        shared_args = dict(
            latent_size=self.transcoder_args.tc_latent_size,
            param_size=self.parameters_shape[0],
            latent_activation=self.transcoder_args.tc_latent_activation,
        )

        if self.transcoder_args.tc_model_name == 'mask_inv':
            transcoder = TranscoderMaskInv(
                ds=self.ds,
                normalise_latents=self.transcoder_args.tc_normalise_latents,
                **shared_args
            )
        elif self.transcoder_args.tc_model_name == 'vae':
            transcoder = TranscoderVAE(
                layers_config=self.transcoder_args.tc_vae_layers,
                **shared_args
            )
        elif self.transcoder_args.tc_model_name == 'tvae':
            transcoder = TranscoderTVAE(
                dim_enc=self.transcoder_args.tc_dim_enc,
                dim_dec=self.transcoder_args.tc_dim_dec,
                num_heads=self.transcoder_args.tc_num_heads,
                dim_feedforward=self.transcoder_args.tc_dim_feedforward,
                num_layers=self.transcoder_args.tc_num_layers,
                depth_enc=self.transcoder_args.tc_depth_enc,
                depth_dec=self.transcoder_args.tc_depth_dec,
                dropout=self.transcoder_args.tc_dropout_prob,
                activation=self.transcoder_args.tc_activation,
                **shared_args
            )
        else:
            raise ValueError(f'Unrecognised transcoder net: {self.transcoder_args.tc_model_name}')

        # Instantiate the network
        logger.info(f'Instantiated transcoder network with {transcoder.get_n_params() / 1e6:.4f}M parameters.')
        if self.print_networks:
            logger.debug(f'----------- Transcoder Network --------------\n\n{transcoder}\n\n')
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            transcoder = nn.DataParallel(transcoder)
        transcoder.to(self.device)

        # Instantiate exponential moving average trackers for the reconstruction and regularisation losses
        self.tc_rec_loss_ema = EMA()
        self.tc_kl_loss_ema = EMA()

        return transcoder

    def _init_rcf(self) -> Optional[RCF]:
        """
        Initialise the Richer Convolutional Features model for edge detection.
        """
        if not self.generator_args.use_rcf:
            return None
        rcf_path = self.generator_args.rcf_model_path
        rcf = RCF()
        checkpoint = torch.load(rcf_path)
        rcf.load_state_dict(checkpoint, strict=False)
        rcf.eval()

        # Instantiate the network
        n_params = sum([p.data.nelement() for p in rcf.parameters()])
        logger.info(f'Instantiated RCF network with {n_params / 1e6:.4f}M parameters from {rcf_path}.')
        if self.print_networks:
            logger.debug(f'----------- RCF Network --------------\n\n{rcf}\n\n')
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            rcf = nn.DataParallel(rcf)
        else:
            rcf = torch.jit.script(rcf)
        rcf.to(self.device)

        return rcf

    def _init_optimisers(self) -> Tuple[
        Optimizer, Scheduler,
        Optional[Optimizer], Optional[Scheduler],
        Optional[Optimizer], Optional[Scheduler],
        Optional[Optimizer], Optional[Scheduler],
        Optional[Optimizer], Optional[Scheduler]
    ]:
        """
        Set up the optimisers and learning rate schedulers.
        """
        logger.info('Initialising optimisers.')
        ra = self.runtime_args
        oa = self.optimiser_args

        shared_opt_args = dict(
            opt=oa.algorithm,
            weight_decay=oa.weight_decay,
        )

        # For cycle based schedulers (cosine, tanh, poly) adjust total epochs for cycles and cooldown
        if oa.lr_scheduler in ['cosine', 'tanh', 'poly']:
            cycles = max(1, oa.lr_cycle_limit)
            if oa.lr_cycle_mul == 1.0:
                n_epochs_with_cycles = ra.n_epochs * cycles
            else:
                n_epochs_with_cycles = int(math.floor(-ra.n_epochs * (oa.lr_cycle_mul**cycles - 1)
                                                      / (1 - oa.lr_cycle_mul)))
            n_epochs_adj = math.ceil(ra.n_epochs * (ra.n_epochs - oa.lr_cooldown_epochs) / n_epochs_with_cycles)
        else:
            n_epochs_adj = ra.n_epochs
        n_epochs_p = n_epochs_g = n_epochs_d = n_epochs_dn = n_epochs_t = 0

        shared_lrs_args = dict(
            sched=oa.lr_scheduler,
            num_epochs=n_epochs_adj,
            decay_epochs=oa.lr_decay_epochs,
            decay_milestones=oa.lr_decay_milestones,
            cooldown_epochs=oa.lr_cooldown_epochs,
            patience_epochs=oa.lr_patience_epochs,
            decay_rate=oa.lr_decay_rate,
            min_lr=oa.lr_min,
            warmup_lr=oa.lr_warmup,
            warmup_epochs=oa.lr_warmup_epochs,
            cycle_mul=oa.lr_cycle_mul,
            cycle_decay=oa.lr_cycle_decay,
            cycle_limit=oa.lr_cycle_limit,
            k_decay=oa.lr_k_decay
        )

        # Create the optimiser and scheduler for the predictor network
        if self.dataset_args.train_predictor:
            # For the pretrained models, separate the feature extractor from the classifier
            if self.net_args.base_net in ['vitnet', 'timm']:
                params_p = [
                    {'params': self.predictor.classifier.parameters(), 'lr': oa.lr_init},
                ]
                if not oa.freeze_pretrained:
                    params_p.append(
                        {'params': self.predictor.model.parameters(), 'lr': oa.lr_pretrained_init}
                    )
            else:
                params_p = [
                    {'params': self.predictor.parameters(), 'lr': oa.lr_init},
                ]
            if self.transcoder_args.use_transcoder and self.transcoder_args.tc_trained_by in ['predictor', 'both']:
                params_p.append(
                    {'params': self.transcoder.parameters(), 'lr': oa.lr_init}
                )

            if self.dataset_args.train_combined:
                params_p.append(
                    {'params': self.generator.parameters(), 'lr': oa.lr_generator_init}
                )

            optimiser_p = create_optimizer_v2(model_or_params=params_p, **shared_opt_args)
            lr_scheduler_p, n_epochs_p = create_scheduler_v2(optimizer=optimiser_p, **shared_lrs_args)
        else:
            optimiser_p = None
            lr_scheduler_p = None

        # Create a separate optimiser for the generator network
        if self.dataset_args.train_generator and not self.dataset_args.train_combined:
            params_g = [
                {'params': self.generator.parameters(), 'lr': oa.lr_generator_init}
            ]
            if self.transcoder_args.use_transcoder and self.transcoder_args.tc_trained_by in ['generator', 'both']:
                params_g.append(
                    {'params': self.transcoder.parameters(), 'lr': oa.lr_generator_init}
                )
            optimiser_g = create_optimizer_v2(model_or_params=params_g, **shared_opt_args)
            lr_scheduler_g, n_epochs_g = create_scheduler_v2(optimizer=optimiser_g, **shared_lrs_args)
        else:
            optimiser_g = None
            lr_scheduler_g = None

        # Create a separate optimiser for the discriminator network
        if self.dataset_args.train_generator and self.generator_args.use_discriminator:
            params_d = [
                {'params': self.discriminator.parameters(), 'lr': oa.lr_discriminator_init}
            ]
            optimiser_d = create_optimizer_v2(model_or_params=params_d, **shared_opt_args)
            lr_scheduler_d, n_epochs_d = create_scheduler_v2(optimizer=optimiser_d, **shared_lrs_args)
        else:
            optimiser_d = None
            lr_scheduler_d = None

        # Create a separate optimiser for the denoiser network
        if self.dataset_args.train_denoiser:
            params_dn = [
                {'params': self.denoiser.encoder.parameters(), 'lr': oa.lr_denoiser_init},
                {'params': self.denoiser.decoder.parameters(), 'lr': oa.lr_denoiser_init},
                {'params': self.denoiser.quantize.parameters(), 'lr': oa.lr_denoiser_init},
            ]
            optimiser_dn = create_optimizer_v2(model_or_params=params_dn, betas=(0.5, 0.9), **shared_opt_args)
            lr_scheduler_dn, n_epochs_dn = create_scheduler_v2(optimizer=optimiser_dn, **shared_lrs_args)
        else:
            optimiser_dn = None
            lr_scheduler_dn = None

        # Create a separate optimiser for the VAE transcoder network (if used)
        if self.transcoder_args.use_transcoder and self.transcoder_args.tc_trained_by == 'self':
            params_t = [
                {'params': self.transcoder.parameters(), 'lr': oa.lr_transcoder_init}
            ]
            optimiser_t = create_optimizer_v2(model_or_params=params_t, **shared_opt_args)
            lr_scheduler_t, n_epochs_t = create_scheduler_v2(optimizer=optimiser_t, **shared_lrs_args)
        else:
            optimiser_t = None
            lr_scheduler_t = None

        # Check that the total number of epochs is the same for all optimisers
        n_epochs = np.array([n_epochs_p, n_epochs_g, n_epochs_d, n_epochs_dn, n_epochs_t])
        if np.all(n_epochs == 0):
            raise RuntimeError('No optimisers were created!')
        assert np.allclose(n_epochs[n_epochs > 0] / ra.n_epochs, 1, atol=0.05)

        return optimiser_p, lr_scheduler_p, optimiser_g, lr_scheduler_g, optimiser_d, lr_scheduler_d, \
            optimiser_dn, lr_scheduler_dn, optimiser_t, lr_scheduler_t

    def _init_metrics(self) -> List[str]:
        """
        Set up the metrics to track.
        """
        metric_keys = []
        if self.dataset_args.train_zingg:
            metric_keys.append('losses/zingg')
        if self.dataset_args.train_distances:
            metric_keys.append('losses/distances')
        if self.dataset_args.train_transformation:
            metric_keys.append('losses/transformation')
        if self.dataset_args.train_3d:
            metric_keys.append('losses/3d')
        if self.dataset_args.train_material:
            metric_keys.append('losses/material')
        if self.dataset_args.train_light:
            metric_keys.append('losses/light')
        if self.dataset_args.train_generator:
            metric_keys.append('losses/generator')
            # metric_keys.append('losses/teacher_net')
            # metric_keys.append('losses/teacher_gen')
            if self.generator_args.use_discriminator:
                metric_keys.append('losses/gan_disc')
                metric_keys.append('losses/gan_gen')
        if self.dataset_args.train_denoiser:
            metric_keys.append('losses/denoiser')
        if self.transcoder_args.use_transcoder:
            metric_keys.append('losses/transcoder')

        return metric_keys

    def _init_devices(self):
        """
        Find available devices and try to use what we want.
        """
        if self.runtime_args.use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise RuntimeError('GPU is not available')
        else:
            device = torch.device('cpu')

        if device.type == 'cuda':
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                logger.info(f'Using {n_gpus} GPUs!')
            else:
                logger.info('Using GPU.')
            cudnn.benchmark = True  # optimises code for constant input sizes

            # Move modules to the gpu
            for k, v in vars(self).items():
                if isinstance(v, torch.nn.Module):
                    v.to(device)
        else:
            logger.info('Using CPU.')

        return device

    def _init_checkpoint(self, save_dir: Optional[Path] = None) -> Checkpoint:
        """
        The current checkpoint instance contains the most up-to-date stats of the model.
        """
        logger.info('Initialising checkpoint.')
        if save_dir is None:
            save_dir = LOGS_PATH / f'ds={self.dataset_args.dataset_path.name}'
        checkpoint = Checkpoint(
            dataset=self.ds,
            dataset_args=self.dataset_args,
            network_args=self.net_args,
            generator_args=self.generator_args,
            denoiser_args=self.denoiser_args,
            optimiser_args=self.optimiser_args,
            transcoder_args=self.transcoder_args,
            runtime_args=self.runtime_args,
            save_dir=save_dir
        )

        # Load the network and optimiser parameter states
        if self.runtime_args.resume and len(checkpoint.snapshots) > 0:
            state_path = checkpoint.get_state_path()
            logger.info(f'Loading network parameters from {state_path}.')
            state = torch.load(state_path, map_location=self.device)

            # Load predictor network parameters and optimiser state
            if self.dataset_args.train_predictor:
                if self.net_args.base_net in ['vitnet', 'timm'] and self.optimiser_args.freeze_pretrained:
                    # Load only the classifier parameters if the pretrained part is frozen
                    self.predictor.classifier.load_state_dict(self._fix_state(state['net_p_state_dict']), strict=False)
                else:
                    self.predictor.load_state_dict(self._fix_state(state['net_p_state_dict']), strict=False)
                self.optimiser_p.load_state_dict(state['optimiser_p_state_dict'])

            # Load generator network parameters and optimiser state
            if self.dataset_args.train_generator:
                self.generator.load_state_dict(self._fix_state(state['net_g_state_dict']), strict=False)
                if not self.dataset_args.train_combined:
                    self.optimiser_g.load_state_dict(state['optimiser_g_state_dict'])

            # Load discriminator network parameters and optimiser state
            if self.dataset_args.train_generator and self.generator_args.use_discriminator:
                self.discriminator.load_state_dict(self._fix_state(state['net_d_state_dict']), strict=False)
                self.optimiser_d.load_state_dict(state['optimiser_d_state_dict'])

            # Load denoiser network parameters and optimiser state
            if self.dataset_args.train_denoiser:
                self.denoiser.load_state_dict(self._fix_state(state['net_dn_state_dict']), strict=False)
                self.optimiser_dn.load_state_dict(state['optimiser_dn_state_dict'])

            # Load transcoder network parameters
            if self.transcoder_args.use_transcoder:
                self.transcoder.load_state_dict(self._fix_state(state['net_t_state_dict']), strict=False)
                if self.optimiser_t is not None and 'optimiser_t_state_dict' in state:
                    self.optimiser_t.load_state_dict(state['optimiser_t_state_dict'])

            # Put all networks in evaluation mode
            self.enable_eval()

        elif self.runtime_args.resume_only:
            raise RuntimeError('Could not resume!')

        return checkpoint

    def _fix_state(self, state):
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        return new_state

    def _init_crystal(self) -> Crystal:
        """
        Initialise the crystal object.
        """
        cs = self.ds.csd_proxy.load(self.ds.dataset_args.crystal_id)
        crystal = Crystal(
            lattice_unit_cell=cs.lattice_unit_cell,
            lattice_angles=cs.lattice_angles,
            miller_indices=self.ds.dataset_args.miller_indices,
            point_group_symbol=cs.point_group_symbol,
            rotation_mode=self.dataset_args.rotation_mode,
            merge_vertices=True,
        )
        crystal.to(self.device)
        return crystal

    def _init_tb_logger(self):
        """Initialise the tensorboard writer."""
        self.tb_logger = SummaryWriter(self.logs_path / 'events' / START_TIMESTAMP, flush_secs=5)

    def configure_paths(self, renew_logs: bool = False):
        """Create the directories."""
        if renew_logs:
            logger.warning('Removing previous log files...')
            shutil.rmtree(self.logs_path, ignore_errors=True)
        self.logs_path.mkdir(exist_ok=True)
        (self.logs_path / 'snapshots').mkdir(exist_ok=True)
        (self.logs_path / 'events').mkdir(exist_ok=True)
        (self.logs_path / 'plots').mkdir(exist_ok=True)

    def log_graph(self):
        """
        Log the graph to tensorboard.
        """
        logger.info('Logging computation graph to tensorboard.')
        with torch.no_grad():
            dummy_input = torch.rand((self.train_loader.batch_size,) + self.image_shape)
            dummy_input = dummy_input.to(self.device)
            if not hasattr(self, 'tb_logger'):
                self._init_tb_logger()
            self.tb_logger.add_graph(self.predictor, [dummy_input, ], verbose=False)
            self.tb_logger.flush()

    def enable_eval(self):
        """
        Put the networks in evaluation mode.
        """
        if self.dataset_args.train_predictor:
            self.predictor.eval()
        if self.generator is not None:
            self.generator.eval()
            if self.discriminator is not None:
                self.discriminator.eval()
        if self.denoiser is not None:
            self.denoiser.eval()
        if self.transcoder is not None:
            self.transcoder.eval()

    def enable_train(self):
        """
        Put the networks in training mode.
        """
        if self.dataset_args.train_predictor:
            self.predictor.train()
        if self.dataset_args.train_generator:
            self.generator.train()
            if self.generator_args.use_discriminator:
                self.discriminator.train()
        if self.dataset_args.train_denoiser:
            self.denoiser.train()
        if self.transcoder_args.use_transcoder:
            self.transcoder.train()

    def train(self, n_epochs: int):
        """
        Train the network for a number of epochs.
        """
        self.configure_paths()
        self._init_tb_logger()
        logger.info(f'Logs path: {self.logs_path}.')
        starting_epoch = self.checkpoint.epoch + 1
        final_epoch = starting_epoch + n_epochs - 1

        # Set up schedulers
        if self.lr_scheduler_p is not None:
            self.lr_scheduler_p.step(starting_epoch - 1)
        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.step(starting_epoch - 1)
        if self.lr_scheduler_d is not None:
            self.lr_scheduler_d.step(starting_epoch - 1)
        if self.lr_scheduler_dn is not None:
            self.lr_scheduler_dn.step(starting_epoch - 1)
        if self.lr_scheduler_t is not None:
            self.lr_scheduler_t.step(starting_epoch - 1)

        for epoch in range(starting_epoch, final_epoch + 1):
            logger.info('{:-^80}'.format(' Train epoch: {} '.format(epoch)))
            self.checkpoint.epoch = epoch
            start_time = time.time()
            if self.optimiser_p is not None:
                self.tb_logger.add_scalar('lr/p', self.optimiser_p.param_groups[0]['lr'], epoch)
            if self.optimiser_g is not None:
                self.tb_logger.add_scalar('lr/g', self.optimiser_g.param_groups[0]['lr'], epoch)
            if self.optimiser_d is not None:
                self.tb_logger.add_scalar('lr/d', self.optimiser_d.param_groups[0]['lr'], epoch)
            if self.optimiser_dn is not None:
                self.tb_logger.add_scalar('lr/dn', self.optimiser_dn.param_groups[0]['lr'], epoch)
            if self.optimiser_t is not None:
                self.tb_logger.add_scalar('lr/t', self.optimiser_t.param_groups[0]['lr'], epoch)

            # Train for an epoch
            self._train_epoch(final_epoch)
            time_per_epoch = time.time() - start_time
            seconds_left = float((final_epoch - epoch) * time_per_epoch)
            logger.info('Time per epoch: {}, Est. complete in: {}'.format(
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))
            if self.lr_scheduler_p is not None:
                self.lr_scheduler_p.step(epoch, self.checkpoint.loss_train)
            if self.lr_scheduler_g is not None:
                self.lr_scheduler_g.step(epoch, self.checkpoint.loss_train)
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step(epoch, self.checkpoint.loss_train)
            if self.lr_scheduler_dn is not None:
                self.lr_scheduler_dn.step(epoch, self.checkpoint.loss_train)
            if self.lr_scheduler_t is not None:
                self.lr_scheduler_t.step(epoch, self.checkpoint.loss_train)

            # Test every n epochs
            if self.runtime_args.test_every_n_epochs > 0 \
                    and (epoch + 1) % self.runtime_args.test_every_n_epochs == 0:
                self.test()
                self.tb_logger.add_scalar(f'epoch/test/total', self.checkpoint.loss_test, epoch)
                for key, val in self.checkpoint.metrics_test.items():
                    self.tb_logger.add_scalar(f'epoch/test/{key}', val, epoch)
                    logger.info(f'Test {key}: {val:.4E}')

            # Update checkpoint and create a snapshot every n epochs
            self.checkpoint.save(
                create_snapshot=self.runtime_args.checkpoint_every_n_epochs > 0 and epoch % self.runtime_args.checkpoint_every_n_epochs == 0.,
                net_p=self.predictor,
                optimiser_p=self.optimiser_p,
                net_g=self.generator,
                optimiser_g=self.optimiser_g,
                net_d=self.discriminator,
                optimiser_d=self.optimiser_d,
                net_dn=self.denoiser,
                optimiser_dn=self.optimiser_dn,
                net_t=self.transcoder,
                optimiser_t=self.optimiser_t,
            )

        logger.info('Training complete.')

    def _train_epoch(self, final_epoch: int):
        """
        Train for a single epoch
        """
        log_freq = self.runtime_args.log_every_n_batches
        num_batches_per_epoch = len(self.train_loader)
        running_loss = 0.
        running_metrics = {k: 0. for k in self.metric_keys}
        epoch_loss = 0.
        epoch_metrics = {k: 0. for k in self.metric_keys}

        for i, data in enumerate(self.train_loader, 0):
            for _ in range(self.runtime_args.steps_per_batch):
                outputs, loss, stats = self._train_batch(data)

            running_loss += loss
            epoch_loss += loss
            for k in self.metric_keys:
                if k in stats:
                    running_metrics[k] += stats[k]
                    epoch_metrics[k] += stats[k]

            # Log statistics every X mini-batches
            if (i + 1) % log_freq == 0:
                batches_loss_avg = running_loss / log_freq
                log_msg = f'[{self.checkpoint.epoch}/{final_epoch}][{i + 1}/{num_batches_per_epoch}]' \
                          f'\tLoss: {batches_loss_avg:.4E}'
                for k, v in running_metrics.items():
                    avg = v / log_freq
                    log_msg += f'\t{k}: {avg:.4E}'
                logger.info(log_msg)
                running_loss = 0.
                running_metrics = {k: 0. for k in self.metric_keys}

            # Plots and checkpoints
            self._make_plots(data, outputs, stats, train_or_test='train')
            if self.runtime_args.checkpoint_every_n_batches > 0 \
                    and (i + 1) % self.runtime_args.checkpoint_every_n_batches == 0:
                self.checkpoint.save(
                    create_snapshot=True,
                    net_p=self.predictor,
                    optimiser_p=self.optimiser_p,
                    net_g=self.generator,
                    optimiser_g=self.optimiser_g,
                    net_d=self.discriminator,
                    optimiser_d=self.optimiser_d,
                    net_dn=self.denoiser,
                    optimiser_dn=self.optimiser_dn,
                    net_t=self.transcoder,
                    optimiser_t=self.optimiser_t,
                )

        gc.collect()

        # Update stats and write debug
        self.checkpoint.loss_train = float(epoch_loss) / num_batches_per_epoch
        self.checkpoint.metrics_train = epoch_metrics
        self.tb_logger.add_scalar('epoch/train/total_loss', self.checkpoint.loss_train, self.checkpoint.epoch)
        for key, val in epoch_metrics.items():
            self.tb_logger.add_scalar(f'epoch/train/{key}', val, self.checkpoint.epoch)
            logger.info(f'Train {key}: {val:.4E}')

        # End-of-epoch plots
        self._make_plots(data, outputs, stats, train_or_test='train', end_of_epoch=True)

    def _train_batch(
            self,
            data: Tuple[dict, Tensor, Tensor, Optional[Tensor], Dict[str, Tensor]]
    ) -> Tuple[dict, float, dict]:
        """
        Train on a single batch of data.
        """
        self.enable_train()
        outputs = {}
        stats = {}
        loss_total = 0.

        def train_net(net, opt, batch_fn, loss_key, batch_fn_args=None, do_step=True):
            nonlocal loss_total
            if batch_fn_args is None:
                batch_fn_args = {}
            outputs_, loss_, stats_ = batch_fn(data, **batch_fn_args)
            (loss_ / self.optimiser_args.accumulate_grad_batches).backward()
            if (self.checkpoint.step + 1) % self.optimiser_args.accumulate_grad_batches == 0:
                if self.optimiser_args.clip_grad_norm != -1:
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.optimiser_args.clip_grad_norm)
                if do_step:
                    opt.step()
                opt.zero_grad()
            outputs.update(outputs_)
            stats.update(stats_)
            loss_total += loss_.item()
            self.tb_logger.add_scalar(f'batch/train/loss_{loss_key}', loss_.item(), self.checkpoint.step)

        # Process the batch, calculate gradients and do optimisation step
        if self.dataset_args.train_predictor:
            train_net(self.predictor, self.optimiser_p, self._process_batch_predictor, 'pred')

        # Do the same for the generator network
        if self.dataset_args.train_generator and not self.dataset_args.train_combined:
            train_net(self.generator, self.optimiser_g, self._process_batch_generator, 'gen')

        # Only train the discriminator when it is doing worse than some threshold
        if self.dataset_args.train_generator and self.generator_args.use_discriminator:
            train_net(self.discriminator, self.optimiser_d, self._process_batch_generator, 'disc',
                      do_step=self.d_loss_ema.val > self.optimiser_args.disc_loss_threshold)

        # Train the denoiser
        if self.dataset_args.train_denoiser:
            train_net(self.denoiser, self.optimiser_dn, self._process_batch_denoiser, 'dn')

        # Train the transcoder
        if self.transcoder_args.use_transcoder and self.transcoder_args.tc_trained_by == 'self':
            train_net(self.transcoder, self.optimiser_t, self._process_batch_transcoder, 'tc',
                      batch_fn_args={'Z_pred': outputs['Z_pred'] if 'Z_pred' in outputs else None})

        # Log losses
        self.tb_logger.add_scalar('batch/train/total_loss', loss_total, self.checkpoint.step)
        for key, val in stats.items():
            self.tb_logger.add_scalar(f'batch/train/{key}', float(val), self.checkpoint.step)

        # Calculate L2 loss for predictor network
        if self.dataset_args.train_predictor:
            weights_cumulative_norm = calculate_model_norm(self.predictor, device=self.device)
            assert not is_bad(weights_cumulative_norm), 'Bad parameters! (Predictor network)'
            self.tb_logger.add_scalar('batch/train/w_norm', weights_cumulative_norm.item(), self.checkpoint.step)

        # Calculate L2 loss for generator network
        if self.dataset_args.train_generator:
            weights_cumulative_norm = calculate_model_norm(self.generator, device=self.device)
            assert not is_bad(weights_cumulative_norm), 'Bad parameters! (Generator network)'
            self.tb_logger.add_scalar('batch/train/w_norm_gen', weights_cumulative_norm.item(), self.checkpoint.step)

        # Calculate L2 loss for discriminator network
        if self.dataset_args.train_generator and self.generator_args.use_discriminator:
            weights_cumulative_norm = calculate_model_norm(self.discriminator, device=self.device)
            assert not is_bad(weights_cumulative_norm), 'Bad parameters! (Discriminator network)'
            self.tb_logger.add_scalar('batch/train/w_norm_disc', weights_cumulative_norm.item(), self.checkpoint.step)

        # Calculate L2 loss for denoiser network
        if self.dataset_args.train_denoiser:
            weights_cumulative_norm = calculate_model_norm(self.denoiser, device=self.device)
            assert not is_bad(weights_cumulative_norm), 'Bad parameters! (Denoiser network)'
            self.tb_logger.add_scalar('batch/train/w_norm_dn', weights_cumulative_norm.item(), self.checkpoint.step)

        # Calculate L2 loss for transcoder network
        if self.transcoder_args.use_transcoder:
            weights_cumulative_norm = calculate_model_norm(self.transcoder, device=self.device)
            assert not is_bad(weights_cumulative_norm), 'Bad parameters! (Transcoder network)'
            self.tb_logger.add_scalar('batch/train/w_norm_tc', weights_cumulative_norm.item(), self.checkpoint.step)

        # Increment global step counter
        self.checkpoint.step += 1
        self.checkpoint.examples_count += self.runtime_args.batch_size

        return outputs, loss_total, stats

    def test(self) -> Tuple[float, Dict]:
        """
        Test across the whole test dataset.
        """
        if not len(self.test_loader):
            raise RuntimeError('No test data available, cannot test!')
        logger.info('Testing.')
        log_freq = self.runtime_args.log_every_n_batches
        self.enable_eval()
        cumulative_loss = 0.
        cumulative_stats = {k: 0. for k in self.metric_keys}

        with torch.no_grad():
            for i, data in enumerate(self.test_loader, 0):
                if (i + 1) % log_freq == 0:
                    logger.info(f'Testing batch {i + 1}/{len(self.test_loader)}')
                loss_total = 0.
                outputs = {}
                stats = {}

                if self.dataset_args.train_predictor:
                    outputs_p, loss_p, stats_p = self._process_batch_predictor(data)
                    outputs.update(outputs_p)
                    stats.update(stats_p)
                    loss_total += loss_p.item()

                if self.dataset_args.train_generator and not self.dataset_args.train_combined:
                    outputs_g, loss_g, stats_g = self._process_batch_generator(data)
                    outputs.update(outputs_g)
                    stats.update(stats_g)
                    loss_total += loss_g.item()

                if self.dataset_args.train_generator and self.generator_args.use_discriminator:
                    outputs_d, loss_d, stats_d = self._process_batch_discriminator(data, outputs['X_pred'])
                    outputs.update(outputs_d)
                    stats.update(stats_d)
                    loss_total += loss_d.item()

                if self.dataset_args.train_denoiser:
                    outputs_dn, loss_dn, stats_dn = self._process_batch_denoiser(data)
                    outputs.update(outputs_dn)
                    stats.update(stats_dn)
                    loss_total += loss_dn.item()

                if self.transcoder_args.use_transcoder and self.transcoder_args.tc_trained_by == 'self':
                    outputs_t, loss_t, stats_t = self._process_batch_transcoder(
                        data,
                        outputs['Z_pred'] if 'Z_pred' in outputs else None
                    )
                    outputs.update(outputs_t)
                    stats.update(stats_t)
                    loss_total += loss_t.item()

                for k in self.metric_keys:
                    if k in stats:
                        cumulative_stats[k] += stats[k]
                cumulative_loss += loss_total

        test_loss = cumulative_loss / len(self.test_loader)
        test_stats = {k: v / len(self.test_loader) for k, v in cumulative_stats.items()}

        self.checkpoint.loss_test = float(test_loss)
        self.checkpoint.metrics_test = test_stats

        self._make_plots(data, outputs, stats, train_or_test='test', end_of_epoch=True)

        return test_loss, test_stats

    def _process_batch_predictor(
            self,
            data: Tuple[dict, Tensor, Tensor, Optional[Tensor], Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Any], Tensor, Dict]:
        """
        Take a batch of images, predict the parameters and calculate the average loss per example.
        """
        _, _, X_target_aug, X_target_clean, Y_target = data  # Use the (possibly) augmented image as input
        X_target_aug = X_target_aug.to(self.device)
        if X_target_clean is not None:
            X_target_clean = X_target_clean.to(self.device)
        Y_target = {
            k: [vi.to(self.device) for vi in v] if isinstance(v, list) else v.to(self.device)
            for k, v in Y_target.items()
        }

        # Predict the parameters and calculate losses
        Y_pred = self.predict(X_target_aug)
        loss, metrics, X_pred2 = self.calculate_predictor_losses(Y_pred, Y_target, X_target_clean)
        outputs = {
            'Y_pred': Y_pred,
            'X_pred2': X_pred2.detach().cpu() if X_pred2 is not None else None
        }

        # Update the EMA of the predictor loss
        self.p_loss_ema(loss.item())

        # If training the full auto encoder variant, then also do the generation step
        if self.dataset_args.train_combined:
            metrics['loss_com/pred'] = loss.item()
            Z = self.transcoder.latents_in
            outputs['Z_pred'] = Z.clone().detach()
            loss_g, metrics_g, Y_pred2 = self.calculate_generator_losses(X_pred2, X_target_clean, Y_target,
                                                                         include_teacher_loss=False,
                                                                         include_transcoder_loss=False)
            metrics.update(metrics_g)
            metrics['loss_gen'] = loss_g.item()
            self.g_loss_ema(metrics['losses/generator'])

            # Replace loss with the generator loss plus the latents consistency loss
            l_z = self._calculate_com_loss(Z, Y_target=Y_target)
            metrics['loss_com/X'] = l_z.item()
            loss = loss_g + self.optimiser_args.w_com_X * l_z

            # Also do the generation step from the target parameters
            with torch.no_grad():
                X_pred = self.generate(Y_target)
                outputs['X_pred'] = X_pred.detach().cpu()
                Y_pred2 = self.predict(X_pred)
                outputs['Y_pred2'] = Y_pred2

        return outputs, loss, metrics

    def _process_batch_generator(
            self,
            data: Tuple[dict, Tensor, Tensor, Tensor, Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Any], Tensor, Dict]:
        """
        Take a batch of input data, push it through the network and calculate the average loss per example.
        """
        if self.dataset_args.train_combined:
            raise RuntimeError('Training the generator separately in train_combined mode is disabled.')
        _, _, _, X_target, Y_target = data  # Use the clean image as the target
        X_target = X_target.to(self.device)
        Y_target = {
            k: [vi.to(self.device) for vi in v] if isinstance(v, list) else v.to(self.device)
            for k, v in Y_target.items()
        }

        # Generate an image from the parameters and calculate losses
        X_pred = self.generate(Y_target)
        loss, metrics, Y_pred2 = self.calculate_generator_losses(X_pred, X_target, Y_target)

        # Update the EMA of the generator loss
        self.g_loss_ema(metrics['losses/generator'])

        outputs = {
            'X_pred': X_pred,
            'Y_pred2': Y_pred2
        }

        return outputs, loss, metrics

    def _process_batch_discriminator(
            self,
            data: Tuple[dict, Tensor, Tensor, Tensor, Dict[str, Tensor]],
            X_pred: Tensor
    ) -> Tuple[Dict[str, Any], Tensor, Dict]:
        """
        Run the discriminator on the real and predicted images and calculate losses.
        """
        _, _, X_target, _, _ = data  # Use the clean image as target
        X_target = X_target.to(self.device)

        # Evaluate the real and fake images
        D_real = self.discriminator(X_target)
        D_fake = self.discriminator(X_pred.detach())

        # Calculate losses
        loss, metrics = self.calculate_discriminator_losses(D_real, D_fake)

        # Update the EMA of the discriminator loss
        self.d_loss_ema(metrics['losses/gan_disc'])

        outputs = {
            'D_real': D_real,
            'D_fake': D_fake,
        }

        return outputs, loss, metrics

    def _process_batch_denoiser(
            self,
            data: Tuple[dict, Tensor, Tensor, Tensor, Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Any], Tensor, Dict]:
        """
        Take a batch of noisy images and try to recover the clean images.
        """
        _, _, X_target_aug, X_target_clean, Y_target = data
        X_target_aug = X_target_aug.to(self.device)
        X_target_clean = X_target_clean.to(self.device)

        # Resize images
        dn_size = self.dn_config.data.init_args.train.params.config.size
        if self.image_shape[-1] != dn_size:
            X_target_aug = F.interpolate(X_target_aug, size=(dn_size, dn_size), mode='bilinear', align_corners=False)
            X_target_clean = F.interpolate(X_target_clean, size=(dn_size, dn_size), mode='bilinear',
                                           align_corners=False)

        # Denoise the image
        X_denoised, l_codebook, l_breakdown = self.denoiser(X_target_aug)

        # Calculate losses
        dn_loss: VQLPIPSWithDiscriminator = self.denoiser.loss

        # Reconstruction (denoising) loss - L1
        l_rec = (X_target_clean.contiguous() - X_denoised.contiguous()).abs().mean()

        # Perceptual loss
        if dn_loss.perceptual_weight > 0:
            l_percept = dn_loss.perceptual_loss(X_target_clean.contiguous(), X_denoised.contiguous()).mean()
        else:
            l_percept = torch.tensor(0., device=self.device)

        # Combine losses
        loss = l_rec \
               + dn_loss.perceptual_weight * l_percept \
               + dn_loss.codebook_weight * l_codebook \
               + dn_loss.commit_weight * l_breakdown.commitment

        outputs = {
            'X_denoised': X_denoised.detach().cpu(),
        }

        metrics = {
            'losses/denoiser': loss.item(),
            'dn/reconst': l_rec.item(),
            'dn/perceptual': l_percept.item(),
            'dn/codebook': l_codebook.item(),
            'dn/commitment': l_breakdown.commitment.item(),
        }

        return outputs, loss, metrics

    def _process_batch_transcoder(
            self,
            data: Tuple[dict, Tensor, Tensor, Tensor, Dict[str, Tensor]],
            Z_pred: Optional[Tensor] = None
    ) -> Tuple[Dict[str, Any], Tensor, Dict]:
        """
        Run the transcoder on the parameter to latent mappings.
        """

        # VAE transcoder
        if self.transcoder_args.tc_model_name in ['vae', 'tvae']:
            _, _, _, _, Y_target = data
            Y_target = {
                k: [vi.to(self.device) for vi in v] if isinstance(v, list) else v.to(self.device)
                for k, v in Y_target.items()
            }
            Yr_mu, Yr_logvar, Z_mu_logits, Z_logvar, Z = self._transcode_loop(
                Y_target, add_latent_noise=self.transcoder.training
            )
            Yr_mu_clean, _, _, _, _ = self._transcode_loop(
                Y_target, add_latent_noise=False
            )

            # Calculate losses
            loss, metrics = self.calculate_vae_transcoder_losses(
                Y_target, Yr_mu, Yr_mu_clean, Z_mu_logits, Z_logvar
            )
            if self.transcoder_args.tc_ae_variant == 'denoising':
                metrics['tc/noise_level'] = np.exp(-(self.tc_rec_loss_ema.val / self.transcoder_args.tc_rec_threshold))

            outputs = {
                'Yr_mu': self.prepare_parameter_dict(Yr_mu),
                'Yr_mu_clean': self.prepare_parameter_dict(Yr_mu_clean),
                'Yr_logvar': self.prepare_parameter_dict(Yr_logvar),
            }

            # If training the full auto encoder variant, then add the latents loss
            if self.dataset_args.train_combined:
                assert Z_pred is not None, 'Z_pred must be provided to the transcoder when training the full auto encoder variant.'
                metrics['loss_com/tc'] = loss.item()
                l_z = self._calculate_com_loss(Z_mu_logits, Z_target=Z_pred)
                metrics['loss_com/Y'] = l_z.item()
                loss = loss + self.optimiser_args.w_com_Y * l_z

        # Masked invertible transcoder
        else:
            loss_p, metrics = self._calculate_slave_transcoder_losses('predictor')
            loss_g, metrics_g = self._calculate_slave_transcoder_losses('generator')
            metrics.update(metrics_g)
            loss = loss_p + loss_g
            outputs = {}

        return outputs, loss, metrics

    def _transcode_loop(
            self,
            Y_target: Union[Tensor, Dict[str, Tensor]],
            add_latent_noise: bool = True,
    ):
        """
        Encode and decode parameters through the transcoder.
        """
        if isinstance(Y_target, Tensor):
            Y_target_vec = Y_target.to(self.device)
        else:
            Y_target_vec = self._Y_to_vec(Y_target)

        # Encode parameters into the latent space
        Z_mu_logits, Z_logvar = self.transcoder.to_latents(Y_target_vec, return_logvar=True, activate=False)

        # Reparameterise the latent vector or add noise
        if add_latent_noise:
            if self.transcoder_args.tc_ae_variant == 'variational':
                Z = self.transcoder.reparameterise(Z_mu_logits, Z_logvar)
            elif self.transcoder_args.tc_ae_variant == 'denoising':
                if self.tc_rec_loss_ema.val is None:
                    noise_level = 0
                else:
                    noise_level = np.exp(-(self.tc_rec_loss_ema.val / self.transcoder_args.tc_rec_threshold))
                Z = Z_mu_logits + torch.randn_like(Z_mu_logits) * noise_level
            else:
                raise NotImplementedError()
        else:
            Z = Z_mu_logits

        # Decode back to the parameter space
        Yr_mu, Yr_logvar = self.transcoder.to_parameters(Z, return_logvar=True)

        return Yr_mu, Yr_logvar, Z_mu_logits, Z_logvar, Z

    def predict(self, X: Tensor, latent_input: bool = False) -> Dict[str, Tensor]:
        """
        Take a batch of images and predict parameters.
        """
        X = X.to(self.device)

        # Generate the parameters from the image
        if not latent_input:
            # Add coordinate grids to the images if required
            if self.dataset_args.add_coord_grid:
                grid = torch.stack(torch.meshgrid(
                    torch.linspace(-1, 1, X.shape[-2]),
                    torch.linspace(-1, 1, X.shape[-1]),
                )).to(self.device)[None, ...].expand(len(X), -1, -1, -1)
                X = torch.cat([X, grid], dim=1)

            # Predict the parameters (or latent vector if using transcoder)
            Y_pred = self.predictor(X)

        # Generate the parameters from the latent vector
        else:
            # If using transcoder, don't need to use the predictor network
            if self.transcoder_args.use_transcoder:
                Y_pred = X.to(self.device)

            # Otherwise, predict the parameters from the latent vector
            else:
                X = X.to(self.device)
                Y_pred = self.predictor.forward_from_latent(X)

        # Transcode
        if self.transcoder_args.use_transcoder:
            Y_pred = self.transcoder(Y_pred, 'parameters')

        # Split up the output into the label groups
        output = self.prepare_parameter_dict(Y_pred)

        return output

    def prepare_parameter_dict(
            self,
            Y: Tensor,
            apply_sigmoid_to_switches: bool = True
    ) -> Dict[str, Tensor]:
        """
        Prepare the predicted parameters into output groups.
        """
        output = {}
        idx = 0

        if self.dataset_args.train_zingg:
            Y_zingg = Y[:, :2]

            # Clip predictions to [0, 1]
            Y_zingg = Y_zingg.clamp(min=0, max=1)

            output['zingg'] = Y_zingg
            idx += 2

        if self.dataset_args.train_distances:
            n_params = len(self.ds.labels_distances_active)
            Y_dists = Y[:, idx:idx + n_params]
            idx += n_params

            if self.dataset_args.use_distance_switches:
                Y_logits = Y[:, idx:idx + n_params]
                if apply_sigmoid_to_switches:
                    Y_switches = torch.sigmoid(Y_logits)
                else:
                    Y_switches = Y_logits
                idx += n_params
                output['distance_switches'] = Y_switches

                # Apply the switches
                ignore_distances = Y_switches < .5
                Y_dists = torch.where(ignore_distances, torch.ones_like(Y_dists) * 100, Y_dists)
            elif self.ds.labels_distances == self.ds.labels_distances_active:
                # Normalise the predictions by the maximum value per batch item
                Ypd_max = Y_dists.amax(dim=1, keepdim=True).abs()
                Y_dists = torch.where(Ypd_max > 0, Y_dists / Ypd_max, Y_dists)

            # Don't allow negative distances
            Y_dists = Y_dists.clamp(min=0)

            output['distances'] = Y_dists

        if self.dataset_args.train_transformation:
            n_params = len(self.ds.labels_transformation)
            if self.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
                n_params += len(self.ds.labels_rotation_quaternion)
            else:
                assert self.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE
                n_params += len(self.ds.labels_rotation_axisangle)
            output['transformation'] = Y[:, idx:idx + n_params]
            idx += n_params

        if self.dataset_args.train_material and len(self.ds.labels_material_active) > 0:
            n_params = len(self.ds.labels_material_active)
            output['material'] = Y[:, idx:idx + n_params]
            idx += n_params

        if self.dataset_args.train_light:
            output['light'] = Y[:, idx:]

        return output

    @staticmethod
    def _Y_to_vec(Y: Dict[str, Tensor]) -> Tensor:
        """
        Convert a dictionary of parameters to a vector, only extracting the actual parameters
        """
        return torch.cat([
            Yk for k, Yk in Y.items()
            if k in ['distances', 'transformation', 'material', 'light']
        ], dim=1)

    def generate(self, Y: Dict[str, Tensor]) -> Optional[Tensor]:
        """
        Take a batch of parameters and generate a batch of images.
        """
        if not self.dataset_args.train_generator:
            return None
        Y_vec = self._Y_to_vec(Y)
        if self.generator_args.gen_include_face_areas:
            Y_vec = torch.cat([Y_vec, Y['face_areas']], dim=1)

        # Add some noise to the input parameters during training
        if self.generator.training and self.generator_args.gen_input_noise_std > 0:
            Y_vec = Y_vec + torch.randn_like(Y_vec) * self.generator_args.gen_input_noise_std

        # Transcode
        if self.transcoder_args.use_transcoder:
            Y_vec = self.transcoder(Y_vec, 'latents')

        X_pred = self.generator(Y_vec)

        return X_pred

    def calculate_predictor_losses(
            self,
            Y_pred: Dict[str, Tensor],
            Y_target: Dict[str, Union[Tensor, List[Tensor]]],
            X_target_clean: Optional[Tensor],
    ) -> Tuple[Tensor, Dict[str, Any], Optional[Tensor]]:
        """
        Calculate losses.
        """
        stats = {}
        losses = []
        X_pred2 = None

        if self.dataset_args.train_zingg:
            loss_z, stats_z = self._calculate_zingg_losses(Y_pred['zingg'], Y_target['zingg'])
            losses.append(self.optimiser_args.w_zingg * loss_z)
            stats.update(stats_z)

        if self.dataset_args.train_distances:
            loss_d, stats_d = self._calculate_distance_losses(Y_pred['distances'], Y_target['distances'])
            losses.append(self.optimiser_args.w_distances * loss_d)
            stats.update(stats_d)

            if self.dataset_args.use_distance_switches:
                loss_s, stats_s = self._calculate_distance_switches_losses(Y_pred['distance_switches'],
                                                                           Y_target['distance_switches'])
                losses.append(self.optimiser_args.w_distances * loss_s)
                stats['losses/distance_switches'] = loss_s.item()

        if self.dataset_args.train_transformation:
            sym_rotations = Y_target['sym_rotations'] if 'sym_rotations' in Y_target else None
            loss_t, stats_t = self._calculate_transformation_losses(Y_pred['transformation'],
                                                                    Y_target['transformation'],
                                                                    sym_rotations)
            losses.append(self.optimiser_args.w_transformation * loss_t)
            stats.update(stats_t)

        if self.dataset_args.train_3d:
            loss_3d, stats_3d = self._calculate_3d_losses(Y_pred, Y_target)
            losses.append(self.optimiser_args.w_3d * loss_3d)
            stats.update(stats_3d)

        if self.dataset_args.train_material and 'material' in Y_pred:
            loss_m, stats_m = self._calculate_material_losses(Y_pred['material'], Y_target['material'])
            losses.append(self.optimiser_args.w_material * loss_m)
            stats.update(stats_m)

        if self.dataset_args.train_light:
            loss_l, stats_l = self._calculate_light_losses(Y_pred['light'], Y_target['light'])
            losses.append(self.optimiser_args.w_light * loss_l)
            stats.update(stats_l)

        # Only include teacher loss if the generator loss is below a threshold
        if self.dataset_args.train_generator:
            loss_t, stats_t, X_pred2 = self._calculate_pred_teacher_losses(Y_pred, X_target_clean)
            if self._can_teach('generator'):
                losses.append(self.optimiser_args.w_net_teacher * loss_t)
            stats.update(stats_t)

        if self.transcoder_args.use_transcoder and self.transcoder_args.tc_trained_by in ['predictor', 'both']:
            loss_t, stats_t = self._calculate_slave_transcoder_losses('predictor')
            losses.append(self.optimiser_args.w_transcoder_pred * loss_t)
            stats.update(stats_t)

        # Sum the losses
        loss = torch.stack(losses).sum()
        assert not is_bad(loss), 'Bad loss!'

        return loss, stats, X_pred2

    def _calculate_zingg_losses(
            self,
            zingg_pred: Tensor,
            zingg_target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the Zingg losses.
        """
        loss = ((zingg_pred - zingg_target)**2).mean()
        stats = {
            'losses/zingg': loss.item()
        }

        return loss, stats

    def _calculate_distance_losses(
            self,
            d_pred: Tensor,
            d_target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate distance losses.
        """
        if self.ds.dataset_args.asymmetry is not None:
            loss = torch.tensor(0., device=self.device)

            # Group asymmetric distances by face group
            for i, hkl in enumerate(self.ds.dataset_args.miller_indices):
                # Collect the distances for this face group
                group_idxs = (self.crystal.symmetry_idx == i).nonzero().squeeze()
                d_pred_i = d_pred[:, group_idxs]
                d_target_i = d_target[:, group_idxs]

                # Sort them
                d_pred_i, _ = d_pred_i.sort(dim=1)
                d_target_i, _ = d_target_i.sort(dim=1)

                # Calculate losses between the sorted distances
                loss = loss + ((d_pred_i - d_target_i)**2).mean()
        else:
            loss = ((d_pred - d_target)**2).mean()

        stats = {
            'losses/distances': loss.item()
        }

        return loss, stats

    def _calculate_distance_switches_losses(
            self,
            s_pred: Tensor,
            s_target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate distance switches losses.
        """
        loss = F.binary_cross_entropy(s_pred, s_target, reduction='mean')
        stats = {
            'distances/switches': loss.item()
        }

        return loss, stats

    def _calculate_transformation_losses(
            self,
            t_pred: Tensor,
            t_target: Tensor,
            sym_rotations: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the transformation losses.
        """
        location_loss = ((t_pred[:, :3] - t_target[:, :3])**2).mean()
        scale_loss = ((t_pred[:, 3] - t_target[:, 3])**2).mean()

        if self.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
            q_pred = t_pred[:, 4:]
            v_norms = q_pred.norm(dim=-1, keepdim=True)
            q_pred = q_pred / v_norms

            # Use the sym_rotations, could be different number for each batch item so have to loop over
            if sym_rotations is not None:
                bs = len(t_pred)
                rotation_losses = torch.zeros(bs, device=self.device)
                self.sym_group_idxs = []
                for i in range(bs):
                    q_pred_i = q_pred[i][None, ...]
                    sym_rotations_i = sym_rotations[i]
                    dot_product = torch.sum(sym_rotations_i * q_pred_i, dim=-1)
                    dot_product = torch.clamp(dot_product, -0.99, 0.99)  # Ensure valid input for arccos
                    angular_differences = 2 * torch.acos(dot_product)

                    # Take the smallest angular difference to any of the symmetric rotations
                    min_idx = int(torch.argmin(angular_differences))
                    self.sym_group_idxs.append(min_idx)
                    rotation_losses[i] = angular_differences[min_idx]
                rotation_loss = rotation_losses.mean()
            else:
                rotation_loss = ((q_pred - t_target[:, 4:])**2).mean()

        else:
            assert self.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE
            R_pred = axis_angle_to_rotation_matrix(t_pred[:, 4:])

            # Use the sym_rotations, could be different number for each batch item so have to loop over
            if sym_rotations is not None:
                bs = len(t_pred)
                rotation_losses = torch.zeros(bs, device=self.device)
                self.sym_group_idxs = []
                for i in range(bs):
                    sym_rotations_i = sym_rotations[i]
                    R_pred_i = R_pred[i][None, ...].expand(len(sym_rotations[i]), -1, -1)
                    angular_differences = geodesic_distance(R_pred_i, sym_rotations_i)

                    # Take the smallest angular difference to any of the symmetric rotations
                    min_idx = int(torch.argmin(angular_differences))
                    self.sym_group_idxs.append(min_idx)
                    rotation_losses[i] = angular_differences[min_idx]
                rotation_loss = rotation_losses.mean()
            else:
                R_target = axis_angle_to_rotation_matrix(t_target[:, 4:])
                rotation_loss = geodesic_distance(R_pred, R_target).mean()

        loss = location_loss + scale_loss + rotation_loss
        if self.dataset_args.rotation_mode != ROTATION_MODE_AXISANGLE:
            preangles_loss = ((v_norms - 1)**2).mean()
            loss = loss + preangles_loss

        stats = {
            'transformation/l_location': location_loss.item(),
            'transformation/l_scale': scale_loss.item(),
            'transformation/l_rotation': rotation_loss.item(),
            'losses/transformation': loss.item()
        }

        if self.dataset_args.rotation_mode != ROTATION_MODE_AXISANGLE:
            stats['transformation/l_preangles'] = preangles_loss.item()

        return loss, stats

    def _calculate_3d_losses(
            self,
            Y_pred: Dict[str, Tensor],
            Y_target: Dict[str, Union[Tensor, List[Tensor]]]
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the distances between 3d vertices.
        """
        distances = self.ds.prep_distances(
            distance_vals=Y_pred['distances'],
            switches=Y_pred['distance_switches'] if 'distance_switches' in Y_pred else None
        )

        # Calculate the polyhedral vertices for the parameters
        v_pred, v_pred_og, nv_pred, farthest_dists = calculate_polyhedral_vertices(
            distances=distances,
            origin=Y_pred['transformation'][:, :3],
            scale=Y_pred['transformation'][:, 3],
            rotation=Y_pred['transformation'][:, 4:],
            symmetry_idx=self.crystal.symmetry_idx if self.ds.dataset_args.asymmetry is None else None,
            plane_normals=self.crystal.N,
        )

        # Pad the vertices so that all batch entries have the same number of vertices
        nv_target = torch.tensor([len(v) for v in Y_target['vertices']], device=self.device)
        max_vertices_target = nv_target.amax()
        v_target = torch.stack([
            torch.cat([v, torch.zeros(max_vertices_target - len(v), 3, device=self.device)])
            for v in Y_target['vertices']
        ])

        # Calculate pairwise distances
        dists = torch.cdist(v_pred, v_target)

        # Create masks for valid vertices
        mask_pred = ((torch.arange(v_pred.shape[1], device=v_pred.device)
                      .expand(v_pred.shape[0], -1) < nv_pred.unsqueeze(1))
                     .unsqueeze(2).expand_as(dists))
        mask_target = ((torch.arange(v_target.shape[1], device=v_target.device)
                        .expand(v_target.shape[0], -1) < nv_target.unsqueeze(1))
                       .unsqueeze(1).expand_as(dists))

        # Set large distances for dummy vertices
        dists = torch.where(
            mask_pred & mask_target,
            dists,
            torch.ones_like(dists) * 1e8
        )

        # Get the minimum distances between the predicted and target vertices (in both directions)
        min_dists1 = dists.amin(dim=1)
        min_dists2 = dists.amin(dim=2)
        d = torch.cat([min_dists1, min_dists2], dim=1)

        # Ensure any dummy large distances are ignored
        invalid_idxs = d > 1e7
        d[invalid_idxs] = 0

        # Calculate the average minimum distance per vertex
        l_vertices = (d.sum(dim=1) / (d > 0).sum(dim=1).clip(1)).mean()

        # Regularise the distances so all planes are touching the polyhedron
        N_dot_v = torch.einsum('pi,bvi->bpv', self.crystal.N, v_pred_og)
        distances_min = N_dot_v.amax(dim=-1)[:, :distances.shape[1]]
        overshoot = distances - distances_min
        l_overshoot = torch.where(
            overshoot > 0,
            overshoot**2,
            torch.zeros_like(overshoot)
        ).mean()

        # Check for undershoot (planes that should be missing but are present)
        eps = 1e-4
        missing_faces: torch.BoolTensor = Y_target['face_areas'] < eps
        undershoot = distances - farthest_dists[:, :distances.shape[1]]
        l_undershoot = torch.where(
            missing_faces,
            undershoot**2,
            torch.zeros_like(undershoot)
        ).mean()

        loss = l_vertices \
               + self.optimiser_args.w_3d_overshoot * l_overshoot \
               + self.optimiser_args.w_3d_undershoot * l_undershoot

        stats = {
            '3d/l_vertices': l_vertices.item(),
            '3d/l_overshoot': l_overshoot.item(),
            '3d/l_undershoot': l_undershoot.item(),
            'losses/3d': loss.item()
        }

        return loss, stats

    def _calculate_material_losses(
            self,
            m_pred: Tensor,
            m_target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the material properties losses.
        """
        losses = []
        stats = {}
        m_idx = 0
        if 'b' in self.ds.labels_material_active:
            b_loss = ((m_pred[:, m_idx] - m_target[:, m_idx])**2).mean()
            m_idx += 1
            stats['material/l_brightness'] = b_loss.item()
            losses.append(b_loss)
        if 'ior' in self.ds.labels_material_active:
            ior_loss = ((m_pred[:, m_idx] - m_target[:, m_idx])**2).mean()
            m_idx += 1
            stats['material/l_ior'] = ior_loss.item()
            losses.append(ior_loss)
        if 'r' in self.ds.labels_material_active:
            r_loss = ((m_pred[:, m_idx] - m_target[:, m_idx])**2).mean()
            m_idx += 1
            stats['material/l_roughness'] = r_loss.item()
            losses.append(r_loss)

        loss = torch.stack(losses).sum()
        stats['losses/material'] = loss.item()

        return loss, stats

    def _calculate_light_losses(
            self,
            l_pred: Tensor,
            l_target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the light properties losses.
        """
        assert l_pred.shape[-1] == len(self.ds.labels_light_active)
        radiance_loss = ((l_pred - l_target)**2).mean()
        loss = radiance_loss
        stats = {
            'light/l_radiance': radiance_loss.item(),
            'losses/light': loss.item()
        }
        return loss, stats

    def _calculate_pred_teacher_losses(
            self,
            Y_pred: Dict[str, Tensor],
            X_target: Tensor,
    ) -> Tuple[Tensor, Dict[str, float], Tensor]:
        """
        Calculate the predictor's teacher losses using the generator network as the teacher.
        """
        # Run the predicted parameters through the generator to get another image
        if self.transcoder_args.use_transcoder:
            Z = self.transcoder.latents_in
        else:
            Z = self._Y_to_vec(Y_pred)
        X_pred2 = self.generator(Z)

        # Calculate how closely the same images come back out
        loss, _ = self._calculate_generator_losses(X_pred2, X_target)
        stats = {
            'losses/teacher_net': loss.item()
        }

        return loss, stats, X_pred2

    def calculate_generator_losses(
            self,
            X_pred: Optional[Tensor],
            X_target: Optional[Tensor],
            Y_target: Dict[str, Union[Tensor, List[Tensor]]],
            include_teacher_loss: bool = True,
            include_discriminator_loss: bool = True,
            include_rcf_loss: bool = True,
            include_transcoder_loss: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Dict[str, Tensor]]]:
        """
        Calculate losses.
        """
        stats = {}
        losses = []
        Y_pred2 = None

        if not self.dataset_args.train_generator:
            return torch.tensor(0., device=self.device), stats, Y_pred2

        # Main generator loss
        loss_g, stats_g = self._calculate_generator_losses(X_pred, X_target)
        losses.append(self.optimiser_args.w_generator * loss_g)
        stats.update(stats_g)

        # Teacher loss - only include if the prediction loss is below a threshold
        if include_teacher_loss and self.dataset_args.train_predictor:
            loss_t, stats_t, Y_pred2 = self._calculate_gen_teacher_losses(X_pred, Y_target)
            if self._can_teach('predictor'):
                losses.append(self.optimiser_args.w_gen_teacher * loss_t)
            stats.update(stats_t)

        # Discriminator loss
        if include_discriminator_loss and self.generator_args.use_discriminator:
            loss_d, stats_d = self._calculate_gen_gan_losses(X_pred)
            losses.append(self.optimiser_args.w_discriminator * loss_d)
            stats.update(stats_d)

        # Edge features loss
        if include_rcf_loss and self.generator_args.use_rcf:
            loss_rcf, stats_rcf = self._calculate_rcf_losses(X_pred, X_target)
            losses.append(self.optimiser_args.w_rcf * loss_rcf)
            stats.update(stats_rcf)

        # Transcoder regularisation loss
        if (include_transcoder_loss and self.transcoder_args.use_transcoder
                and self.transcoder_args.tc_trained_by in ['generator', 'both']):
            loss_t, stats_t = self._calculate_slave_transcoder_losses('generator')
            losses.append(self.optimiser_args.w_transcoder_gen * loss_t)
            stats.update(stats_t)

        # Sum the losses
        loss = torch.stack(losses).sum()
        assert not is_bad(loss), 'Bad loss!'

        return loss, stats, Y_pred2

    def _calculate_generator_losses(
            self,
            X_pred: Tensor,
            X_target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the generator network losses.
        """
        if self.generator_args.gen_image_loss == 'l1':
            loss = (X_pred - X_target).abs().mean()
        elif self.generator_args.gen_image_loss == 'l2':
            loss = ((X_pred - X_target)**2).mean()
        else:
            raise NotImplementedError()
        stats = {
            'losses/generator': loss.item()
        }

        return loss, stats

    def _calculate_gen_teacher_losses(
            self,
            X_pred: Tensor,
            Y_target: Dict[str, Union[Tensor, List[Tensor]]],
    ) -> Tuple[Tensor, Dict[str, float], Dict[str, Tensor]]:
        """
        Calculate the teacher losses using the parameter network as the teacher.
        """
        # Run the predicted image through the parameter network to estimate the parameters (or latent vector logits)
        Y_pred2 = self.predict(X_pred)
        stats = {}

        # Calculate loss in the latent space if using a transcoder
        if self.transcoder_args.use_transcoder:
            Z1 = self.transcoder.latents_out  # Latent vector from parameters via transcoder
            Z2 = self.transcoder.latents_in  # Latent vector from image via predictor
            loss = ((Z1 - Z2)**2).mean()
        else:
            # Calculate how closely the same parameters come back out
            loss, r_stats = self._calculate_parameter_reconstruction_losses(
                Y_target, Y_pred2, check_sym_rotations=True
            )
            for k, v in r_stats.items():
                stats[f'losses/teacher_gen/{k}'] = v

        stats['losses/teacher_gen'] = loss.item()

        return loss, stats, Y_pred2

    def _calculate_gen_gan_losses(
            self,
            X_pred: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the GAN losses on the generator.
        """
        D_fake = self.discriminator(X_pred)
        loss = self.discriminator.gen_loss(D_fake)
        stats = {
            'losses/gan_gen': loss.item()
        }

        return loss, stats

    def _calculate_rcf_losses(
            self,
            X_pred: Tensor,
            X_target: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Calculate the RCF losses on the generator.
        """
        bs = len(X_pred)
        rcf_input = torch.cat([X_pred, X_target])
        rcf_feats = self.rcf(rcf_input, apply_sigmoid=False)

        def loss_(X_pred_, X_target_):
            if self.generator_args.rcf_loss_type == 'l1':
                l = (X_pred_ - X_target_).abs().mean()
            elif self.generator_args.rcf_loss_type == 'l2':
                l = ((X_pred_ - X_target_)**2).mean()
            else:
                raise NotImplementedError()
            return l

        losses = [loss_(f[:bs], f[bs:]) for f in rcf_feats]
        loss = torch.stack(losses).sum()

        stats = {
            **{f'rcf/feat_{i}': l.item() for i, l in enumerate(losses)},
            **{'losses/rcf': loss.item()}
        }

        return loss, stats

    def calculate_discriminator_losses(
            self,
            D_real: Tensor,
            D_fake: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Calculate discriminator losses.
        """
        stats = {}
        losses = []

        if not self.dataset_args.train_generator or not self.generator_args.use_discriminator:
            return torch.tensor(0., device=self.device), stats

        # Discriminator loss
        loss_d, stats_d = self._calculate_disc_gan_losses(D_real, D_fake)
        losses.append(loss_d)
        stats.update(stats_d)

        # Sum the losses
        loss = torch.stack(losses).sum()
        assert not is_bad(loss), 'Bad loss!'

        return loss, stats

    def _calculate_disc_gan_losses(
            self,
            D_real: Tensor,
            D_fake: Tensor,
    ):
        """
        Calculate the GAN losses on the discriminator.
        """
        loss = self.discriminator.discr_loss(D_fake, D_real)
        stats = {
            'losses/gan_disc': loss.item()
        }

        return loss, stats

    def _calculate_slave_transcoder_losses(self, which: str):
        """
        Calculate the transcoder losses.
        """
        if not self.transcoder_args.use_transcoder:
            return torch.tensor(0., device=self.device), {}

        if which == 'predictor':
            l_latents = ((self.transcoder.latents_in.norm(dim=-1, p=2) - 1)**2).mean()
        elif which == 'generator':
            l_latents = ((self.transcoder.latents_out.norm(dim=-1, p=2) - 1)**2).mean()
        else:
            raise RuntimeError(f'Unknown transcoder loss context: {which}')

        # Orthogonality loss
        W = self.transcoder.weight
        I = torch.eye(W.shape[1], device=W.device)
        l_orthogonality = torch.norm(W.T @ W - I, p='fro')

        loss = l_latents + l_orthogonality

        stats = {
            f'transcoder/latents_{which}': l_latents.item(),
            f'transcoder/orth_{which}': l_orthogonality.item(),
            'losses/transcoder': loss.item()
        }

        return loss, stats

    def calculate_vae_transcoder_losses(
            self,
            Y_target: Dict[str, Union[Tensor, List[Tensor]]],
            Yr_mu_vec: Tensor,
            Yr_mu_clean: Tensor,
            Z_mu_logits: Tensor,
            Z_logvar: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Calculate VAE transcoder losses.
        """
        if not self.transcoder_args.use_transcoder or self.transcoder_args.tc_model_name not in ['vae', 'tvae']:
            return torch.tensor(0., device=self.device), {}

        # Reconstruction losses - noisy version
        l_reconst, stats_reconst = self._calculate_parameter_reconstruction_losses(
            Y_target, Yr_mu_vec, check_sym_rotations=True
        )
        self.tc_rec_loss_ema(l_reconst.item())

        # Reconstruction losses - clean version (just for debug)
        l_reconst_clean, _ = self._calculate_parameter_reconstruction_losses(Y_target, Yr_mu_clean)

        # Parameter independence loss
        l_indep, stats_indep = self._calculate_transcoder_parameter_independence_losses(Y_target)

        # L1 and L2 regularisation terms
        l_l1 = torch.norm(Z_mu_logits, p=1, dim=-1).mean()
        l_l2 = torch.norm(Z_mu_logits, p=2, dim=-1).mean()

        # KL divergence regularisation term
        l_kl = -0.5 * torch.sum(1 + Z_logvar - Z_mu_logits.pow(2) - Z_logvar.exp(), dim=-1).mean()
        self.tc_kl_loss_ema(l_kl.item())

        # Blend in the KL regularisation loss otherwise it dominates
        w_kl = max(
            self.transcoder_args.tc_min_w_kl,
            self.transcoder_args.tc_max_w_kl * min(
                1, max(0, 1 - self.tc_rec_loss_ema.val / self.transcoder_args.tc_rec_threshold)
            )**2
        )

        # Sum the losses
        loss = l_reconst \
               + self.transcoder_args.tc_w_l1 * l_l1 \
               + self.transcoder_args.tc_w_l2 * l_l2 \
               + self.transcoder_args.tc_w_indep * l_indep
        if self.transcoder_args.tc_ae_variant == 'variational':
            loss = loss + w_kl * l_kl
        assert not is_bad(loss), 'Bad loss!'

        stats = {
            'losses/tc/reconst': l_reconst.item(),
            'losses/tc/reconst_clean': l_reconst_clean.item(),
            'losses/tc/z_l1': l_l1.item(),
            'losses/tc/z_l2': l_l2.item(),
            'losses/tc/kl': l_kl.item(),
            'losses/transcoder': loss.item(),
            'tc/w_kl': w_kl,
            **{f'tc/{k}': v for k, v in stats_reconst.items()},
            **stats_indep,
        }

        return loss, stats

    def _calculate_parameter_reconstruction_losses(
            self,
            Y_target: Dict[str, Union[Tensor, List[Tensor]]],
            Y_pred: Union[dict, Tensor],
            check_sym_rotations: bool = False
    ):
        """
        Calculate the parameter errors.
        """
        if isinstance(Y_pred, Tensor):
            Y_pred = self.prepare_parameter_dict(Y_pred)
        losses = []
        stats = {}

        if self.dataset_args.train_zingg:
            loss_z, stats_z = self._calculate_zingg_losses(Y_pred['zingg'], Y_target['zingg'])
            losses.append(loss_z)
            stats.update(stats_z)

        if self.dataset_args.train_distances:
            loss_d, stats_d = self._calculate_distance_losses(Y_pred['distances'], Y_target['distances'])
            losses.append(loss_d)
            stats.update(stats_d)

            if self.dataset_args.use_distance_switches:
                loss_s, stats_s = self._calculate_distance_switches_losses(Y_pred['distance_switches'],
                                                                           Y_target['distance_switches'])
                losses.append(loss_s)
                stats['losses/distance_switches'] = loss_s.item()

        if self.dataset_args.train_transformation:
            sym_rotations = Y_target['sym_rotations'] if check_sym_rotations and 'sym_rotations' in Y_target else None
            loss_t, stats_t = self._calculate_transformation_losses(Y_pred['transformation'],
                                                                    Y_target['transformation'],
                                                                    sym_rotations)
            losses.append(loss_t)
            stats.update(stats_t)

        if self.dataset_args.train_material and 'material' in Y_pred:
            loss_m, stats_m = self._calculate_material_losses(Y_pred['material'], Y_target['material'])
            losses.append(loss_m)
            stats.update(stats_m)

        if self.dataset_args.train_light:
            loss_l, stats_l = self._calculate_light_losses(Y_pred['light'], Y_target['light'])
            losses.append(loss_l)
            stats.update(stats_l)

        # Combine the reconstruction losses
        l_reconst = torch.stack(losses).sum()

        return l_reconst, stats

    def _calculate_transcoder_parameter_independence_losses(
            self,
            Y_target: Dict[str, Union[Tensor, List[Tensor]]],
    ):
        """
        Calculate the parameter independence loss.
        """
        Y_vec = self._Y_to_vec(Y_target)
        noise_level = 0.1
        bs = len(Y_vec)

        # The parameters should be independent
        groups = {}
        Y_perturbations = []
        idx = 0

        def add_to_batch(start_idx, end_idx):
            noise = torch.randn_like(Y_vec) * noise_level
            noise[:, start_idx:end_idx] = 0
            Y_perturbations.append(Y_vec.clone() + noise)

        # Distances define the morphology
        if self.dataset_args.train_distances:
            n_params = len(self.ds.labels_distances_active)
            if self.dataset_args.use_distance_switches:
                n_params *= 2
            if self.dataset_args.train_zingg:
                n_params += 2
            groups['distances'] = (idx, idx + n_params)
            idx += n_params
        elif self.dataset_args.train_zingg:
            idx += 2

        # Transformation has three independent components - location, scale and rotation
        if self.dataset_args.train_transformation:
            # groups['t/location'] = (idx, idx + 3)
            # idx += 3
            groups['t/origin_x'] = (idx, idx + 1)
            idx += 1
            groups['t/origin_y'] = (idx, idx + 1)
            idx += 1
            groups['t/origin_z'] = (idx, idx + 1)
            idx += 1
            # Leave out scale as it is sort of dependent on the morphology
            # groups['t/scale'] = (idx, idx + 1)
            idx += 1
            if self.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
                groups['t/rotation'] = (idx, idx + 4)
                idx += 4
            else:
                assert self.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE
                groups['t/rotation'] = (idx, idx + 3)
                idx += 3

        # All material properties are independent
        if self.dataset_args.train_material:
            for i, k in enumerate(self.ds.labels_material_active):
                groups[f'm/{k}'] = (idx, idx + 1)
                idx += 1

        # Lighting has one three-component radiance property
        if self.dataset_args.train_light:
            groups['l/radiance'] = (idx, idx + 3)
            idx += 1

        # Process the batches of variations together
        for group_idxs in groups.values():
            add_to_batch(*group_idxs)
        Y_perturbations = torch.cat(Y_perturbations, dim=0)
        Yr, _, _, _, _ = self._transcode_loop(Y_perturbations, add_latent_noise=False)
        Yr_dict = self.prepare_parameter_dict(Yr)

        # For each independent group, calculate how far the parameters varied when changing the others
        losses = []
        stats = {}

        def calculate_independence_loss(key, batch_idx, start_idx, end_idx):
            Yp_ = Y_perturbations[batch_idx * bs:(batch_idx + 1) * bs, start_idx:end_idx]
            Yr_ = Yr[batch_idx * bs:(batch_idx + 1) * bs, start_idx:end_idx]
            if key == 'distances':
                if self.dataset_args.train_zingg:
                    l_z = torch.mean((Yp_[:, :2] - Yr_[:, :2])**2)
                    Yp_ = Yp_[:, 2:]
                else:
                    l_z = 0.
                n = len(self.ds.labels_distances_active)
                dists_r = Yr_dict['distances'][batch_idx * bs:(batch_idx + 1) * bs]
                l_d = torch.mean((Yp_[:, :n] - dists_r)**2)
                if self.dataset_args.use_distance_switches:
                    switches_r = Yr_dict['distance_switches'][batch_idx * bs:(batch_idx + 1) * bs]
                    l_s = F.binary_cross_entropy(switches_r, Yp_[:, n:], reduction='mean')
                else:
                    l_s = 0
                l = l_z + l_d + l_s
            else:
                l = torch.mean((Yp_ - Yr_)**2)
            losses.append(l)
            stats[f'tc/indep/{key}'] = l.item()

        for i, (k, group_idxs) in enumerate(groups.items()):
            calculate_independence_loss(k, i, *group_idxs)

        # Combine the losses
        l_indep = torch.stack(losses).sum()
        stats['losses/tc/indep'] = l_indep.item()

        return l_indep, stats

    def _calculate_com_loss(
            self,
            Z: Tensor,
            Z_target: Optional[Tensor] = None,
            Y_target: Optional[Dict[str, Union[Tensor, List[Tensor]]]] = None,
            use_sym_rotations: bool = True
    ):
        """
        Calculate the latents loss - difference between the latent Z and the target latents (from the parameters).
        """
        if Y_target is not None:
            assert Z_target is None
            Y_target_vec = self._Y_to_vec(Y_target)

            # Make a new batch of Y_targets with the rotation picked to be as close to the predicted rotation as possible
            if use_sym_rotations and 'sym_rotations' in Y_target:
                bs = len(Y_target_vec)
                q0 = self.ds.labels.index('rw')
                for i in range(bs):
                    Y_target_vec[i, q0:q0 + 4] = Y_target['sym_rotations'][i][self.sym_group_idxs[i]]

            # Measure distance to the target latents
            with torch.no_grad():
                Z_target = self.transcoder.to_latents(Y_target_vec)
        l_z = torch.mean((Z - Z_target)**2)

        return l_z

    def _can_teach(self, which: str) -> bool:
        """
        Check if the network is good enough to be teaching.
        """
        p = self.p_loss_ema.val is not None and self.p_loss_ema.val < self.optimiser_args.teach_threshold_pred
        g = self.g_loss_ema.val is not None and self.g_loss_ema.val < self.optimiser_args.teach_threshold_gen
        if self.optimiser_args.teach_threshold_combined:
            return p & g
        if which == 'predictor':
            return p
        if which == 'generator':
            return g
        raise RuntimeError(f'Unknown teacher network: {which}')

    def _make_plots(
            self,
            data: Tuple[dict, Tensor, Tensor, Tensor, Dict[str, Tensor]],
            outputs: Dict[str, Any],
            stats: Dict[str, Tensor],
            train_or_test: str,
            end_of_epoch: bool = False
    ):
        """
        Generate some example plots.
        """
        if self.runtime_args.plot_n_examples > 0 and (
                end_of_epoch or
                (self.runtime_args.plot_every_n_batches > -1
                 and (self.checkpoint.step + 1) % self.runtime_args.plot_every_n_batches == 0)
        ):
            logger.info('Plotting.')
            n_examples = min(self.runtime_args.plot_n_examples, self.runtime_args.batch_size)
            idxs = np.random.choice(self.runtime_args.batch_size, n_examples, replace=False)
            if self.dataset_args.train_predictor:
                fig = plot_training_samples(self, data, outputs, train_or_test, idxs)
                self._save_plot(fig, 'samples', train_or_test)
            elif self.dataset_args.train_generator:
                fig = plot_generator_samples(self, data, outputs, train_or_test, idxs)
                self._save_plot(fig, 'samples', train_or_test)
            if self.dataset_args.train_denoiser:
                fig = plot_denoiser_samples(self, data, outputs, train_or_test, idxs)
                self._save_plot(fig, 'denoiser', train_or_test)
            # if self.transcoder_args.use_transcoder: todo - fix
            #     fig = plot_vaetc_examples(data, outputs, train_or_test, idxs)
            #     self._save_plot(fig, 'vaetc_examples', train_or_test)

    def _save_plot(self, fig: Figure, plot_type: str, train_or_test: str = None):
        """
        Log the figure to the tensorboard logger and optionally save it to disk.
        """
        suffix = f'_{train_or_test}' if train_or_test is not None else ''

        # Save to disk
        if self.runtime_args.save_plots:
            save_dir = self.logs_path / 'plots' / f'{plot_type}{suffix}'
            save_dir.mkdir(exist_ok=True)
            path = save_dir / f'{self.checkpoint.step + 1:08d}.png'
            plt.savefig(path, bbox_inches='tight')

        # Log to tensorboard
        if self.runtime_args.save_plots_to_tb:
            self.tb_logger.add_figure(f'{plot_type}{suffix}', fig, self.checkpoint.step)
            self.tb_logger.flush()

        plt.close(fig)
