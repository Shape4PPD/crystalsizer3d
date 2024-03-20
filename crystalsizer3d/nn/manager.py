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
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.scheduler.scheduler import Scheduler
from torch import nn
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs, PREANGLES_MODE_AXISANGLE, \
    PREANGLES_MODE_QUATERNION, PREANGLES_MODE_SINCOS
from crystalsizer3d.args.generator_args import GeneratorArgs
from crystalsizer3d.args.network_args import NetworkArgs
from crystalsizer3d.args.optimiser_args import OptimiserArgs
from crystalsizer3d.args.runtime_args import RuntimeArgs
from crystalsizer3d.args.transcoder_args import TranscoderArgs
from crystalsizer3d.crystal_renderer import render_from_parameters
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
from crystalsizer3d.nn.models.resnet import ResNet
from crystalsizer3d.nn.models.timmnet import TimmNet
from crystalsizer3d.nn.models.transcoder import Transcoder
from crystalsizer3d.nn.models.transcoder_mask_inv import TranscoderMaskInv
from crystalsizer3d.nn.models.transcoder_tvae import TranscoderTVAE
from crystalsizer3d.nn.models.transcoder_vae import TranscoderVAE
from crystalsizer3d.nn.models.vit_pretrained import ViTPretrainedNet
from crystalsizer3d.nn.models.vitvae import ViTVAE
from crystalsizer3d.util.ema import EMA
from crystalsizer3d.util.utils import equal_aspect_ratio, geodesic_distance, is_bad, to_numpy

try:
    from ccdc.morphology import VisualHabitMorphology
    from crystalsizer3d.crystal_generator import CrystalGenerator

    CCDC_AVAILABLE = True
except RuntimeError as e:
    if str(e) == 'A valid licence cannot be found':
        CCDC_AVAILABLE = False
    else:
        raise e
except ImportError:
    CCDC_AVAILABLE = False

if CCDC_AVAILABLE:
    logger.info('CCDC is available!')
else:
    logger.warning('CCDC is not available. Crystal generation will not work.')


# torch.autograd.set_detect_anomaly(True)


class Manager:
    def __init__(
            self,
            runtime_args: RuntimeArgs,
            dataset_args: DatasetTrainingArgs,
            net_args: NetworkArgs,
            generator_args: GeneratorArgs,
            transcoder_args: TranscoderArgs,
            optimiser_args: OptimiserArgs,
            save_dir: Optional[Path] = None
    ):
        # Argument groups
        self.runtime_args = runtime_args
        self.dataset_args = dataset_args
        self.net_args = net_args
        self.generator_args = generator_args
        self.transcoder_args = transcoder_args
        self.optimiser_args = optimiser_args

        # Dataset and data loaders
        self.ds = Dataset(self.dataset_args)
        self.train_loader, self.test_loader = self._init_data_loaders()

        # Crystal generator
        self.crystal_generator = self._init_crystal_generator()

        # Networks
        self.predictor = self._init_predictor()
        self.generator = self._init_generator()
        self.discriminator = self._init_discriminator()
        self.transcoder = self._init_transcoder()

        # Optimiser
        (self.optimiser_p, self.lr_scheduler_p,
         self.optimiser_g, self.lr_scheduler_g,
         self.optimiser_d, self.lr_scheduler_d,
         self.optimiser_t, self.lr_scheduler_t) = self._init_optimisers()

        # Metrics
        self.metric_keys = self._init_metrics()

        # Runtime params
        self.device = self._init_devices()

        # Checkpoints
        self.checkpoint = self._init_checkpoint(save_dir=save_dir)

    @classmethod
    def load(
            cls,
            model_path: Path,
            args_changes: Dict[str, Dict[str, Any]],
            save_dir: Optional[Path] = None
    ) -> 'Manager':
        """
        Instantiate a manager from a checkpoint json file.
        """
        logger.info(f'Loading model from {model_path}.')
        with open(model_path, 'r') as f:
            data = json.load(f)

        # Ensure all the required arguments are present
        required_args = ['runtime_args', 'dataset_args', 'network_args',
                         'generator_args', 'transcoder_args', 'optimiser_args']
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
            transcoder_args=TranscoderArgs.from_args(data['transcoder_args']),
            optimiser_args=OptimiserArgs.from_args(data['optimiser_args'])
        )

        # Update the arguments with required changes
        for arg_group, arg_changes in args_changes.items():
            for k, v in arg_changes.items():
                setattr(args[arg_group], k, v)

        return cls(**args, save_dir=save_dir)

    @property
    def image_shape(self) -> Tuple[int, ...]:
        n_channels = 3 if self.dataset_args.add_coord_grid else 1
        return n_channels, self.ds.image_size, self.ds.image_size

    @property
    def parameters_shape(self) -> Tuple[int, ...]:
        return self.ds.label_size,

    @property
    def logs_path(self) -> Path:
        return self.checkpoint.save_dir / self.checkpoint.id

    def _init_crystal_generator(self) -> Optional['CrystalGenerator']:
        """
        Initialise the crystal generator.
        """
        if CCDC_AVAILABLE:
            dsa = self.ds.dataset_args
            generator = CrystalGenerator(
                crystal_id=dsa.crystal_id,
                ratio_means=dsa.ratio_means,
                ratio_stds=dsa.ratio_stds,
                zingg_bbox=dsa.zingg_bbox,
                constraints=dsa.distance_constraints,
            )
        else:
            generator = None
        return generator

    def _init_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get the data loaders.
        """
        logger.info('Initialising data loaders.')
        loaders = {}
        for tt in ['train', 'test']:
            loaders[tt] = get_data_loader(
                ds=self.ds,
                augment=self.dataset_args.augment,
                train_or_test=tt,
                batch_size=self.runtime_args.batch_size,
                n_workers=self.runtime_args.n_dataloader_workers
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
        logger.debug(f'----------- Predictor Network --------------\n\n{predictor}\n\n')

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

        input_shape = self.parameters_shape if not self.transcoder_args.use_transcoder \
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
        logger.debug(f'----------- Generator Network --------------\n\n{generator}\n\n')

        # Instantiate an exponential moving average tracker for the generator loss
        self.g_loss_ema = EMA()

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
        logger.debug(f'----------- Discriminator Network --------------\n\n{discriminator}\n\n')

        # Instantiate an exponential moving average tracker for the discriminator loss
        self.d_loss_ema = EMA()

        return discriminator

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
        logger.debug(f'----------- Transcoder Network --------------\n\n{transcoder}\n\n')

        # Instantiate exponential moving average trackers for the reconstruction and regularisation losses
        self.tc_rec_loss_ema = EMA()
        self.tc_kl_loss_ema = EMA()

        return transcoder

    def _init_optimisers(self) -> Tuple[
        Optimizer, Scheduler,
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
        n_epochs_p = n_epochs_g = n_epochs_d = n_epochs_t = 0

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
        n_epochs = np.array([n_epochs_p, n_epochs_g, n_epochs_d, n_epochs_t])
        if np.all(n_epochs == 0):
            raise RuntimeError('No optimisers were created!')
        assert np.allclose(n_epochs[n_epochs > 0] / ra.n_epochs, 1, atol=0.05)

        return optimiser_p, lr_scheduler_p, optimiser_g, lr_scheduler_g, optimiser_d, lr_scheduler_d, optimiser_t, lr_scheduler_t

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
                if self.predictor is not None:
                    self.predictor = nn.DataParallel(self.predictor)
                if self.generator is not None:
                    self.generator = nn.DataParallel(self.generator)
                if self.discriminator is not None:
                    self.discriminator = nn.DataParallel(self.discriminator)
                if self.transcoder is not None:
                    self.transcoder = nn.DataParallel(self.transcoder)
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
                self.predictor.eval()

            # Load generator network parameters and optimiser state
            if self.dataset_args.train_generator:
                self.generator.load_state_dict(self._fix_state(state['net_g_state_dict']), strict=False)
                if not self.dataset_args.train_combined:
                    self.optimiser_g.load_state_dict(state['optimiser_g_state_dict'])
                self.generator.eval()

            # Load discriminator network parameters and optimiser state
            if self.dataset_args.train_generator and self.generator_args.use_discriminator:
                self.discriminator.load_state_dict(self._fix_state(state['net_d_state_dict']), strict=False)
                self.optimiser_d.load_state_dict(state['optimiser_d_state_dict'])
                self.discriminator.eval()

            # Load transcoder network parameters
            if self.transcoder_args.use_transcoder:
                self.transcoder.load_state_dict(self._fix_state(state['net_t_state_dict']), strict=False)
                if self.optimiser_t is not None and 'optimiser_t_state_dict' in state:
                    self.optimiser_t.load_state_dict(state['optimiser_t_state_dict'])
                self.transcoder.eval()

        elif self.runtime_args.resume_only:
            raise RuntimeError('Could not resume!')

        return checkpoint

    def _fix_state(self, state):
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        return new_state

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
            data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Tuple[dict, float, dict]:
        """
        Train on a single batch of data.
        """
        if self.dataset_args.train_predictor:
            self.predictor.train()
        if self.dataset_args.train_generator:
            self.generator.train()
            if self.generator_args.use_discriminator:
                self.discriminator.train()
        if self.transcoder_args.use_transcoder:
            self.transcoder.train()

        outputs = {}
        stats = {}
        loss_total = 0.

        # Process the batch, calculate gradients and do optimisation step
        if self.dataset_args.train_predictor:
            outputs_p, loss_p, stats_p = self._process_batch_predictor(data)
            self.optimiser_p.zero_grad()
            loss_p.backward()
            if self.optimiser_args.clip_grad_norm != -1:
                nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=self.optimiser_args.clip_grad_norm)
            self.optimiser_p.step()
            outputs.update(outputs_p)
            stats.update(stats_p)
            loss_total += loss_p.item()
            self.tb_logger.add_scalar('batch/train/loss_pred', loss_p, self.checkpoint.step)

        # Do the same for the generator network
        if self.dataset_args.train_generator and not self.dataset_args.train_combined:
            outputs_g, loss_g, stats_g = self._process_batch_generator(data)
            self.optimiser_g.zero_grad()
            loss_g.backward()
            if self.optimiser_args.clip_grad_norm != -1:
                nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.optimiser_args.clip_grad_norm)
            self.optimiser_g.step()
            outputs.update(outputs_g)
            stats.update(stats_g)
            loss_total += loss_g.item()
            self.tb_logger.add_scalar('batch/train/loss_gen', loss_g, self.checkpoint.step)

        # Only train the discriminator when it is doing worse than some threshold
        if self.dataset_args.train_generator and self.generator_args.use_discriminator:
            outputs_d, loss_d, stats_d = self._process_batch_discriminator(data, outputs['X_pred'])
            if self.d_loss_ema.val > self.optimiser_args.disc_loss_threshold:
                self.optimiser_d.zero_grad()
                loss_d.backward()
                if self.optimiser_args.clip_grad_norm != -1:
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                             max_norm=self.optimiser_args.clip_grad_norm)
                self.optimiser_d.step()
            outputs.update(outputs_d)
            stats.update(stats_d)
            loss_total += loss_d.item()
            self.tb_logger.add_scalar('batch/train/loss_disc', loss_d, self.checkpoint.step)

        # Train the transcoder
        if self.transcoder_args.use_transcoder and self.transcoder_args.tc_trained_by == 'self':
            outputs_t, loss_t, stats_t = self._process_batch_transcoder(
                data,
                outputs['Z_pred'] if 'Z_pred' in outputs else None
            )
            self.optimiser_t.zero_grad()
            loss_t.backward()
            if self.optimiser_args.clip_grad_norm != -1:
                nn.utils.clip_grad_norm_(self.transcoder.parameters(), max_norm=self.optimiser_args.clip_grad_norm)
            self.optimiser_t.step()
            outputs.update(outputs_t)
            stats.update(stats_t)
            loss_total += loss_t.item()
            self.tb_logger.add_scalar('batch/train/loss_tc', loss_t, self.checkpoint.step)

        # Log losses
        self.tb_logger.add_scalar('batch/train/total_loss', loss_total, self.checkpoint.step)
        for key, val in stats.items():
            self.tb_logger.add_scalar(f'batch/train/{key}', float(val), self.checkpoint.step)

        # Calculate L2 loss for predictor network
        if self.dataset_args.train_predictor:
            norms = self.predictor.calc_norms()
            weights_cumulative_norm = torch.tensor(0., dtype=torch.float32, device=self.device)
            for _, norm in norms.items():
                weights_cumulative_norm += norm
            assert not is_bad(weights_cumulative_norm), 'Bad parameters! (Predictor network)'
            self.tb_logger.add_scalar('batch/train/w_norm', weights_cumulative_norm.item(), self.checkpoint.step)

        # Calculate L2 loss for generator network
        if self.dataset_args.train_generator:
            norms = self.generator.calc_norms()
            weights_cumulative_norm = torch.tensor(0., dtype=torch.float32, device=self.device)
            for _, norm in norms.items():
                weights_cumulative_norm += norm
            assert not is_bad(weights_cumulative_norm), 'Bad parameters! (Generator network)'
            self.tb_logger.add_scalar('batch/train/w_norm_gen', weights_cumulative_norm.item(), self.checkpoint.step)

        # Calculate L2 loss for discriminator network
        if self.dataset_args.train_generator and self.generator_args.use_discriminator:
            norms = self.discriminator.calc_norms()
            weights_cumulative_norm = torch.tensor(0., dtype=torch.float32, device=self.device)
            for _, norm in norms.items():
                weights_cumulative_norm += norm
            assert not is_bad(weights_cumulative_norm), 'Bad parameters! (Discriminator network)'
            self.tb_logger.add_scalar('batch/train/w_norm_disc', weights_cumulative_norm.item(), self.checkpoint.step)

        # Calculate L2 loss for transcoder network
        if self.transcoder_args.use_transcoder:
            norms = self.transcoder.calc_norms()
            weights_cumulative_norm = torch.tensor(0., dtype=torch.float32, device=self.device)
            for _, norm in norms.items():
                weights_cumulative_norm += norm
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
        if self.dataset_args.train_predictor:
            self.predictor.eval()
        if self.dataset_args.train_generator:
            self.generator.eval()
            if self.generator_args.use_discriminator:
                self.discriminator.eval()
        if self.transcoder_args.use_transcoder:
            self.transcoder.eval()
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
            data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, Any], torch.Tensor, Dict]:
        """
        Take a batch of images, predict the parameters and calculate the average loss per example.
        """
        _, X_target_og, X_target, Y_target = data  # Use the (possibly) augmented image as input
        X_target_og = X_target_og.to(self.device)
        X_target = X_target.to(self.device)
        Y_target = {
            k: [vi.to(self.device) for vi in v] if isinstance(v, list) else v.to(self.device)
            for k, v in Y_target.items()
        }

        # Predict the parameters and calculate losses
        Y_pred = self.predict(X_target)
        loss, metrics, X_pred2 = self.calculate_predictor_losses(Y_pred, Y_target, X_target_og)
        outputs = {
            'Y_pred': Y_pred,
            'X_pred2': X_pred2
        }

        # Update the EMA of the predictor loss
        self.p_loss_ema(loss.item())

        # If training the full auto encoder variant, then also do the generation step
        if self.dataset_args.train_combined:
            metrics['loss_com/pred'] = loss.item()
            Z = self.transcoder.latents_in
            outputs['Z_pred'] = Z.clone().detach()
            X_pred = self.generator(Z)
            outputs['X_pred'] = X_pred
            loss_g, metrics_g, Y_pred2 = self.calculate_generator_losses(X_pred, X_target, Y_target,
                                                                         include_teacher_loss=False,
                                                                         include_transcoder_loss=False)
            metrics.update(metrics_g)
            metrics['loss_gen'] = loss_g.item()
            self.g_loss_ema(metrics['losses/generator'])
            outputs['Y_pred2'] = Y_pred2

            # Replace loss with the generator loss plus the latents consistency loss
            l_z = self._calculate_com_loss(Z, Y_target)
            metrics['loss_com/X'] = l_z.item()
            loss = loss_g + self.optimiser_args.w_com_X * l_z

        return outputs, loss, metrics

    def _process_batch_generator(
            self,
            data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, Any], torch.Tensor, Dict]:
        """
        Take a batch of input data, push it through the network and calculate the average loss per example.
        """
        if self.dataset_args.train_combined:
            raise RuntimeError('Training the generator separately in train_combined mode is disabled.')
        _, X_target, _, Y_target = data  # Use the non-augmented image as the target
        X_target = X_target.to(self.device)
        Y_target = {
            k: [vi.to(self.device) for vi in v] if type(v) == list else v.to(self.device)
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
            data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
            X_pred: torch.Tensor
    ) -> Tuple[Dict[str, Any], torch.Tensor, Dict]:
        """
        Run the discriminator on the real and predicted images and calculate losses.
        """
        _, X_target, _, _ = data  # Use the non-augmented image as target
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

    def _process_batch_transcoder(
            self,
            data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
            Z_pred: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, Any], torch.Tensor, Dict]:
        """
        Run the transcoder on the parameter to latent mappings.
        """

        # VAE transcoder
        if self.transcoder_args.tc_model_name in ['vae', 'tvae']:
            _, _, _, Y_target = data
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
                l_z = self._calculate_com_loss(Z_pred, Y_target)
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
            Y_target: Union[torch.Tensor, Dict[str, torch.Tensor]],
            add_latent_noise: bool = True,
    ):
        """
        Encode and decode parameters through the transcoder.
        """
        if isinstance(Y_target, torch.Tensor):
            Y_target_vec = Y_target.to(self.device)
        else:
            Y_target_vec = torch.cat([Yk for k, Yk in Y_target.items() if k != 'sym_rotations'], dim=1).to(self.device)

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

    def predict(self, X: torch.Tensor, latent_input: bool = False) -> Dict[str, torch.Tensor]:
        """
        Take a batch of input data and return network output.
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
            Y: torch.Tensor,
            apply_sigmoid_to_switches: bool = True
    ) -> Dict[str, torch.Tensor]:
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
                Y_dists = torch.where(ignore_distances, torch.zeros_like(Y_dists), Y_dists)
            elif (self.ds.dataset_args.distance_constraints is not None
                  and len(self.ds.dataset_args.distance_constraints) > 0):
                # No need to normalise if there are constraints, but no negative predictions will be present
                Y_dists = torch.clamp(Y_dists, min=0)
            else:
                # Normalise the predictions by the maximum value per item
                Ypd_max = Y_dists.amax(dim=1, keepdim=True).abs()
                Y_dists = torch.where(Ypd_max > 0, Y_dists / Ypd_max, Y_dists)

            # Clip negative predictions to -1
            Y_dists = Y_dists.clamp(min=-1)

            output['distances'] = Y_dists

        if self.dataset_args.train_transformation:
            n_params = len(self.ds.labels_transformation)
            if self.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
                n_params += len(self.ds.labels_transformation_sincos)
            elif self.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                n_params += len(self.ds.labels_transformation_quaternion)
            else:
                assert self.dataset_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                n_params += len(self.ds.labels_transformation_axisangle)
            output['transformation'] = Y[:, idx:idx + n_params]
            idx += n_params

        if self.dataset_args.train_material and len(self.ds.labels_material_active) > 0:
            n_params = len(self.ds.labels_material_active)
            output['material'] = Y[:, idx:idx + n_params]
            idx += n_params

        if self.dataset_args.train_light:
            output['light'] = Y[:, idx:]

        return output

    def generate(self, Y: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Take a batch of parameters and generate a batch of images.
        """
        if not self.dataset_args.train_generator:
            return None
        Y_vector = torch.cat([Yk for k, Yk in Y.items() if k != 'sym_rotations'], dim=1)

        # Add some noise to the input parameters during training
        if self.generator.training and self.generator_args.gen_input_noise_std > 0:
            Y_vector = Y_vector + torch.randn_like(Y_vector) * self.generator_args.gen_input_noise_std

        # Transcode
        if self.transcoder_args.use_transcoder:
            Y_vector = self.transcoder(Y_vector, 'latents')

        X_pred = self.generator(Y_vector)

        return X_pred

    def calculate_predictor_losses(
            self,
            Y_pred: Dict[str, torch.Tensor],
            Y_target: Dict[str, torch.Tensor],
            X_target: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
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
            loss_t, stats_t = self._calculate_transformation_losses(Y_pred['transformation'],
                                                                    Y_target['transformation'],
                                                                    Y_target['sym_rotations'])
            losses.append(self.optimiser_args.w_transformation * loss_t)
            stats.update(stats_t)

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
            loss_t, stats_t, X_pred2 = self._calculate_net_teacher_losses(Y_pred, X_target)
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
            zingg_pred: torch.Tensor,
            zingg_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
            d_pred: torch.Tensor,
            d_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate distance losses.
        If using distance switches, ignored predicted distances have been set to 0.
        Otherwise, assuming negative values are predicting the missing distances.
        """
        d_pos = d_target > 0

        # Where there is a positive length in the target, calculate the difference
        Y_pred_pos = d_pred.clone()
        Y_pred_pos[d_target <= 0] = 0

        # Calculate average error per non-zero distance per item
        n_pos = d_pos.sum(dim=1)
        avg_errors_pos = ((Y_pred_pos - d_target)**2).sum(dim=1) / n_pos
        l_pos = avg_errors_pos.sum() / len(d_pred)

        # Where there is a missing distance in the target, penalise positive predictions
        Y_pred_neg = d_pred.clone()
        Y_pred_neg[d_target > 0] = 0

        # Calculate average error per non-zero distance per item
        n_neg = (~d_pos).sum(dim=1)
        avg_errors_neg = torch.where(
            n_neg > 0,
            (Y_pred_neg**2).sum(dim=1) / n_neg,
            torch.zeros(len(d_pred), device=self.device)
        )
        l_neg = avg_errors_neg.sum() / len(d_pred)

        loss = l_pos.sum() + l_neg.sum()

        stats = {
            'distances/l_pos': l_pos.item(),
            'distances/l_neg': l_neg.item(),
            'losses/distances': loss.item()
        }

        return loss, stats

    def _calculate_distance_switches_losses(
            self,
            s_pred: torch.Tensor,
            s_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
            t_pred: torch.Tensor,
            t_target: torch.Tensor,
            sym_rotations: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate the transformation losses.
        """
        location_loss = ((t_pred[:, :3] - t_target[:, :3])**2).mean()
        scale_loss = ((t_pred[:, 3] - t_target[:, 3])**2).mean()

        if self.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
            preangles_pred = t_pred[:, 4:].reshape(-1, 3, 2)
            preangles_target = t_target[:, 4:].reshape(-1, 3, 2)
            v_norms = preangles_pred.norm(dim=-1, keepdim=True)
            preangles_pred = preangles_pred / v_norms
            rotation_loss = ((preangles_pred - preangles_target)**2).mean()
        elif self.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
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
            assert self.dataset_args.preangles_mode == PREANGLES_MODE_AXISANGLE
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
        if self.dataset_args.preangles_mode != PREANGLES_MODE_AXISANGLE:
            preangles_loss = ((v_norms - 1)**2).mean()
            loss = loss + preangles_loss

        stats = {
            'transformation/l_location': location_loss.item(),
            'transformation/l_scale': scale_loss.item(),
            'transformation/l_rotation': rotation_loss.item(),
            'losses/transformation': loss.item()
        }

        if self.dataset_args.preangles_mode != PREANGLES_MODE_AXISANGLE:
            stats['transformation/l_preangles'] = preangles_loss.item()

        return loss, stats

    def _calculate_material_losses(
            self,
            m_pred: torch.Tensor,
            m_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
            l_pred: torch.Tensor,
            l_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate the light properties losses.
        """
        if self.ds.renderer_args.transmission_mode:
            assert l_pred.shape[-1] == 1
            energy_loss = ((l_pred - l_target)**2).mean()
            loss = energy_loss
            stats = {
                'light/l_energy': energy_loss.item(),
                'losses/light': loss.item()
            }

        else:
            location_loss = ((l_pred[:, :3] - l_target[:, :3])**2).mean()
            energy_loss = ((l_pred[:, 3] - l_target[:, 3])**2).mean()

            if self.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
                preangles_pred = l_pred[:, 4:].reshape(-1, 2, 2)
                preangles_target = l_target[:, 4:].reshape(-1, 2, 2)
                v_norms = preangles_pred.norm(dim=-1, keepdim=True)
                preangles_pred = preangles_pred / v_norms
                rotation_loss = ((preangles_pred - preangles_target)**2).mean()
            elif self.ds_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                q_pred = l_pred[:, 4:]
                q_target = l_target[:, 4:]
                v_norms = q_pred.norm(dim=-1, keepdim=True)
                q_pred = q_pred / v_norms
                rotation_loss = ((q_pred - q_target)**2).mean()
            else:
                assert self.dataset_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                R_pred = axis_angle_to_rotation_matrix(l_pred[:, 4:])
                R_target = axis_angle_to_rotation_matrix(l_target[:, 4:])
                rotation_loss = geodesic_distance(R_pred, R_target).mean()

            loss = location_loss + energy_loss + rotation_loss
            if self.dataset_args.preangles_mode != PREANGLES_MODE_AXISANGLE:
                preangles_loss = ((v_norms - 1)**2).mean()
                loss = loss + preangles_loss

            stats = {
                'light/l_location': location_loss.item(),
                'light/l_energy': energy_loss.item(),
                'light/l_rotation': rotation_loss.item(),
                'losses/light': loss.item()
            }

            if self.dataset_args.preangles_mode != PREANGLES_MODE_AXISANGLE:
                stats['light/l_preangles'] = preangles_loss.item()

        return loss, stats

    def _calculate_net_teacher_losses(
            self,
            Y_pred: Dict[str, torch.Tensor],
            X_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """
        Calculate the teacher losses using the generator network as the teacher.
        """
        # Run the predicted parameters through the generator to get another image
        if self.transcoder_args.use_transcoder:
            Z = self.transcoder.latents_in
        else:
            Z = torch.cat([Yk for k, Yk in Y_pred.items() if k != 'sym_rotations'], dim=1)
        X_pred2 = self.generator(Z)

        # Calculate how closely the same images come back out
        loss, _ = self._calculate_generator_losses(X_pred2, X_target)
        stats = {
            'losses/teacher_net': loss.item()
        }

        return loss, stats, X_pred2.detach().cpu()

    def calculate_generator_losses(
            self,
            X_pred: Optional[torch.Tensor],
            X_target: Optional[torch.Tensor],
            Y_target: Dict[str, torch.Tensor],
            include_teacher_loss: bool = True,
            include_discriminator_loss: bool = True,
            include_transcoder_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
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
            X_pred: torch.Tensor,
            X_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
            X_pred: torch.Tensor,
            Y_target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Calculate the teacher losses using the parameter network as the teacher.
        """
        # Run the predicted image through the parameter network to estimate the parameters (or latent vector logits)
        Y_pred2 = self.predictor(X_pred)
        stats = {}

        # Calculate loss in the latent space if using a transcoder
        if self.transcoder_args.use_transcoder:
            Z1 = self.transcoder.latents_out  # Latent vector from parameters via transcoder
            Z2 = self.transcoder.latent_activation_fn(Y_pred2)  # Latent vector from image via predictor
            loss = ((Z1 - Z2)**2).mean()
        else:
            # Calculate how closely the same parameters come back out
            loss, r_stats = self._calculate_parameter_reconstruction_losses(Y_target, Y_pred2, check_sym_rotations=True)
            for k, v in r_stats.items():
                stats[f'losses/teacher_gen/{k}'] = v

        stats['losses/teacher_gen'] = loss.item()

        return loss, stats, self.prepare_parameter_dict(Y_pred2.detach().cpu())

    def _calculate_gen_gan_losses(
            self,
            X_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate the GAN losses on the generator.
        """
        D_fake = self.discriminator(X_pred)
        loss = self.discriminator.gen_loss(D_fake)
        stats = {
            'losses/gan_gen': loss.item()
        }

        return loss, stats

    def calculate_discriminator_losses(
            self,
            D_real: torch.Tensor,
            D_fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            D_real: torch.Tensor,
            D_fake: torch.Tensor,
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
            Y_target: Dict[str, torch.Tensor],
            Yr_mu_vec: torch.Tensor,
            Yr_mu_clean: torch.Tensor,
            Z_mu_logits: torch.Tensor,
            Z_logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            Y_target: Dict[str, torch.Tensor],
            Y_pred_vec: torch.Tensor,
            check_sym_rotations: bool = False
    ):
        """
        Calculate the parameter errors.
        """
        Y_pred = self.prepare_parameter_dict(Y_pred_vec)
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
            sym_rotations = Y_target['sym_rotations'] if check_sym_rotations else None
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
            Y_target: Dict[str, torch.Tensor]
    ):
        """
        Calculate the parameter independence loss.
        """
        Y_vector = torch.cat([Yk for k, Yk in Y_target.items() if k != 'sym_rotations'], dim=1)
        noise_level = 0.1
        bs = len(Y_vector)

        # The parameter groups should be independent
        groups = {}
        Y_perturbations = []
        idx = 0

        def add_to_batch(start_idx, end_idx):
            noise = torch.randn_like(Y_vector) * noise_level
            noise[:, start_idx:end_idx] = 0
            Y_perturbations.append(Y_vector.clone() + noise)

        # Zinng parameters and distances define the morphology
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
            groups['t/location'] = (idx, idx + 3)
            idx += 3
            groups['t/scale'] = (idx, idx + 1)
            idx += 1
            if self.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
                groups['t/rotation'] = (idx, idx + 6)
                idx += 6
            elif self.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                groups['t/rotation'] = (idx, idx + 4)
                idx += 4
            else:
                assert self.dataset_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                groups['t/rotation'] = (idx, idx + 3)
                idx += 3

        # All material properties are independent
        if self.dataset_args.train_material:
            for i, k in enumerate(self.ds.labels_material_active):
                groups[f'm/{k}'] = (idx, idx + 1)
                idx += 1

        # Lighting has three independent components - location, energy and rotation
        if self.dataset_args.train_light:
            if self.ds.renderer_args.transmission_mode:
                groups['l/energy'] = (idx, idx + 1)
                idx += 1
            else:
                groups['l/location'] = (idx, idx + 3)
                idx += 3
                groups['l/energy'] = (idx, idx + 1)
                idx += 1
                if self.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
                    groups['l/rotation'] = (idx, idx + 6)
                    idx += 6
                elif self.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                    groups['l/rotation'] = (idx, idx + 4)
                    idx += 4
                else:
                    assert self.dataset_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                    groups['l/rotation'] = (idx, idx + 3)
                    idx += 3

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
            Z: torch.Tensor,
            Y_target: Dict[str, torch.Tensor],
            use_sym_rotations: bool = True
    ):
        """
        Calculate the latents loss - difference between the latent Z and the target latents from the parameters.
        """
        Y_target_vec = torch.cat([Yk for k, Yk in Y_target.items() if k != 'sym_rotations'], dim=1)

        # Make a new batch of Y_targets with the rotation picked to be as close to the predicted rotation as possible
        if use_sym_rotations:
            bs = len(Y_target_vec)
            q0 = self.ds.labels.index('rw')
            for i in range(bs):
                Y_target_vec[i, q0:q0 + 4] = Y_target['sym_rotations'][i][self.sym_group_idxs[i]]

        # Measure distance to the target latents
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
            data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
            outputs: Dict[str, Any],
            stats: Dict[str, torch.Tensor],
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
                self._plot_examples(data, outputs, train_or_test, idxs)
            if self.transcoder_args.use_transcoder:
                self._plot_vaetc_examples(data, outputs, train_or_test, idxs)

    def _plot_examples(
            self,
            data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
            outputs: Dict[str, Any],
            train_or_test: str,
            idxs: np.ndarray
    ):
        """
        Plot the image and parameter comparisons.
        """
        n_examples = min(self.runtime_args.plot_n_examples, self.runtime_args.batch_size)
        metas, images, images_aug, Y_target = data
        Y_pred = outputs['Y_pred']
        if self.dataset_args.train_generator:
            X_pred = outputs['X_pred']
            X_pred2 = outputs['X_pred2']
            Y_pred2 = outputs['Y_pred2']
        n_rows = 4 \
                 + int(self.dataset_args.train_zingg) \
                 + int(self.dataset_args.train_distances) \
                 + int(self.dataset_args.train_transformation) \
                 + int(self.dataset_args.train_material and len(self.ds.labels_material_active) > 0) \
                 + int(self.dataset_args.train_light) \
                 + int(self.dataset_args.train_generator) * 2

        height_ratios = [1.2, 1.2, 1, 1]  # images and 3d plots
        if self.dataset_args.train_zingg:
            height_ratios.append(0.5)
        if self.dataset_args.train_distances:
            height_ratios.append(1)
        if self.dataset_args.train_transformation:
            height_ratios.append(0.7)
        if self.dataset_args.train_material and len(self.ds.labels_material_active) > 0:
            height_ratios.append(0.7)
        if self.dataset_args.train_light:
            height_ratios.append(0.7)
        if self.dataset_args.train_generator:
            height_ratios.insert(0, 1.2)
            height_ratios.insert(0, 1.2)

        fig = plt.figure(figsize=(n_examples * 2.6, n_rows * 2.4))
        gs = GridSpec(
            nrows=n_rows,
            ncols=n_examples,
            wspace=0.06,
            hspace=0.4,
            width_ratios=[1] * n_examples,
            height_ratios=height_ratios,
            top=0.97,
            bottom=0.015,
            left=0.04,
            right=0.99
        )

        loss = getattr(self.checkpoint, f'loss_{train_or_test}')
        fig.suptitle(
            f'epoch={self.checkpoint.epoch}, '
            f'step={self.checkpoint.step + 1}, '
            f'loss={loss:.4E}',
            fontweight='bold',
            y=0.995
        )

        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colours = prop_cycle.by_key()['color']
        share_ax = {}

        def plot_error(ax_, err_):
            txt = ax_.text(
                0.5, 0.5, err_,
                horizontalalignment='center',
                verticalalignment='center',
                wrap=True
            )
            txt._get_wrap_line_width = lambda: ax_.bbox.width * 0.7
            ax_.axis('off')

        def plot_image(ax_, title_, img_):
            ax_.set_title(title_)
            ax_.imshow(img_, cmap='gray', vmin=0, vmax=1)
            ax_.axis('off')

        def add_discriminator_value(ax_, D_key_, idx_):
            if D_key_ in outputs:
                d_val = outputs[D_key_][idx_].item()
                colour = 'red' if d_val < 0 else 'green'
                ax_.text(
                    0.99, -0.02, f'{d_val:.3E}',
                    ha='right', va='top', transform=ax.transAxes,
                    color=colour, fontsize=14
                )

        def plot_3d(ax_, title_, morph_, mesh_, colour_):
            ax_.set_title(title_)
            if morph_ is not None:
                for f in morph_.facets:
                    for edge in f.edges:
                        coords = np.array(edge)
                        ax_.plot(*coords.T, c=colour_)
            ax_.plot_trisurf(
                mesh_.vertices[:, 0],
                mesh_.vertices[:, 1],
                triangles=mesh_.faces,
                Z=mesh_.vertices[:, 2],
                color=colour_,
                alpha=0.5
            )
            equal_aspect_ratio(ax_)

        def plot_zingg(ax_, idx_):
            z_pred = to_numpy(Y_pred['zingg'][idx_])
            z_target = to_numpy(Y_target['zingg'][idx_])
            ax_.scatter(z_target[0], z_target[1], c='r', marker='x', s=100, label='Target')
            ax_.scatter(z_pred[0], z_pred[1], c='b', marker='o', s=100, label='Predicted')
            ax_.set_title('Zingg')
            ax_.set_xlabel('S/I', labelpad=-5)
            ax_.set_xlim(0, 1)
            ax_.set_xticks([0, 1])
            if 'zingg' not in share_ax:
                ax_.legend()
                ax_.set_ylim(0, 1)
                ax_.set_yticks([0, 1])
                ax_.set_ylabel('I/L', labelpad=0)
                share_ax['zingg'] = ax_
            else:
                ax_.sharey(share_ax['zingg'])
                ax_.yaxis.set_tick_params(labelleft=False)

        def plot_distances(ax_, idx_):
            ax_pos = ax_.get_position()
            ax_.set_position([ax_pos.x0, ax_pos.y0 + 0.02, ax_pos.width, ax_pos.height - 0.02])
            d_pred = to_numpy(Y_pred['distances'][idx_])
            d_target = to_numpy(Y_target['distances'][idx_])
            if self.dataset_args.train_generator:
                d_pred2 = to_numpy(Y_pred2['distances'][idx_])

            # Clip predictions to -1 to avoid large negatives skewing the plots
            d_pred = np.clip(d_pred, a_min=-1, a_max=np.inf)

            locs = np.arange(len(d_target))
            if self.dataset_args.train_generator:
                bar_width = 0.25
                offset = bar_width
                ax_.bar(locs - offset, d_target, bar_width, label='Target')
                ax_.bar(locs, d_pred, bar_width, label='Predicted')
                ax_.bar(locs + offset, d_pred2, bar_width, label='Predicted2')
            else:
                bar_width = 0.35
                offset = bar_width / 2
                ax_.bar(locs - offset, d_target, bar_width, label='Target')
                ax_.bar(locs + offset, d_pred, bar_width, label='Predicted')

            if self.dataset_args.use_distance_switches:
                s_pred = to_numpy(Y_pred['distance_switches'][idx_])
                s_target = to_numpy(Y_target['distance_switches'][idx_])
                k = 2.3
                colours = []
                for i, (sp, st) in enumerate(zip(s_pred, s_target)):
                    if st > 0.5:
                        ax_.axvspan(i - k * offset, i, alpha=0.1, color='blue')
                    if sp > 0.5:
                        ax_.axvspan(i, i + k * offset, alpha=0.1, color='red' if st < 0.5 else 'green')
                    colours.append('red' if (st < 0.5 < sp) or (st > 0.5 > sp) else 'green')
                ax_.scatter(locs + offset, s_pred, color=colours, marker='+', s=100, label='Switches')

            ax_.set_title('Distances')
            ax_.set_xticks(locs)
            ax_.set_xticklabels(self.ds.labels_distances_active)
            ax_.tick_params(axis='x', rotation=270)
            if 'distances' not in share_ax:
                ax_.legend()
                share_ax['distances'] = ax_
            else:
                ax_.sharey(share_ax['distances'])
                ax_.yaxis.set_tick_params(labelleft=False)
                ax_.autoscale()

        def plot_transformation(ax_, idx_):
            t_pred = to_numpy(Y_pred['transformation'][idx_])
            t_target = to_numpy(Y_target['transformation'][idx_])
            if self.dataset_args.train_generator:
                t_pred2 = to_numpy(Y_pred2['transformation'][idx_])

            # Adjust the target rotation to the best matching symmetry group
            if self.dataset_args.preangles_mode in [PREANGLES_MODE_QUATERNION, PREANGLES_MODE_AXISANGLE]:
                sgi = self.sym_group_idxs[idx_]
                if self.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                    t_target[4:] = to_numpy(Y_target['sym_rotations'][idx_][sgi])
                else:
                    R = to_numpy(Y_target['sym_rotations'][idx_][sgi])
                    t_target[4:] = Rotation.from_matrix(R).as_rotvec()

            locs = np.arange(len(t_target))

            if self.dataset_args.train_generator:
                bar_width = 0.25
                offset = bar_width
                ax_.bar(locs - offset, t_target, bar_width, label='Target')
                ax_.bar(locs, t_pred, bar_width, label='Predicted')
                ax_.bar(locs + offset, t_pred2, bar_width, label='Predicted2')
                k = 2 * offset
            else:
                bar_width = 0.35
                offset = bar_width / 2
                ax_.bar(locs - offset, t_target, bar_width, label='Target')
                ax_.bar(locs + offset, t_pred, bar_width, label='Predicted')
                k = 3 * offset
            ax_.axvspan(locs[0] - k, locs[2] + k, alpha=0.1, color='green')
            ax_.axvspan(locs[3] - k, locs[3] + k, alpha=0.1, color='red')
            ax_.axvspan(locs[4] - k, locs[-1] + k, alpha=0.1, color='blue')
            ax_.set_title('Transformation')
            ax_.set_xticks(locs)
            xlabels = self.ds.labels_transformation.copy()
            if self.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
                xlabels += self.ds.labels_transformation_sincos
            elif self.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                xlabels += self.ds.labels_transformation_quaternion
            else:
                xlabels += self.ds.labels_transformation_axisangle
            ax_.set_xticklabels(xlabels)
            if 'transformation' not in share_ax:
                ax_.legend()
                share_ax['transformation'] = ax_
            else:
                ax_.sharey(share_ax['transformation'])
                ax_.yaxis.set_tick_params(labelleft=False)
                ax_.autoscale()

        def plot_material(ax_, idx_):
            m_pred = to_numpy(Y_pred['material'][idx_])
            m_target = to_numpy(Y_target['material'][idx_])
            if self.dataset_args.train_generator:
                m_pred2 = to_numpy(Y_pred2['material'][idx_])

            locs = np.arange(len(m_target))
            if self.dataset_args.train_generator:
                bar_width = 0.25
                offset = bar_width
                ax_.bar(locs - offset, m_target, bar_width, label='Target')
                ax_.bar(locs, m_pred, bar_width, label='Predicted')
                ax_.bar(locs + offset, m_pred2, bar_width, label='Predicted2')
            else:
                bar_width = 0.35
                offset = bar_width / 2
                ax_.bar(locs - offset, m_target, bar_width, label='Target')
                ax_.bar(locs + offset, m_pred, bar_width, label='Predicted')
            ax_.set_title('Material')
            ax_.set_xticks(locs)
            for i in range(len(locs) - 1):
                ax_.axvline(locs[i] + .5, color='black', linestyle='--', linewidth=1)
            labels = []
            if 'b' in self.ds.labels_material_active:
                labels.append('Brightness')
            if 'ior' in self.ds.labels_material_active:
                labels.append('IOR')
            if 'r' in self.ds.labels_material_active:
                labels.append('Roughness')
            ax_.set_xticklabels(labels)
            if 'material' not in share_ax:
                ax_.legend()
                share_ax['material'] = ax_
            else:
                ax_.sharey(share_ax['material'])
                ax_.yaxis.set_tick_params(labelleft=False)
                ax_.autoscale()

        def plot_light(ax_, idx_):
            l_pred = to_numpy(Y_pred['light'][idx_])
            l_target = to_numpy(Y_target['light'][idx_])
            locs = np.arange(len(l_target))
            bar_width = 0.35
            offset = bar_width / 2
            ax_.bar(locs - offset, l_target, bar_width, label='Target')
            ax_.bar(locs + offset, l_pred, bar_width, label='Predicted')
            ax_.set_title('Light')
            ax_.set_xticks(locs)

            if self.ds.renderer_args.transmission_mode:
                xlabels = self.ds.labels_light.copy()
            else:
                xlabels = self.ds.labels_light_location.copy()
                xlabels += self.ds.labels_light.copy()
                if self.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
                    xlabels += self.ds.labels_light_sincos
                elif self.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                    xlabels += self.ds.labels_light_quaternion
                else:
                    xlabels += self.ds.labels_light_axisangle

                k = 3
                ax_.axvspan(locs[0] - k * offset, locs[2] + k * offset, alpha=0.1, color='green')
                ax_.axvspan(locs[3] - k * offset, locs[3] + k * offset, alpha=0.1, color='red')
                ax_.axvspan(locs[4] - k * offset, locs[-1] + k * offset, alpha=0.1, color='blue')

            ax_.set_xticklabels(xlabels)
            if 'light' not in share_ax:
                ax_.legend()
                share_ax['light'] = ax_
            else:
                ax_.sharey(share_ax['light'])
                ax_.yaxis.set_tick_params(labelleft=False)
                ax_.autoscale()

        def prep_distances(distance_vals_):
            distances = np.zeros(len(self.ds.labels_distances))
            pos_active = [self.ds.labels_distances.index(k) for k in self.ds.labels_distances_active]
            for i, pos in enumerate(pos_active):
                distances[pos] = distance_vals_[i]
            if self.crystal_generator.constraints is not None:
                largest_hkl = ''.join([str(hkl) for hkl in self.crystal_generator.constraints[0]])
                largest_pos = [d[-3:] for d in self.ds.labels_distances].index(largest_hkl)
                distances[largest_pos] = 1
            return distances

        for i, idx in enumerate(idxs):
            meta = metas[idx]
            row_idx = 0

            # Plot the (possibly augmented) input image
            img = to_numpy(images_aug[idx]).squeeze()
            ax = fig.add_subplot(gs[row_idx, i])
            plot_image(ax, meta['image'].name, img)
            add_discriminator_value(ax, 'D_real', idx)
            row_idx += 1

            # Plot the generated image
            if self.dataset_args.train_generator:
                img = to_numpy(X_pred[idx]).squeeze()
                ax = fig.add_subplot(gs[row_idx, i])
                plot_image(ax, 'Generated', img)
                add_discriminator_value(ax, 'D_fake', idx)
                row_idx += 1
                img = to_numpy(X_pred2[idx]).squeeze()
                ax = fig.add_subplot(gs[row_idx, i])
                plot_image(ax, 'Generated2', img)
                row_idx += 1

            # Rebuild the target crystal
            if CCDC_AVAILABLE:
                distance_vals = to_numpy(Y_target['distances'][idx]).astype(float)
                distances = prep_distances(distance_vals)
                growth_rates = self.crystal_generator.get_expanded_growth_rates(distances)
                morph_target = VisualHabitMorphology.from_growth_rates(self.crystal_generator.crystal, growth_rates)
            else:
                morph_target = None

            # Plot the 3d crystal
            mesh_target = self.ds.load_mesh(meta['idx'])
            plot_3d(fig.add_subplot(gs[row_idx + 1, i], projection='3d'), 'Morphology',
                    morph_target, mesh_target, default_colours[0])

            # If CCDC isn't available then we can't build the predicted crystal
            if CCDC_AVAILABLE:
                mesh_pred = None

                # Normalise the predicted distances
                distance_vals = to_numpy(Y_pred['distances'][idx]).astype(float)
                if self.dataset_args.use_distance_switches:
                    switches = to_numpy(Y_pred['distance_switches'][idx])
                    distance_vals = np.where(switches < .5, 0, distance_vals)
                distance_vals[distance_vals < 0] = 0
                distances = prep_distances(distance_vals)
                if distances.max() > 1e-8:
                    # Build the crystal and plot the mesh
                    distances /= distances.max()
                    try:
                        growth_rates = self.crystal_generator.get_expanded_growth_rates(distances)
                        morph_pred = VisualHabitMorphology.from_growth_rates(self.crystal_generator.crystal,
                                                                             growth_rates)
                        _, _, mesh_pred = self.crystal_generator.generate_crystal(rel_rates=distances,
                                                                                  validate=False)
                        plot_3d(fig.add_subplot(gs[row_idx + 2, i], projection='3d'), 'Predicted',
                                morph_pred, mesh_pred, default_colours[1])
                    except Exception as e:
                        plot_error(fig.add_subplot(gs[row_idx + 2, i]), f'Failed to build crystal:\n{e}')
                else:
                    plot_error(fig.add_subplot(gs[row_idx + 2, i]), 'Failed to build crystal:\nno positive distances.')

                # Render the crystal
                if mesh_pred is not None:
                    try:
                        img = render_from_parameters(
                            mesh=mesh_pred,
                            settings_path=self.ds.path / 'vcw_settings.json',
                            r_params=self.ds.denormalise_rendering_params(Y_pred, idx,
                                                                          metas[idx]['rendering_parameters']),
                            attempts=1
                        )
                        plot_image(fig.add_subplot(gs[row_idx, i]), 'Blender', img)
                    except Exception as e:
                        plot_error(fig.add_subplot(gs[row_idx, i]), f'Rendering failed:\n{e}')
                else:
                    plot_error(fig.add_subplot(gs[row_idx, i]), 'No crystal to render.')
            else:
                plot_error(fig.add_subplot(gs[row_idx, i]), 'CCDC unavailable.')
                plot_error(fig.add_subplot(gs[row_idx + 2, i]), 'CCDC unavailable')

            row_idx += 3
            if self.dataset_args.train_zingg:
                plot_zingg(fig.add_subplot(gs[row_idx, i]), idx)
                row_idx += 1
            if self.dataset_args.train_distances:
                plot_distances(fig.add_subplot(gs[row_idx, i]), idx)
                row_idx += 1
            if self.dataset_args.train_transformation:
                plot_transformation(fig.add_subplot(gs[row_idx, i]), idx)
                row_idx += 1
            if self.dataset_args.train_material and len(self.ds.labels_material_active) > 0:
                plot_material(fig.add_subplot(gs[row_idx, i]), idx)
                row_idx += 1
            if self.dataset_args.train_light:
                plot_light(fig.add_subplot(gs[row_idx, i]), idx)

        self._save_plot(fig, 'samples', train_or_test)

    def _plot_vaetc_examples(
            self,
            data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
            outputs: Dict[str, Any],
            train_or_test: str,
            idxs: np.ndarray
    ):
        """
        Plot some VAE transcoder examples.
        """
        n_examples = min(self.runtime_args.plot_n_examples, self.runtime_args.batch_size)
        metas, images, images_aug, params = data
        Yr_noisy = outputs['Yr_mu']
        Yr_clean = outputs['Yr_mu_clean']
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colours = prop_cycle.by_key()['color']
        n_rows = int(self.dataset_args.train_zingg) \
                 + int(self.dataset_args.train_distances) \
                 + int(self.dataset_args.train_transformation) \
                 + int(self.dataset_args.train_material and len(self.ds.labels_material_active) > 0) \
                 + int(self.dataset_args.train_light)

        # Plot properties
        bar_width = 0.3
        sep = 0.05

        height_ratios = []
        if self.dataset_args.train_zingg:
            height_ratios.append(0.5)
        if self.dataset_args.train_distances:
            height_ratios.append(1)
        if self.dataset_args.train_transformation:
            height_ratios.append(0.8)
        if self.dataset_args.train_material and len(self.ds.labels_material_active) > 0:
            height_ratios.append(0.7)
        if self.dataset_args.train_light:
            height_ratios.append(0.8)

        fig = plt.figure(figsize=(n_examples * 2.6, n_rows * 2.3))
        gs = GridSpec(
            nrows=n_rows,
            ncols=n_examples,
            wspace=0.06,
            hspace=0.4,
            width_ratios=[1] * n_examples,
            height_ratios=height_ratios,
            top=0.95,
            bottom=0.04,
            left=0.05,
            right=0.99
        )

        loss = getattr(self.checkpoint, f'loss_{train_or_test}')
        fig.suptitle(
            f'epoch={self.checkpoint.epoch}, '
            f'step={self.checkpoint.step + 1}, '
            f'loss={loss:.4E}',
            fontweight='bold',
            y=0.995
        )
        share_ax = {}

        def plot_zingg(ax_, idx_):
            z_noisy = to_numpy(Yr_noisy['zingg'][idx_])
            z_clean = to_numpy(Yr_clean['zingg'][idx_])
            z_target = to_numpy(params['zingg'][idx_])
            ax_.scatter(z_target[0], z_target[1], c=default_colours[0], marker='o', s=100, label='Target')
            ax_.scatter(z_noisy[0], z_noisy[1], c=default_colours[1], marker='+', s=100, label='Sample')
            ax_.scatter(z_noisy[0], z_clean[1], c=default_colours[2], marker='x', s=100, label='Clean')
            ax_.set_title('Zingg')
            ax_.set_xlabel('S/I', labelpad=-5)
            ax_.set_xlim(0, 1)
            ax_.set_xticks([0, 1])
            if 'zingg' not in share_ax:
                ax_.legend()
                ax_.set_ylim(0, 1)
                ax_.set_yticks([0, 1])
                ax_.set_ylabel('I/L', labelpad=0)
                share_ax['zingg'] = ax_
            else:
                ax_.sharey(share_ax['zingg'])
                ax_.yaxis.set_tick_params(labelleft=False)

        def _plot_bar_chart(key_, idx_, ax_, labels_, title_):
            noisy_ = to_numpy(Yr_noisy[key_][idx_])
            clean_ = to_numpy(Yr_clean[key_][idx_])
            target_ = to_numpy(params[key_][idx_])
            locs = np.arange(len(target_))
            ax_.bar(locs - bar_width, target_, bar_width - sep / 2, label='Target')
            ax_.bar(locs, noisy_, bar_width - sep / 2, label='Noisy')
            ax_.bar(locs + bar_width, clean_, bar_width - sep / 2, label='Clean')
            ax_.set_title(title_)
            ax_.set_xticks(locs)
            ax_.set_xticklabels(labels_)
            if key_ not in share_ax:
                ax_.legend()
                share_ax[key_] = ax_
            else:
                ax_.sharey(share_ax[key_])
                ax_.yaxis.set_tick_params(labelleft=False)
                ax_.autoscale()
            return locs

        def plot_distances(ax_, idx_):
            ax_pos = ax_.get_position()
            ax_.set_position([ax_pos.x0, ax_pos.y0 + 0.02, ax_pos.width, ax_pos.height - 0.02])
            _plot_bar_chart('distances', idx_, ax_, self.ds.labels_distances_active, 'Distances')
            ax_.tick_params(axis='x', rotation=270)

            if self.dataset_args.use_distance_switches:
                s_noisy = to_numpy(Yr_noisy['distance_switches'][idx_])
                s_clean = to_numpy(Yr_clean['distance_switches'][idx_])
                s_target = to_numpy(params['distance_switches'][idx_])
                locs = np.arange(len(s_target))
                colours = []
                for i, (sn, sc, st) in enumerate(zip(s_noisy, s_clean, s_target)):
                    if st > 0.5:
                        ax_.axvspan(i - 3 / 2 * bar_width, i - 1 / 2 * bar_width, alpha=0.1, color='blue')
                    if sn > 0.5:
                        ax_.axvspan(i - 1 / 2 * bar_width, i + 1 / 2 * bar_width, alpha=0.1,
                                    color='red' if sn < 0.5 else 'green')
                    if sc > 0.5:
                        ax_.axvspan(i + 1 / 2 * bar_width, i + 3 / 2 * bar_width, alpha=0.1,
                                    color='red' if st < 0.5 else 'green')
                    colours.append('red' if (st < 0.5 < sc) or (st > 0.5 > sc) else 'green')
                ax_.scatter(locs, s_noisy, color=colours, marker='+', s=30, label='Switches')
                ax_.scatter(locs + bar_width, s_clean, color=colours, marker='+', s=30)

        def plot_transformation(ax_, idx_):
            xlabels = self.ds.labels_transformation.copy()
            if self.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
                xlabels += self.ds.labels_transformation_sincos
            elif self.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                xlabels += self.ds.labels_transformation_quaternion
            else:
                xlabels += self.ds.labels_transformation_axisangle
            _plot_bar_chart('transformation', idx_, ax_, xlabels, 'Transformation')
            locs = np.arange(len(xlabels))
            offset = 1.6 * bar_width
            ax_.axvspan(locs[0] - offset, locs[2] + offset, alpha=0.1, color='green')
            ax_.axvspan(locs[3] - offset, locs[3] + offset, alpha=0.1, color='red')
            ax_.axvspan(locs[4] - offset, locs[-1] + offset, alpha=0.1, color='blue')

        def plot_material(ax_, idx_):
            labels = []
            if 'b' in self.ds.labels_material_active:
                labels.append('Brightness')
            if 'ior' in self.ds.labels_material_active:
                labels.append('IOR')
            if 'r' in self.ds.labels_material_active:
                labels.append('Roughness')
            locs = _plot_bar_chart('material', idx_, ax_, labels, 'Material')
            for i in range(len(locs) - 1):
                ax_.axvline(locs[i] + .5, color='black', linestyle='--', linewidth=1)

        def plot_light(ax_, idx_):
            if self.ds.renderer_args.transmission_mode:
                xlabels = self.ds.labels_light.copy()
            else:
                xlabels = self.ds.labels_light_location.copy()
                xlabels += self.ds.labels_light
                if self.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
                    xlabels += self.ds.labels_light_sincos
                elif self.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                    xlabels += self.ds.labels_light_quaternion
                else:
                    xlabels += self.ds.labels_light_axisangle
            locs = _plot_bar_chart('light', idx_, ax_, xlabels, 'Light')
            if not self.ds.renderer_args.transmission_mode:
                offset = 1.6 * bar_width
                ax_.axvspan(locs[0] - offset, locs[2] + offset, alpha=0.1, color='green')
                ax_.axvspan(locs[3] - offset, locs[3] + offset, alpha=0.1, color='red')
                ax_.axvspan(locs[4] - offset, locs[-1] + offset, alpha=0.1, color='blue')

        for i, idx in enumerate(idxs):
            row_idx = 0
            if self.dataset_args.train_zingg:
                plot_zingg(fig.add_subplot(gs[row_idx, i]), idx)
                row_idx += 1
            if self.dataset_args.train_distances:
                plot_distances(fig.add_subplot(gs[row_idx, i]), idx)
                row_idx += 1
            if self.dataset_args.train_transformation:
                plot_transformation(fig.add_subplot(gs[row_idx, i]), idx)
                row_idx += 1
            if self.dataset_args.train_material and len(self.ds.labels_material_active) > 0:
                plot_material(fig.add_subplot(gs[row_idx, i]), idx)
                row_idx += 1
            if self.dataset_args.train_light:
                plot_light(fig.add_subplot(gs[row_idx, i]), idx)

        self._save_plot(fig, 'vaetc', train_or_test)

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
