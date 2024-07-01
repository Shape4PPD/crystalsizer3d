from argparse import ArgumentParser, _ArgumentGroup
from typing import List

import torch
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch import nn

from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.util.utils import str2bool

GAN_LOSS_OPTIONS = ['hinge', 'bce']


class OptimiserArgs(BaseArgs):
    def __init__(
            self,
            algorithm: str,

            lr_init: float = 0.1,
            lr_pretrained_init: float = 0.1,
            lr_generator_init: float = 0.1,
            lr_discriminator_init: float = 0.1,
            lr_denoiser_init: float = 0.1,
            lr_transcoder_init: float = 0.1,
            lr_scheduler: str = 'cosine',
            lr_min: float = 1e-6,
            lr_warmup: float = 1e-5,
            lr_warmup_epochs: int = 0,
            lr_decay_epochs: int = 30,
            lr_decay_milestones: List[int] = [],
            lr_cooldown_epochs: int = 0,
            lr_patience_epochs: int = 10,
            lr_decay_rate: float = 0.1,
            lr_cycle_mul: float = 1.0,
            lr_cycle_decay: float = 0.1,
            lr_cycle_limit: int = 1,
            lr_k_decay: float = 1.0,

            accumulate_grad_batches: int = 1,
            weight_decay: float = 1e-5,
            clip_grad_norm: float = -1,
            freeze_pretrained: bool = False,
            gan_loss: str = 'hinge',
            disc_loss_threshold: float = 0.1,

            teach_threshold_pred: float = 1e8,
            teach_threshold_gen: float = 1e8,
            teach_threshold_combined: bool = True,

            w_zingg: float = 1.0,
            w_distances: float = 1.0,
            w_transformation: float = 1.0,
            w_3d: float = 1.0,
            w_3d_overshoot: float = 0.,
            w_3d_undershoot: float = 0.,
            w_material: float = 1.0,
            w_light: float = 1.0,
            w_generator: float = 1.0,
            w_net_teacher: float = 1.0,
            w_gen_teacher: float = 1.0,
            w_discriminator: float = 1.0,
            w_rcf: float = 0.,
            w_transcoder_pred: float = 1.0,
            w_transcoder_gen: float = 1.0,
            w_com_X: float = 0.,
            w_com_Y: float = 0.,

            **kwargs
    ):
        # Optimisation algorithm
        p = nn.Parameter(torch.tensor(0.))
        opt = create_optimizer_v2(model_or_params=[p], opt=algorithm)  # check it is supported
        self.algorithm = algorithm

        # Learning rates and scheduling
        sched, _ = create_scheduler_v2(
            optimizer=opt,
            sched=lr_scheduler,
            num_epochs=300,
            decay_epochs=lr_decay_epochs,
            decay_milestones=lr_decay_milestones,
            cooldown_epochs=lr_cooldown_epochs,
            patience_epochs=lr_patience_epochs,
            decay_rate=lr_decay_rate,
            min_lr=lr_min,
            warmup_lr=lr_warmup,
            warmup_epochs=lr_warmup_epochs,
            cycle_mul=lr_cycle_mul,
            cycle_decay=lr_cycle_decay,
            cycle_limit=lr_cycle_limit,
            k_decay=lr_k_decay
        )
        assert sched is not None, f'Unsupported lr_scheduler: {lr_scheduler}'
        self.lr_init = lr_init
        self.lr_pretrained_init = lr_pretrained_init
        self.lr_generator_init = lr_generator_init
        self.lr_discriminator_init = lr_discriminator_init
        self.lr_denoiser_init = lr_denoiser_init
        self.lr_transcoder_init = lr_transcoder_init
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.lr_warmup = lr_warmup
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_decay_epochs = lr_decay_epochs
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_cooldown_epochs = lr_cooldown_epochs
        self.lr_patience_epochs = lr_patience_epochs
        self.lr_decay_rate = lr_decay_rate
        self.lr_cycle_mul = lr_cycle_mul
        self.lr_cycle_decay = lr_cycle_decay
        self.lr_cycle_limit = lr_cycle_limit
        self.lr_k_decay = lr_k_decay

        # Gradient accumulation
        self.accumulate_grad_batches = accumulate_grad_batches

        # Weight decay
        self.weight_decay = weight_decay

        # Gradient clipping
        self.clip_grad_norm = clip_grad_norm

        # Optimise the pretrained weights?
        self.freeze_pretrained = freeze_pretrained

        # GAN loss
        assert gan_loss in GAN_LOSS_OPTIONS, f'Unsupported gan_loss: {gan_loss}'
        self.gan_loss = gan_loss
        self.disc_loss_threshold = disc_loss_threshold

        # Teacher loss thresholds - only include teacher losses when the teacher's loss is low enough
        self.teach_threshold_pred = teach_threshold_pred
        self.teach_threshold_gen = teach_threshold_gen
        self.teach_threshold_combined = teach_threshold_combined

        # Loss weightings
        self.w_zingg = w_zingg
        self.w_distances = w_distances
        self.w_transformation = w_transformation
        self.w_3d = w_3d
        self.w_3d_overshoot = w_3d_overshoot
        self.w_3d_undershoot = w_3d_undershoot
        self.w_material = w_material
        self.w_light = w_light
        self.w_generator = w_generator
        self.w_net_teacher = w_net_teacher
        self.w_gen_teacher = w_gen_teacher
        self.w_discriminator = w_discriminator
        self.w_rcf = w_rcf
        self.w_transcoder_pred = w_transcoder_pred
        self.w_transcoder_gen = w_transcoder_gen
        self.w_com_X = w_com_X
        self.w_com_Y = w_com_Y

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Optimiser Args')
        group.add_argument('--algorithm', type=str, default='AdamW',
                           help='Optimisation algorithm.')

        group.add_argument('--lr-init', type=float, default=0.1,
                           help='Initial learning rate.')
        group.add_argument('--lr-pretrained-init', type=float, default=0.1,
                           help='Learning rate for pretrained model parameters.')
        group.add_argument('--lr-generator-init', type=float, default=0.1,
                           help='Learning rate for generator network.')
        group.add_argument('--lr-discriminator-init', type=float, default=0.1,
                           help='Learning rate for discriminator network.')
        group.add_argument('--lr-denoiser-init', type=float, default=0.1,
                           help='Learning rate for denoiser network.')
        group.add_argument('--lr-transcoder-init', type=float, default=0.1,
                           help='Learning rate for transcoder network (if trained by self).')
        group.add_argument('--lr-scheduler', type=str, default='cosine',
                           help='Learning rate scheduler.')
        group.add_argument('--lr-min', type=float, default=1e-6,
                           help='Minimum learning rate.')
        group.add_argument('--lr-warmup', type=float, default=1e-5,
                           help='Warmup learning rate.')
        group.add_argument('--lr-warmup-epochs', type=int, default=0,
                           help='Number of warmup epochs.')
        group.add_argument('--lr-decay-epochs', type=int, default=30,
                           help='Number of epochs before learning rate decay.')
        group.add_argument('--lr-decay-milestones', type=lambda s: [int(item) for item in s.split(',')], default=[],
                           help='Epochs at which to decay learning rate.')
        group.add_argument('--lr-cooldown-epochs', type=int, default=0,
                           help='Number of epochs before learning rate cooldown.')
        group.add_argument('--lr-patience-epochs', type=int, default=10,
                           help='Number of epochs before learning rate patience.')
        group.add_argument('--lr-decay-rate', type=float, default=0.1,
                           help='Learning rate decay rate.')
        group.add_argument('--lr-cycle-mul', type=float, default=1.0,
                           help='Learning rate cycle multiplier.')
        group.add_argument('--lr-cycle-decay', type=float, default=0.1,
                           help='Learning rate cycle decay.')
        group.add_argument('--lr-cycle-limit', type=int, default=1,
                           help='Learning rate cycle limit.')
        group.add_argument('--lr-k-decay', type=float, default=1.0,
                           help='Learning rate k decay.')

        group.add_argument('--accumulate-grad-batches', type=int, default=1,
                           help='Number of batches to accumulate gradients over.')
        group.add_argument('--weight-decay', type=float, default=1e-5,
                           help='Weight decay.')
        group.add_argument('--clip-grad-norm', type=float, default=-1,
                           help='Clip gradient norm. -1 disables clipping.')
        group.add_argument('--freeze-pretrained', type=str2bool, default=False,
                           help='Freeze pretrained model parameters.')
        group.add_argument('--gan-loss', type=str, default='hinge', choices=GAN_LOSS_OPTIONS,
                           help='GAN loss type. Only used when there is a discriminator.')
        group.add_argument('--disc-loss-threshold', type=float, default=0.1,
                           help='Only train the discriminator when it\'s loss is bigger than this.')

        group.add_argument('--teach-threshold-pred', type=float, default=1e8,
                           help='Only train the generator using the prediction network as a teacher when the prediction network loss is lower than this.')
        group.add_argument('--teach-threshold-gen', type=float, default=1e8,
                           help='Only train the predictor using the generator network as a teacher when the generator network loss is lower than this.')
        group.add_argument('--teach-threshold-combined', type=str2bool, default=True,
                           help='Only use teacher losses when both predictor and generator losses are below their thresholds.')

        group.add_argument('--w-zingg', type=float, default=1.0,
                           help='Weight for Zingg loss.')
        group.add_argument('--w-distances', type=float, default=1.0,
                           help='Weight for distances loss.')
        group.add_argument('--w-transformation', type=float, default=1.0,
                           help='Weight for transformation loss.')
        group.add_argument('--w-3d', type=float, default=1.0,
                           help='Weight for 3D loss.')
        group.add_argument('--w-3d-overshoot', type=float, default=0.,
                           help='Weight for overshoot component of the 3D loss (gets applied to the 3d loss before w_3d applied to the whole 3d loss).')
        group.add_argument('--w-3d-undershoot', type=float, default=0.,
                           help='Weight for undershoot component of the 3D loss (gets applied to the 3d loss before w_3d applied to the whole 3d loss).')
        group.add_argument('--w-material', type=float, default=1.0,
                           help='Weight for material loss.')
        group.add_argument('--w-light', type=float, default=1.0,
                           help='Weight for light loss.')
        group.add_argument('--w-generator', type=float, default=1.0,
                           help='Weight for generator loss.')
        group.add_argument('--w-net-teacher', type=float, default=1.0,
                           help='Weight for net teacher loss.')
        group.add_argument('--w-gen-teacher', type=float, default=1.0,
                           help='Weight for gen teacher loss.')
        group.add_argument('--w-discriminator', type=float, default=1.0,
                           help='Weight for discriminator loss.')
        group.add_argument('--w-rcf', type=float, default=0.,
                           help='Weight for the edge detection RCF loss for the generator.')
        group.add_argument('--w-transcoder-pred', type=float, default=1.0,
                           help='Weight for transcoder regularisation loss to include in the prediction network losses (if training it).')
        group.add_argument('--w-transcoder-gen', type=float, default=1.0,
                           help='Weight for transcoder regularisation loss to include in the generator network losses (if training it).')
        group.add_argument('--w-com-X', type=float, default=0.,
                           help='Weight for combined X loss - used in train_combined mode.')
        group.add_argument('--w-com-Y', type=float, default=0.,
                           help='Weight for combined Y loss - used in train_combined mode.')

        return group
