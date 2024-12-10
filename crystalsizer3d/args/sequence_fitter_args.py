from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import List, Optional

from crystalsizer3d.args.base_args import BaseArgs


class SequenceFitterArgs(BaseArgs):
    def __init__(
            self,

            # Target sequence
            images_dir: Path | str,
            image_ext: str = 'jpg',
            start_image: int = 0,
            end_image: int = -1,
            initial_scene: Path | str | None = None,
            fix_parameters: List[str] | None = None,
            stationary_parameters: List[str] | None = None,

            # Sequence encoder model parameters
            seq_encoder_model: str = 'transformer',
            hidden_dim: int = 256,
            n_layers: int = 4,
            n_heads: int = 8,
            n_exponents: int = 1,
            dropout: float = 0.1,
            activation: str = 'gelu',

            # Adaptive sampler parameters
            ema_decay: float = 0.9,
            ema_val_init: float = 1.0,

            # Pretraining settings
            n_pretraining_steps: int = 1000,
            pretrain_batch_size: int = 32,
            lr_pretrain: float = 1e-3,

            # Optimisation settings
            seed: Optional[int] = None,
            max_steps: int = 1000,
            batch_size: int = 1,
            opt_algorithm: str = 'sgd',
            clip_grad_norm: float = 0.0,
            weight_decay: float = 0.0,
            lr_init: float = 1e-3,
            lr_min: float = 1e-6,
            lr_patience_steps: int = 10,
            lr_decay_rate: float = 0.1,
            w_negative_growth: float = 0.0,

            # Evaluation settings
            eval_batch_size: int = 32,

            **kwargs
    ):
        # Convert string paths to Path objects
        if isinstance(images_dir, str):
            images_dir = Path(images_dir)
        if isinstance(initial_scene, str):
            initial_scene = Path(initial_scene)

        # Target sequence
        self.images_dir = images_dir
        self.image_ext = image_ext
        self.start_image = start_image
        self.end_image = end_image
        if initial_scene is not None:
            assert initial_scene.exists(), f'Initial scene file {initial_scene} does not exist.'
        self.initial_scene = initial_scene
        if fix_parameters is not None:
            assert initial_scene is not None, 'Must provide an initial scene file if fixing parameters.'
            from crystalsizer3d.sequence.sequence_fitter import PARAMETER_KEYS
            assert all(param in PARAMETER_KEYS for param in fix_parameters), \
                f'Invalid parameter key: {fix_parameters}.'
        self.fix_parameters = fix_parameters
        if stationary_parameters is not None:
            from crystalsizer3d.sequence.sequence_fitter import PARAMETER_KEYS
            assert all(param in PARAMETER_KEYS for param in stationary_parameters), \
                f'Invalid parameter key: {stationary_parameters}.'
        self.stationary_parameters = stationary_parameters

        # Sequence encoder model parameters
        self.seq_encoder_model = seq_encoder_model
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_exponents = n_exponents
        self.dropout = dropout
        self.activation = activation

        # Adaptive sampler parameters
        self.ema_decay = ema_decay
        self.ema_val_init = ema_val_init

        # Pretraining settings
        self.n_pretraining_steps = n_pretraining_steps
        self.pretrain_batch_size = pretrain_batch_size
        self.lr_pretrain = lr_pretrain

        # Optimisation settings
        self.seed = seed
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.opt_algorithm = opt_algorithm
        self.clip_grad_norm = clip_grad_norm
        self.weight_decay = weight_decay
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.lr_patience_steps = lr_patience_steps
        self.lr_decay_rate = lr_decay_rate
        self.w_negative_growth = w_negative_growth

        # Evaluation settings
        self.eval_batch_size = eval_batch_size

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Sequence Encoder Args')

        # Target sequence
        parser.add_argument('--images-dir', type=Path,
                            help='Directory containing the sequence of images.')
        parser.add_argument('--image-ext', type=str, default='jpg',
                            help='Image extension.')
        parser.add_argument('--start-image', type=int, default=0,
                            help='Start processing from this image.')
        parser.add_argument('--end-image', type=int, default=-1,
                            help='End processing at this image.')
        parser.add_argument('--initial-scene', type=Path,
                            help='Path to the initial scene file. Will be used for any fixed parameters.')
        parser.add_argument('--fix-parameters', type=lambda s: s.split(','), default=[],
                            help='Fix these parameters to the values set in the initial scene file.')
        parser.add_argument('--stationary-parameters', type=lambda s: s.split(','), default=[],
                            help='Stationary parameters that do not change over time but still need to be trained.')

        # Sequence encoder model parameters
        group.add_argument('--seq-encoder-model', type=str, default='transformer', choices=['transformer', 'ffn'],
                           help='Sequence encoder model to use.')
        group.add_argument('--hidden-dim', type=int, default=256,
                           help='Hidden dimension of the transformer encoder.')
        group.add_argument('--n-layers', type=int, default=4,
                           help='Number of transformer encoder layers.')
        group.add_argument('--n-heads', type=int, default=8,
                           help='Number of attention heads in the transformer encoder.')
        group.add_argument('--n-exponents', type=int, default=1,
                           help='Number of exponents to apply to the input time point when expanding it '
                                'across the hidden dim for input to the transformer.')
        group.add_argument('--dropout', type=float, default=0.1,
                           help='Dropout probability in the transformer encoder.')
        group.add_argument('--activation', type=str, default='gelu',
                           help='Activation function in the transformer encoder.')

        # Adaptive sampler parameters
        group.add_argument('--ema-decay', type=float, default=0.9,
                           help='Exponential moving average decay for the adaptive sampler.')
        group.add_argument('--ema-val-init', type=float, default=1.0,
                           help='Initial values for the exponential moving averages of the adaptive sampler.')

        # Pretraining settings
        group.add_argument('--n-pretraining-steps', type=int, default=1000,
                           help='Number of pretraining steps.')
        group.add_argument('--pretrain-batch-size', type=int, default=32,
                           help='Batch size for pretraining.')
        group.add_argument('--lr-pretrain', type=float, default=1e-3,
                           help='Learning rate for pretraining.')

        # Optimisation settings
        group.add_argument('--seed', type=int,
                           help='Seed for the random number generator.')
        group.add_argument('--max-steps', type=int, default=5000,
                           help='Maximum number of refinement steps.')
        group.add_argument('--batch-size', type=int, default=10,
                           help='Number of gradient accumulation steps.')
        group.add_argument('--opt-algorithm', type=str, default='adamw',
                           help='Optimisation algorithm to use.')
        group.add_argument('--clip-grad-norm', type=float, default=1.,
                           help='Clip the gradient norm to this value.')
        group.add_argument('--weight-decay', type=float, default=0.0,
                           help='Weight decay for the optimiser.')
        group.add_argument('--lr-init', type=float, default=1e-3,
                           help='Initial learning rate.')
        group.add_argument('--lr-min', type=float, default=1e-6,
                           help='Minimum learning rate.')
        group.add_argument('--lr-patience-steps', type=int, default=10,
                           help='Number of steps before learning rate patience.')
        group.add_argument('--lr-decay-rate', type=float, default=0.5,
                           help='Learning rate decay rate.')
        group.add_argument('--w-negative-growth', type=float, default=0.,
                           help='Regularisation weight to penalise negative growth of the crystal distances across a batch.')

        # Evaluation settings
        group.add_argument('--eval-batch-size', type=int, default=32,
                           help='Batch size for evaluation.')

        return group
