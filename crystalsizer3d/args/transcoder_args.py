from argparse import ArgumentParser, _ArgumentGroup
from typing import List

from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.util.utils import str2bool

TC_LATENT_ACTIVATIONS = ['none', 'tanh', 'sigmoid', 'relu']
TC_TRAINERS = ['self', 'predictor', 'discriminator', 'both']
TC_MODEL_NAMES = ['mask_inv', 'vae', 'tvae']
TC_AE_VARIANTS = ['variational', 'denoising']


class TranscoderArgs(BaseArgs):
    def __init__(
            self,
            use_transcoder: bool = False,
            tc_latent_size: int = 128,
            tc_latent_activation: str = 'none',
            tc_trained_by: str = 'both',
            tc_model_name: str = 'mask_inv',
            tc_vae_layers: List[int] = [],
            tc_normalise_latents: bool = True,

            tc_ae_variant: str = 'variational',
            tc_dim_enc: int = 256,
            tc_dim_dec: int = 256,
            tc_num_heads: int = 16,
            tc_dim_feedforward: int = 1024,
            tc_num_layers: int = 5,
            tc_depth_enc: int = 6,
            tc_depth_dec: int = 5,
            tc_dropout_prob: float = 0.,
            tc_activation: str = 'gelu',

            tc_min_w_kl: float = 1e-3,
            tc_max_w_kl: float = 1e-1,
            tc_rec_threshold: float = 0.1,
            tc_w_l1: float = 0.,
            tc_w_l2: float = 0.,
            tc_w_indep: float = 0.,

            **kwargs
    ):
        self.use_transcoder = use_transcoder
        self.tc_latent_size = tc_latent_size
        assert tc_latent_activation in TC_LATENT_ACTIVATIONS
        self.tc_latent_activation = tc_latent_activation
        assert tc_trained_by in TC_TRAINERS
        self.tc_trained_by = tc_trained_by
        assert tc_model_name in TC_MODEL_NAMES
        if self.use_transcoder:
            if tc_model_name in ['vae', 'tvae']:
                assert tc_trained_by == 'self'
            else:
                assert tc_trained_by != 'self', 'Only the VAE transcoder can be trained by itself.'
        self.tc_model_name = tc_model_name
        self.tc_vae_layers = tc_vae_layers
        self.tc_normalise_latents = tc_normalise_latents

        # Transformer autoencoder specific
        assert tc_ae_variant in TC_AE_VARIANTS
        self.tc_ae_variant = tc_ae_variant
        self.tc_dim_enc = tc_dim_enc
        self.tc_dim_dec = tc_dim_dec
        self.tc_num_heads = tc_num_heads
        self.tc_dim_feedforward = tc_dim_feedforward
        self.tc_num_layers = tc_num_layers
        self.tc_depth_enc = tc_depth_enc
        self.tc_depth_dec = tc_depth_dec
        self.tc_dropout_prob = tc_dropout_prob
        self.tc_activation = tc_activation

        # Regularisation terms
        self.tc_min_w_kl = tc_min_w_kl
        self.tc_max_w_kl = tc_max_w_kl
        self.tc_rec_threshold = tc_rec_threshold
        self.tc_w_l1 = tc_w_l1
        self.tc_w_l2 = tc_w_l2
        self.tc_w_indep = tc_w_indep

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Transcoder Args')
        group.add_argument('--use-transcoder', type=str2bool, default=False,
                           help='Use the transcoder module to interface between the parameters and latent spaces.')
        group.add_argument('--tc-latent-size', type=int, default=128,
                           help='Size of the latent vector.')
        group.add_argument('--tc-latent-activation', type=str, default='none', choices=TC_LATENT_ACTIVATIONS,
                           help='Activation function to use on the latent vector.')
        group.add_argument('--tc-trained-by', type=str, default='self', choices=TC_TRAINERS,
                           help='Which optimiser should train the transcoder parameters.')
        group.add_argument('--tc-model-name', type=str, default='mask_inv', choices=TC_MODEL_NAMES,
                           help='Name of the transcoder model to use.')
        parser.add_argument(f'--tc-vae-layers', type=lambda s: [int(item) for item in s.split(',')],
                            default=[],
                            help='Comma delimited list of hidden layer sizes to use in the autoencoder.')
        group.add_argument('--tc-normalise-latents', type=str2bool, default=True,
                           help='Normalise the latent vectors before/after mapping.')

        # Transformer autoencoder specific
        group.add_argument('--tc-ae-variant', type=str, default='variational', choices=TC_AE_VARIANTS,
                           help='Which variant of the transformer autoencoder to use.')
        group.add_argument('--tc-dim-enc', type=int, default=256,
                           help='Size of the transformer representation in the encoder.')
        group.add_argument('--tc-dim-dec', type=int, default=256,
                           help='Size of the transformer representation in the decoder.')
        group.add_argument('--tc-num-heads', type=int, default=16,
                           help='Number of attention heads in the transformer.')
        group.add_argument('--tc-dim-feedforward', type=int, default=1024,
                           help='Size of the feedforward layer in the transformer.')
        group.add_argument('--tc-num-layers', type=int, default=5,
                           help='Number of layers in the transformer.')
        group.add_argument('--tc-depth-enc', type=int, default=6,
                           help='Depth of the encoder.')
        group.add_argument('--tc-depth-dec', type=int, default=5,
                           help='Depth of the decoder.')
        group.add_argument('--tc-dropout-prob', type=float, default=0.,
                           help='Dropout probability in the transformer.')
        group.add_argument('--tc-activation', type=str, default='gelu', choices=['gelu', 'relu'],
                           help='Activation function to use in the transformer.')

        # Regularisation terms
        group.add_argument('--tc-min-w-kl', type=float, default=1e-3,
                           help='Minimum weight of the KL divergence term.')
        group.add_argument('--tc-max-w-kl', type=float, default=1e-1,
                           help='Maximum weight of the KL divergence term.')
        group.add_argument('--tc-rec-threshold', type=float, default=0.1,
                           help='Threshold for the reconstruction loss before the KL weighting increases from the min.')
        group.add_argument('--tc-w-l1', type=float, default=0.,
                           help='Weight of the L1 regularisation term applied to the latent vector.')
        group.add_argument('--tc-w-l2', type=float, default=0.,
                           help='Weight of the L2 regularisation term applied to the latent vector.')
        group.add_argument('--tc-w-indep', type=float, default=0.,
                           help='Weight of the independence regularisation term applied to the latent vector.')

        return group
