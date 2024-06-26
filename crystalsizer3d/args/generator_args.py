from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import Tuple, Union

from crystalsizer3d import ROOT_PATH
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.util.utils import str2bool

GEN_IMAGE_LOSS_CHOICES = ['l1', 'l2']


class GeneratorArgs(BaseArgs):
    def __init__(
            self,
            gen_input_noise_std: float = 0.,
            gen_model_name: str = 'dcgan',
            gen_latent_size: int = 100,
            gen_image_loss: str = 'l2',

            # DCGAN args
            dcgan_n_blocks: int = 4,
            dcgan_n_base_filters: int = 64,

            # GigaGAN args
            ggan_oversample: bool = False,
            ggan_dim_capacity: int = 16,
            ggan_dim_max: int = 512,
            ggan_self_attn_resolutions: Tuple[int, ...] = (32, 16),
            ggan_self_attn_dim_head: int = 32,
            ggan_self_attn_heads: int = 4,
            ggan_self_attn_ff_mult: int = 4,
            ggan_cross_attn_resolutions: Tuple[int, ...] = (32, 16),
            ggan_cross_attn_dim_head: int = 32,
            ggan_cross_attn_heads: int = 4,
            ggan_cross_attn_ff_mult: int = 4,

            # ViTVAE args
            vitvae_oversample: bool = False,
            vitvae_n_layers: int = 3,
            vitvae_patch_size: int = 16,
            vitvae_dim_head: int = 32,
            vitvae_heads: int = 4,
            vitvae_ff_mult: int = 4,

            # Discriminator args
            use_discriminator: bool = False,
            disc_n_base_filters: int = 32,
            disc_n_layers: int = 4,

            # RCF model args
            use_rcf: bool = False,
            rcf_model_path: Union[str, Path] = ROOT_PATH / 'data' / 'bsds500_pascal_model.pth',
            rcf_loss_type: str = 'l2',

            **kwargs
    ):
        self.gen_input_noise_std = gen_input_noise_std
        self.gen_model_name = gen_model_name
        self.gen_latent_size = gen_latent_size
        assert gen_image_loss in GEN_IMAGE_LOSS_CHOICES, \
            f'Invalid image loss {gen_image_loss}, must be one of {GEN_IMAGE_LOSS_CHOICES}'
        self.gen_image_loss = gen_image_loss

        # DCGAN args
        self.dcgan_n_blocks = dcgan_n_blocks
        self.dcgan_n_base_filters = dcgan_n_base_filters

        # GigaGAN args
        self.ggan_oversample = ggan_oversample
        self.ggan_dim_capacity = ggan_dim_capacity
        self.ggan_dim_max = ggan_dim_max
        self.ggan_self_attn_resolutions = ggan_self_attn_resolutions
        self.ggan_self_attn_dim_head = ggan_self_attn_dim_head
        self.ggan_self_attn_heads = ggan_self_attn_heads
        self.ggan_self_attn_ff_mult = ggan_self_attn_ff_mult
        self.ggan_cross_attn_resolutions = ggan_cross_attn_resolutions
        self.ggan_cross_attn_dim_head = ggan_cross_attn_dim_head
        self.ggan_cross_attn_heads = ggan_cross_attn_heads
        self.ggan_cross_attn_ff_mult = ggan_cross_attn_ff_mult

        # ViTVAE args
        self.vitvae_oversample = vitvae_oversample
        self.vitvae_n_layers = vitvae_n_layers
        self.vitvae_patch_size = vitvae_patch_size
        self.vitvae_dim_head = vitvae_dim_head
        self.vitvae_heads = vitvae_heads
        self.vitvae_ff_mult = vitvae_ff_mult

        # Discriminator args
        self.use_discriminator = use_discriminator
        self.disc_n_base_filters = disc_n_base_filters
        self.disc_n_layers = disc_n_layers

        # RCF model args
        self.use_rcf = use_rcf
        if isinstance(rcf_model_path, str):
            rcf_model_path = Path(rcf_model_path)
        if use_rcf:
            assert rcf_model_path.exists(), f'RCF model path {rcf_model_path} does not exist.'
        self.rcf_model_path = rcf_model_path
        assert rcf_loss_type in ['l1', 'l2'], f'Invalid RCF loss type {rcf_loss_type}, must be one of [l1, l2]'
        self.rcf_loss_type = rcf_loss_type

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Generator Args')
        group.add_argument('--gen-input-noise-std', type=float, default=0.,
                           help='Standard deviation of the input noise.')
        group.add_argument('--gen-model-name', type=str, default='dcgan',
                           help='Name of the model to use.')
        group.add_argument('--gen-latent-size', type=int, default=100,
                           help='Size of the latent vector.')
        group.add_argument('--gen-image-loss', type=str, default='l2', choices=GEN_IMAGE_LOSS_CHOICES,
                           help='Loss function to use for the image reconstruction.')

        # DCGAN args
        group.add_argument('--dcgan-n-blocks', type=int, default=4,
                           help='Number of up-scaling blocks in the DCGAN generator.')
        group.add_argument('--dcgan-n-base-filters', type=int, default=64,
                           help='Number of base filters in the final block of the DCGAN generator, doubles for each previous block.')

        # GigaGAN args
        group.add_argument('--ggan-oversample', type=str2bool, default=False,
                           help='If output image size is not multiple of two, either generate smaller image and upsample (default) '
                                'or a larger image and downsample (oversample=False).')
        group.add_argument('--ggan-dim-capacity', type=int, default=16,
                           help='Number of base filters that are scaled up in each layer of the GigaGAN network.')
        group.add_argument('--ggan-dim-max', type=int, default=512,
                           help='Maximum number of filters in the GigaGAN network.')
        group.add_argument('--ggan-self-attn-resolutions', type=lambda s: [int(item) for item in s.split(',')],
                           default='32,16', help='Resolutions to apply self-attention at.')
        group.add_argument('--ggan-self-attn-dim-head', type=int, default=32,
                           help='Dimension of the attention heads in the self-attention layers.')
        group.add_argument('--ggan-self-attn-heads', type=int, default=4,
                           help='Number of attention heads in the self-attention layers.')
        group.add_argument('--ggan-self-attn-ff-mult', type=int, default=4,
                           help='Multiplier for the hidden dimension in the self-attention layers.')
        group.add_argument('--ggan-cross-attn-resolutions', type=lambda s: [int(item) for item in s.split(',')],
                           default='32,16', help='Resolutions to apply cross-attention at.')
        group.add_argument('--ggan-cross-attn-dim-head', type=int, default=32,
                           help='Dimension of the attention heads in the cross-attention layers.')
        group.add_argument('--ggan-cross-attn-heads', type=int, default=4,
                           help='Number of attention heads in the cross-attention layers.')
        group.add_argument('--ggan-cross-attn-ff-mult', type=int, default=4,
                           help='Multiplier for the hidden dimension in the cross-attention layers.')

        # ViTVAE args
        group.add_argument('--vitvae-oversample', type=str2bool, default=False,
                           help='If output image size is not multiple of two, either generate smaller image and upsample (default) '
                                'or a larger image and downsample (oversample=False).')
        group.add_argument('--vitvae-n-layers', type=int, default=3,
                           help='Number of layers in the ViTVAE generator.')
        group.add_argument('--vitvae-patch-size', type=int, default=16,
                           help='Patch size for the ViTVAE generator.')
        group.add_argument('--vitvae-dim-head', type=int, default=32,
                           help='Dimension of the attention heads in the ViTVAE generator.')
        group.add_argument('--vitvae-heads', type=int, default=4,
                           help='Number of attention heads in the ViTVAE generator.')
        group.add_argument('--vitvae-ff-mult', type=int, default=4,
                           help='Multiplier for the hidden dimension in the ViTVAE generator.')

        # Discriminator args
        group.add_argument('--use-discriminator', type=str2bool, default=False,
                           help='Build and train a discriminator alongside the generator.')
        group.add_argument('--disc-n-base-filters', type=int, default=32,
                           help='Number of base filters in the first block of the discriminator, doubles for each subsequent block.')
        group.add_argument('--disc-n-layers', type=int, default=4,
                           help='Number of downsampling, residual convolution blocks in the discriminator.')

        # RCF model args
        group.add_argument('--use-rcf', type=str2bool, default=False,
                           help='Use the Richer Convolutional Features model for edge detection.')
        group.add_argument('--rcf-model-path', type=Path, default=ROOT_PATH / 'data' / 'bsds500_pascal_model.pth',
                           help='Path to the RCF model checkpoint.')
        group.add_argument('--rcf-loss-type', type=str, default='l2', choices=['l1', 'l2'],
                           help='Loss function to use for the RCF features comparison.')

        return group
