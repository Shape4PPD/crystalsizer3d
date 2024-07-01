import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gigagan_pytorch import Generator as GigaganGenerator

from crystalsizer3d import logger
from crystalsizer3d.nn.models.basenet import BaseNet
from crystalsizer3d.util.utils import is_power_of_two


class Generator(GigaganGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_ = None

    @property
    def device(self):
        return self.device_


class GigaGAN(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],
            latent_size: int,
            oversample: bool = False,
            dim_capacity: int = 8,
            dim_max: int = 256,
            self_attn_resolutions: Tuple[int, ...] = (32, 16),
            self_attn_dim_head: int = 16,
            self_attn_heads: int = 4,
            self_attn_ff_mult: int = 4,
            cross_attn_resolutions: Tuple[int, ...] = (32, 16),
            cross_attn_dim_head: int = 16,
            cross_attn_heads: int = 4,
            cross_attn_ff_mult: int = 4,
            img_mean: float = 0.5,
            img_std: float = 0.5,
            build_model: bool = True
    ):
        super().__init__(input_shape, output_shape)
        self.latent_size = latent_size
        self.oversample = oversample
        self.dim_capacity = dim_capacity
        self.dim_max = dim_max
        self.self_attn_resolutions = self_attn_resolutions
        self.self_attn_dim_head = self_attn_dim_head
        self.self_attn_heads = self_attn_heads
        self.self_attn_ff_mult = self_attn_ff_mult
        self.cross_attn_resolutions = cross_attn_resolutions
        self.cross_attn_dim_head = cross_attn_dim_head
        self.cross_attn_heads = cross_attn_heads
        self.cross_attn_ff_mult = cross_attn_ff_mult

        # Load the image means and stds
        self.register_buffer('img_mean', torch.tensor(img_mean)[None, None, None, None])
        self.register_buffer('img_std', torch.tensor(img_std)[None, None, None, None])

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return f'gigagan/l={self.latent_size}'

    def _build_model(self):
        in_size = int(torch.prod(torch.tensor(self.input_shape)))
        out_size = self.output_shape

        # GigaGAN generator requires image sizes to be powers of two
        image_size = out_size[-1]
        if not is_power_of_two(image_size):
            if self.oversample:
                image_size = 2**int(math.ceil(math.log2(image_size)))
                logger.warning(f'Generator will output images of size {image_size} and '
                               f'bilinearly downsample to required output size ({out_size[-1]}).')
            else:
                image_size = 2**int(math.floor(math.log2(image_size)))
                logger.warning(f'Generator will output images of size {image_size} and '
                               f'bilinearly upsample to required output size ({out_size[-1]}).')

        # Start with a linear layer to expand the inputs to the latent vector
        self.linear = nn.Linear(in_size, self.latent_size)

        # Build generator
        self.generator = Generator(
            image_size=image_size,
            dim_capacity=self.dim_capacity,
            dim_max=self.dim_max,
            channels=self.output_shape[0],
            style_network_dim=self.latent_size,
            dim_latent=self.latent_size,
            self_attn_resolutions=tuple(self.self_attn_resolutions),
            self_attn_dim_head=self.self_attn_dim_head,
            self_attn_heads=self.self_attn_heads,
            self_attn_ff_mult=self.self_attn_ff_mult,
            cross_attn_resolutions=tuple(self.cross_attn_resolutions),
            cross_attn_dim_head=self.cross_attn_dim_head,
            cross_attn_heads=self.cross_attn_heads,
            cross_attn_ff_mult=self.cross_attn_ff_mult,
            num_conv_kernels=2,
            unconditional=True,
            pixel_shuffle_upsample=False
        )

    def _init_params(self):
        # Only need to initialise the linear layer, as the generator is initialised separately
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # Expand parameters to latent space and input to generator as "styles"
        x = self.linear(x)
        self.generator.device_ = x.device
        x = self.generator(styles=x)

        # Resize to required output shape
        if x.shape[-1] != self.output_shape[-1]:
            x = F.interpolate(x, size=self.output_shape[-2:], mode='bilinear', align_corners=False)

        # De-normalise
        x = x * self.img_std + self.img_mean

        return x
