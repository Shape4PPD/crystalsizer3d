import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from parti_pytorch.vit_vqgan import ViTEncDec

from crystalsizer3d import logger
from crystalsizer3d.nn.models.basenet import BaseNet
from crystalsizer3d.util.utils import is_power_of_two


class ViTVAE(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],
            latent_size: int,

            oversample: bool = False,
            n_layers: int = 3,
            patch_size: int = 16,
            dim_head: int = 32,
            heads: int = 4,
            ff_mult: int = 4,

            img_mean: float = 0.5,
            img_std: float = 0.5,
            build_model: bool = True
    ):
        super().__init__(input_shape, output_shape)
        self.latent_size = latent_size

        self.oversample = oversample
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult

        # Load the image means and stds
        self.register_buffer('img_mean', torch.tensor(img_mean)[None, None, None, None])
        self.register_buffer('img_std', torch.tensor(img_std)[None, None, None, None])

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return f'vitvae/l={self.latent_size}'

    def _build_model(self):
        in_size = int(torch.prod(torch.tensor(self.input_shape)))
        out_size = self.output_shape

        # Generator requires image sizes to be powers of two
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

        # Then expand the latent vector to have spatial dimensions
        self.latent_spatial_size = image_size // self.patch_size

        # Build the ViT VAE (we're only using the decoder)
        self.enc_dec = ViTEncDec(
            dim=self.latent_size,
            image_size=image_size,
            channels=1,
            layers=self.n_layers,
            patch_size=self.patch_size,
            dim_head=self.dim_head,
            heads=self.heads,
            ff_mult=self.ff_mult,
        )

    def forward(self, x):
        bs = x.shape[0]

        # Expand parameters to latent space and input to decoder
        x = self.linear(x)
        x = x.reshape(bs, self.latent_size, 1, 1)
        x = x.expand(bs, self.latent_size, self.latent_spatial_size, self.latent_spatial_size)
        x = self.enc_dec.decode(x)

        # Resize to required output shape
        if x.shape[-1] != self.output_shape[-1]:
            x = F.interpolate(x, size=self.output_shape[-2:], mode='bilinear', align_corners=False)

        # De-normalise
        x = x * self.img_std + self.img_mean

        return x
