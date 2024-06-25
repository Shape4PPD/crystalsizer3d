from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from crystalsizer3d.nn.models.basenet import BaseNet


class GeneratorNet(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],
            latent_size: int,
            n_blocks: int = 4,
            n_base_filters: int = 64,
            img_mean: float = 0.5,
            img_std: float = 0.5,
            build_model: bool = True
    ):
        super().__init__(input_shape, output_shape)

        self.latent_size = latent_size
        self.n_blocks = n_blocks
        self.n_base_filters = n_base_filters

        # Load the image means and stds
        self.register_buffer('img_mean', torch.tensor(img_mean)[None, None, None, None])
        self.register_buffer('img_std', torch.tensor(img_std)[None, None, None, None])

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return f'generator/l={self.latent_size}_b={self.n_blocks}_f={self.n_base_filters}'

    def _build_model(self):
        in_size = int(torch.prod(torch.tensor(self.input_shape)))
        out_size = self.output_shape

        # Start with a linear layer to expand the inputs to the latent vector
        self.linear = nn.Linear(in_size, self.latent_size)

        # Then upsample to the required image size using transposed convolutions
        generator = nn.Sequential()
        in_channels = self.latent_size
        for i in range(self.n_blocks - 1):
            out_channels = self.n_base_filters * 2**(self.n_blocks - 2 - i)
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=1 if i == 0 else 2,
                    padding=0 if i == 0 else 1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
            generator.append(block)
            in_channels = out_channels

        # Add the final block
        block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_size[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            # Activation?
        )
        generator.append(block)
        self.generator = generator

    def forward(self, x):
        bs = x.shape[0]

        # Expand parameters to latent space
        x = self.linear(x)
        x = x.reshape(bs, self.latent_size, 1, 1)

        # Generate image using transposed convolution layers
        x = self.generator(x)

        # Resize to required output shape
        if x.shape[-1] != self.output_shape[-1]:
            x = F.interpolate(x, size=self.output_shape[-2:], mode='bilinear', align_corners=False)

        # De-normalise
        x = x * self.img_std + self.img_mean

        return x
