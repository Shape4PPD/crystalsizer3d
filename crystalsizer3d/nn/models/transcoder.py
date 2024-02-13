from abc import ABC, abstractmethod

import torch
from torch import nn

from crystalsizer3d.nn.models.basenet import BaseNet


class Transcoder(BaseNet, ABC):
    latents_in: torch.Tensor
    latents_out: torch.Tensor

    def __init__(
            self,
            latent_size: int,
            param_size: int,
            latent_activation: str = 'none'
    ):
        super().__init__((latent_size,), (param_size,))
        self.latent_size = latent_size
        self.param_size = param_size
        self.register_buffer('latents_in', None)
        self.register_buffer('latents_out', None)
        self.latent_activation = latent_activation
        if self.latent_activation == 'none':
            self.latent_activation_fn = nn.Identity()
        elif self.latent_activation == 'tanh':
            self.latent_activation_fn = nn.Tanh()
        elif self.latent_activation == 'sigmoid':
            self.latent_activation_fn = nn.Sigmoid()
        elif self.latent_activation == 'relu':
            self.latent_activation_fn = nn.ReLU()
        else:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor, to_which: str = 'parameters', **kwargs):
        if to_which == 'parameters':
            return self.to_parameters(x, **kwargs)
        elif to_which == 'latents':
            return self.to_latents(x, **kwargs)
        raise NotImplementedError()

    @abstractmethod
    def to_parameters(self, z: torch.Tensor, **kwargs):
        """
        Convert a latent vector to a parameter vector.
        """
        pass

    @abstractmethod
    def to_latents(self, p: torch.Tensor, activate: bool = True, **kwargs):
        """
        Convert a parameter vector to a latent vector.
        """
        pass
