from typing import List

import torch
import torch.nn as nn

from crystalsizer3d.nn.models.fcnet import FCLayer
from crystalsizer3d.nn.models.transcoder import Transcoder


class TranscoderVAE(Transcoder):
    def __init__(
            self,
            latent_size: int,
            param_size: int,
            latent_activation: str = 'none',
            layers_config: List[int] = [],
    ):
        super().__init__(latent_size, param_size, latent_activation)
        self.layers_config = layers_config
        self._build_model()

    def _build_model(self):
        """
        Initialise the encoder and decoder networks.
        """

        # Build the encoder
        size = self.param_size
        self.encoder = nn.Sequential()
        layers = self.layers_config
        for i, n in enumerate(layers):
            self.encoder.add_module(
                f'Layer{i}',
                FCLayer(size, n, activation=i != 0)
            )
            size = n
        self.latents_mu = FCLayer(size, self.latent_size, activation=True)
        self.latents_logvar = FCLayer(size, self.latent_size, activation=True)

        # Build the decoder
        size = self.latent_size
        self.decoder = nn.Sequential()
        layers = self.layers_config[::-1]
        for i, n in enumerate(layers):
            self.decoder.add_module(
                f'Layer{i}',
                FCLayer(size, n, activation=i != 0)
            )
            size = n
        self.reconst_mu = FCLayer(size, self.param_size, activation=True)
        self.reconst_logvar = FCLayer(size, self.param_size, activation=True)

    def to_latents(self, p: torch.Tensor, activate: bool = True, return_logvar: bool = False):
        """
        Convert a parameter vector to a latent vector.
        """
        Z = self.encoder(p)
        Z_mu = self.latents_mu(Z)
        if activate:
            Z_mu = self.latent_activation_fn(Z_mu)
        self.latents_out = Z_mu
        if return_logvar:
            z_logvar = self.latents_logvar(Z)
            return Z_mu, z_logvar
        return Z_mu

    def to_parameters(self, z: torch.Tensor, return_logvar: bool = False):
        """
        Convert a latent vector to a parameter vector.
        """
        z = self.latent_activation_fn(z)  # Assume input z is logits
        self.latents_in = z
        p = self.decoder(z)
        p_mu = self.reconst_mu(p)
        if return_logvar:
            p_logvar = self.reconst_logvar(p)
            return p_mu, p_logvar
        return p_mu

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterise the latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
