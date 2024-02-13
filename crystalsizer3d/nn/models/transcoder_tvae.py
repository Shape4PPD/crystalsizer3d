import torch
import torch.nn as nn

from crystalsizer3d.nn.models.transcoder import Transcoder


class TranscoderTVAE(Transcoder):
    def __init__(
            self,
            latent_size: int,
            param_size: int,
            latent_activation: str = 'none',
            dim_enc: int = 128,
            dim_dec: int = 128,
            num_heads: int = 4,
            dim_feedforward: int = 512,
            num_layers: int = 3,
            depth_enc: int = 3,
            depth_dec: int = 3,
            dropout: float = 0.,
            activation: str = 'gelu'
    ):
        super().__init__(latent_size, param_size, latent_activation)
        self.dim_enc = dim_enc
        self.dim_dec = dim_dec
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.depth_enc = depth_enc
        self.depth_dec = depth_dec
        self.dropout = dropout
        self.activation = activation
        self._build_model()

    def _build_model(self):
        """
        Initialise the encoder and decoder networks.
        """
        layer_params = dict(
            batch_first=True,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation
        )

        # Build the encoder
        self.exp_enc = nn.Linear(self.param_size, self.depth_enc * self.dim_enc)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.dim_enc, **layer_params),
            num_layers=self.num_layers
        )
        self.latents_mu = nn.Linear(self.dim_enc * self.depth_enc, self.latent_size)
        self.latents_logvar = nn.Linear(self.dim_enc * self.depth_enc, self.latent_size)

        # Build the decoder
        self.exp_dec = nn.Linear(self.latent_size, self.depth_dec * self.dim_dec)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.dim_dec, **layer_params),
            num_layers=self.num_layers
        )
        self.reconst_mu = nn.Linear(self.dim_dec * self.depth_dec, self.param_size)
        self.reconst_logvar = nn.Linear(self.dim_dec * self.depth_dec, self.param_size)

    def to_latents(self, p: torch.Tensor, activate: bool = True, return_logvar: bool = False):
        """
        Convert a parameter vector to a latent vector.
        """
        bs = p.shape[0]
        p = self.exp_enc(p).reshape(bs, self.depth_enc, -1)
        Z = self.encoder(p).reshape(bs, -1)
        Z_mu = self.latents_mu(Z)
        if activate:
            Z_mu = self.latent_activation_fn(Z_mu)
        self.latents_out = Z_mu
        if return_logvar:
            Z_logvar = self.latents_logvar(Z)
            return Z_mu, Z_logvar
        return Z_mu

    def to_parameters(self, z: torch.Tensor, return_logvar: bool = False):
        """
        Convert a latent vector to a parameter vector.
        """
        z = self.latent_activation_fn(z)  # Assume input z is logits
        self.latents_in = z
        bs = z.shape[0]
        Z = self.exp_dec(z).reshape(bs, self.depth_dec, -1)
        p = self.decoder(Z, torch.zeros_like(Z)).reshape(bs, -1)
        Y_mu = self.reconst_mu(p)
        if return_logvar:
            Y_logvar = self.reconst_logvar(p)
            return Y_mu, Y_logvar
        return Y_mu

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterise the latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
