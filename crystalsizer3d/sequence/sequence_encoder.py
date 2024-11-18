import math

import torch
import torch.nn as nn
from torch import Tensor

from crystalsizer3d.nn.models.basenet import BaseNet


class SequenceEncoder(BaseNet):
    def __init__(
            self,
            param_dim: int,
            hidden_dim: int,
            n_layers: int,
            n_heads: int,
            max_freq: float,
            dropout: float = 0.1,
            activation: str = 'gelu'
    ):
        super().__init__(input_shape=(1,), output_shape=(param_dim,))

        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_freq = max_freq
        self.dropout = dropout
        self.activation = activation

        # Define frequency bands for Fourier features
        self.n_fourier_features = hidden_dim // 2  # Number of sin/cos pairs
        self.bands = 2**torch.linspace(0, math.log2(self.max_freq), self.n_fourier_features)

        # Build model
        self._build_model()
        self._init_params()

    def _build_model(self):
        """
        Build the transformer encoder model.
        """
        # Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                activation=self.activation,
            ),
            num_layers=self.n_layers
        )

        # Linear Projection Layers
        self.output_projection = nn.Linear(self.hidden_dim, self.param_dim)

    def fourier_encode(self, time_points: Tensor):
        """
        Apply Fourier encoding to continuous time points.
        """
        time_points = time_points[:, None] * self.bands.to(
            time_points.device)  # Broadcasting time points across frequencies
        sin_features = torch.sin(time_points)
        cos_features = torch.cos(time_points)
        return torch.cat([sin_features, cos_features], dim=-1)  # Concatenate sin and cos features

    def forward(self, time_points: Tensor):
        """
        Encode continuous time points with Fourier features and pass through transformer encoder.
        """
        assert time_points.ndim == 1, 'Time points must be 1D tensor.'
        assert time_points.amin() >= 0 and time_points.amax() <= 1, 'Time points must be normalised to [0, 1].'
        fourier_encoded = self.fourier_encode(time_points)  # Shape: (batch_size, hidden_dim)

        # Reshape for transformer (1, batch_size, hidden_dim)
        x = fourier_encoded[None, ...]

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (1, batch_size, hidden_dim)
        x = x[0]

        # Project output to final parameter dimension
        x = self.output_projection(x)

        return x
