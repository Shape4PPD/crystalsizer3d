import math

import torch
import torch.nn as nn
from torch import Tensor

from crystalsizer3d.nn.models.basenet import BaseNet


class SequenceEncoderFFN(BaseNet):
    def __init__(
            self,
            param_dim: int,
            hidden_dim: int,
            n_layers: int,
            dropout: float = 0.1,
            activation: str = 'gelu'
    ):
        super().__init__(input_shape=(1,), output_shape=(param_dim,))

        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation

        # Build model
        self._build_model()
        self._init_params()

    def _build_model(self):
        """
        Build the feedforward network encoder model.
        """
        if self.activation == 'gelu':
            act = nn.GELU()
        elif self.activation == 'relu':
            act = nn.ReLU()
        elif self.activation == 'sigmoid':
            act = nn.Sigmoid()
        elif self.activation == 'tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f'Invalid activation function: {self.activation}')
        self.activation_module = act
        self.dropout_module = nn.Dropout(self.dropout) if self.dropout > 0 else None

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            # Each layer takes the current hidden state + time input
            self.layers.append(nn.Linear(self.hidden_dim + 1 if i > 0 else 1, self.hidden_dim))
        self.output_layer = nn.Linear(self.hidden_dim + 1, self.param_dim)

    def forward(self, time_points: Tensor):
        """
        Encode continuous time points using a feedforward network.
        """
        if time_points.ndim == 1:
            time_points = time_points[:, None]

        # Start with the time input and pass through the layers
        x = time_points
        for i in range(self.n_layers):
            x = self.layers[i](x)
            x = self.activation_module(x)
            if self.dropout > 0:
                x = self.dropout_module(x)

            # Concatenate the time input to the current hidden state
            x = torch.cat([time_points, x], dim=-1)

        # Final output layer
        return self.output_layer(x)
