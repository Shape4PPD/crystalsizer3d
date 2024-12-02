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
        else:
            raise ValueError(f'Invalid activation function: {self.activation}')
        self.model = nn.Sequential()
        n_in = 1
        for i in range(self.n_layers):
            self.model.add_module(f'HiddenLayer{i}', nn.Linear(n_in, self.hidden_dim))
            self.model.add_module(f'Activation{i}', act)
            if self.dropout > 0:
                self.model.add_module(f'Dropout{i}', nn.Dropout(self.dropout))
            n_in = self.hidden_dim
        self.model.add_module('OutputLayer', nn.Linear(n_in, self.param_dim))

    def forward(self, time_points: Tensor):
        """
        Encode continuous time points with Fourier features and pass through transformer encoder.
        """
        if time_points.ndim == 1:
            time_points = time_points[:, None]

        # Pass through encoder
        x = self.model(time_points)

        return x
