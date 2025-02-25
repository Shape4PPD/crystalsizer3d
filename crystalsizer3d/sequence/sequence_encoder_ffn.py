from typing import Any, Dict, List
import math

import torch
import torch.nn as nn
from torch import Tensor

from crystalsizer3d.nn.models.basenet import BaseNet
from crystalsizer3d.util.utils import init_tensor


class SequenceEncoderFFN(BaseNet):
    def __init__(
            self,
            scene_dict: Dict[str, Any],
            fixed_parameters: Dict[str, Tensor],
            stationary_parameters: List[str],
            hidden_dim: int,
            n_layers: int,
            dropout: float = 0.1,
            activation: str = 'gelu'
    ):
        from crystalsizer3d.sequence.sequence_fitter import PARAMETER_KEYS

        # Work out the parameter shapes
        self.stationary_parameters = {}
        self.nonstationary_param_dim = 0
        self.parameter_shapes = {}
        for k in PARAMETER_KEYS:
            if k in scene_dict:
                v = scene_dict[k]
            else:
                v = scene_dict['crystal'][k]
            n = len(v) if isinstance(v, (list, tuple)) else 1
            self.parameter_shapes[k] = n
            if k in stationary_parameters:
                self.stationary_parameters[k] = v
            elif k not in fixed_parameters:
                self.nonstationary_param_dim += n
        self.fixed_parameters = fixed_parameters
        self.scene_dict = scene_dict

        super().__init__(input_shape=(1,), output_shape=(self.nonstationary_param_dim,))

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

        # Feedforward network equivalent to the Transformer Encoder layers
        layers = []
        for _ in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim * 4))
            layers.append(self.activation_module)
            if self.dropout > 0:
                layers.append(self.dropout_module)
            layers.append(nn.Linear(self.hidden_dim * 4, self.hidden_dim))

        self.feedforward = nn.Sequential(*layers)
        self.output_projection = nn.Linear(self.hidden_dim, self.nonstationary_param_dim, bias=False)

    def _init_params(self):
        """
        Initialise the model parameters.
        """
        super()._init_params()  # Initialise the model parameters

        # Initialise the fixed and stationary parameters
        for k, v in self.fixed_parameters.items():
            self.register_buffer(k, init_tensor(v))
        for k, v in self.stationary_parameters.items():
            self.register_parameter(k, nn.Parameter(init_tensor(v), requires_grad=True))

    def forward(self, time_points: Tensor) -> Tensor:
        """
        Decode continuous time points into nonstationary parameters using a ffn encoder
        and combine with the fixed and stationary parameters.
        """
        bs = time_points.shape[0]

        # Reshape to hidden_dim (bs, hidden_dim)
        x = time_points[:, None].repeat(1, self.hidden_dim)

        # Raise the points to a range of exponents
        n_exponents = 4
        exponents = torch.arange(1, n_exponents + 1, dtype=x.dtype, device=x.device) \
            .repeat_interleave(self.hidden_dim // n_exponents)
        x = x.pow(exponents)

        # Pass through encoder
        x = self.feedforward(x)  # (bs, hidden_dim)
        x = self.output_projection(x)  # (bs, nonstationary_param_dim)

        # Combine with fixed and stationary parameters
        parameters = []
        i = 0
        for k, n in self.parameter_shapes.items():
            if k in self.fixed_parameters or k in self.stationary_parameters:
                p = getattr(self, k).unsqueeze(0)
                if p.ndim == 1:
                    p = p.unsqueeze(0)
                p = p.repeat(bs, 1)
            else:
                p = x[:, i:i + n]
                i += n
            parameters.append(p)
        parameters = torch.cat(parameters, dim=-1)

        return parameters
