from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from crystalsizer3d.nn.models.basenet import BaseNet
from crystalsizer3d.util.utils import init_tensor


class SequenceEncoder(BaseNet):
    def __init__(
            self,
            scene_dict: Dict[str, Any],
            fixed_parameters: Dict[str, Tensor],
            stationary_parameters: List[str],
            hidden_dim: int,
            n_layers: int,
            n_heads: int,
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
        self.n_heads = n_heads
        self.dropout = dropout
        self.activation = activation

        # Build model
        self._build_model()
        self._init_params()

    def _build_model(self):
        """
        Build the transformer encoder model that generates the nonstationary parameters.
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
        Decode continuous time points into nonstationary parameters using a transformer encoder
        and combine with the fixed and stationary parameters.
        """
        bs = time_points.shape[0]

        # Reshape for transformer (1, bs, hidden_dim)
        x = time_points[None, :, None].repeat(1, 1, self.hidden_dim)

        # Raise the points to a range of exponents
        n_exponents = 4
        exponents = torch.arange(1, n_exponents + 1, dtype=x.dtype, device=x.device) \
            .repeat_interleave(self.hidden_dim // n_exponents)
        x = x.pow(exponents)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (1, bs, hidden_dim)
        x = self.output_projection(x[0])  # (bs, nonstationary_param_dim)

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
