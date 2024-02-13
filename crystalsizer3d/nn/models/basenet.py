from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class BaseNet(nn.Module, ABC):
    latent_state: torch.Tensor

    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model: nn.Module = None
        self.register_buffer('latent_state', None)
        self.last_op: nn.Module = None

    @abstractmethod
    def _build_model(self):
        pass

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def get_n_params(self) -> int:
        return sum([p.data.nelement() for p in self.parameters()])

    @torch.jit.ignore
    def calc_norms(self, p: int = 2) -> Dict[str, float]:
        p_norms = {}
        for name, m in self.named_modules():
            if hasattr(m, 'parameters'):
                p_norm_m = 0
                for i, param in enumerate(m.parameters()):
                    p_norm_m += param.norm(p)
                p_norms[name] = p_norm_m
        return p_norms

    def forward(self, x):
        return self.model(x)

    def forward_from_latent(self, x):
        bs = x.shape[0]
        x = self.last_op(x)
        x = x.reshape(bs, self.output_shape[0])
        return x

    def latent_hook(self, module, input, output):
        self.latent_state = input[0].clone()


class InputLayer(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(n_channels_in, n_channels_out,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, n_channels_in, output_shape):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.output_shape = output_shape
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(n_channels_in, int(np.prod(output_shape)))

    def forward(self, x):
        bs = x.shape[0]
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.linear(x)
        # todo: used to be *self.output_shape but that breaks jit.
        # todo: this change might break other things though..!
        x = x.reshape(bs, self.output_shape[0])
        return x
