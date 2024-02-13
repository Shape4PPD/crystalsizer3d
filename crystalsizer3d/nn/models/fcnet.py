from typing import Tuple

import torch
import torch.nn as nn

from crystalsizer3d.nn.models.basenet import BaseNet


class FCLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, activation: bool = True):
        super().__init__()
        self.activation = activation
        self.bn = nn.BatchNorm1d(n_in, affine=False)
        if activation:
            self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        x = self.linear(x)
        return x


class FCNet(BaseNet):
    def __init__(
            self,
            input_shape: tuple,
            output_shape: tuple,
            layers_config: Tuple[int],
            dropout_prob: float = 0.,
            build_model=True,
            act_out=None
    ):
        super().__init__(input_shape, output_shape)

        self.layers_config = layers_config
        self.dropout_prob = dropout_prob
        self.act_out = act_out

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return f'FCNet/{",".join(map(str, self.layers_config))}_d={self.dropout_prob}'

    def _build_model(self):
        size = int(torch.prod(torch.tensor(self.input_shape)))
        self.model = nn.Sequential()
        for i, n in enumerate(self.layers_config):
            self.model.add_module(
                f'HiddenLayer{i}',
                FCLayer(size, n, activation=i != 0)  # skip relu going into first layer
            )
            size = n

        out_size = int(torch.prod(torch.tensor(self.output_shape)))
        self.model.add_module(
            'OutputLayer',
            FCLayer(size, out_size, activation=self.act_out)
        )

        return self.model

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return x.reshape(x.shape[0], *self.output_shape)
