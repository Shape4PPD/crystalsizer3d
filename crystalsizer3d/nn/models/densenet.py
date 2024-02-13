import math
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn

from crystalsizer3d.nn.models.basenet import BaseNet, InputLayer, OutputLayer


class DenseNet(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],
            n_init_channels: int,
            growth_rate: int,
            block_config: Tuple[int, ...],
            compression_factor: int,
            dropout_prob: float = 0.,
            build_model: bool = True
    ):
        super().__init__(input_shape, output_shape)
        self.n_init_channels = n_init_channels
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.compression_factor = compression_factor
        self.dropout_prob = dropout_prob
        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return 'DenseNet/{}i_{}_k={}_c={}_d={}'.format(self.n_init_channels, ','.join(map(str, self.block_config)),
                                                       self.growth_rate, self.compression_factor, self.dropout_prob)

    def _build_model(self):
        # First convolution layer
        input_layer = InputLayer(self.input_shape[0], self.n_init_channels)
        n_channels = self.n_init_channels

        # Add DenseBlocks
        components = nn.Sequential()
        for i, n_layers in enumerate(self.block_config):
            block = self._get_dense_block(n_layers, n_channels)
            components.add_module('Block_%d' % (i + 1), block)
            n_channels = n_channels + n_layers * self.growth_rate

            # Add Transition layers between DenseBlocks
            if i != len(self.block_config) - 1:
                n_compressed_channels = math.floor(n_channels * self.compression_factor)
                trans = _Transition(n_channels_in=n_channels,
                                    n_channels_out=n_compressed_channels,
                                    dropout_prob=self.dropout_prob)
                components.add_module('Transition_%d' % (i + 1), trans)
                n_channels = n_compressed_channels

        # Add OutputLayer
        output_layer = OutputLayer(n_channels_in=n_channels, output_shape=self.output_shape)

        # Construct model
        self.model = nn.Sequential(OrderedDict([
            ('input_layer', input_layer),
            ('components', components),
            ('output_layer', output_layer),
        ]))

        return self.model

    def _get_dense_block(self, n_layers, n_channels_in):
        return _DenseBlock(
            n_layers=n_layers,
            n_channels_in=n_channels_in,
            growth_rate=self.growth_rate,
            dropout_prob=self.dropout_prob)


class _DenseBlock(nn.Sequential):
    def __init__(self, n_layers, n_channels_in, growth_rate, dropout_prob=0.):
        super().__init__()
        self.dropout_prob = dropout_prob
        n_channels = n_channels_in
        for i in range(n_layers):
            layer = self._get_dense_layer(n_channels, growth_rate)
            self.add_module('Layer_%d' % (i + 1), layer)
            n_channels += growth_rate

    def _get_dense_layer(self, n_channels_in, n_channels_out):
        return _DenseLayer(
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            dropout_prob=self.dropout_prob)


class _CompositeLayer(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, kernel_size=3, stride=1, padding=1, dropout_prob=0.):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(n_channels_in, n_channels_out,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dropout = None
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class _DenseLayer(_CompositeLayer):
    def __init__(self, n_channels_in, n_channels_out, dropout_prob=0.):
        super().__init__(n_channels_in, n_channels_out,
                         kernel_size=3, stride=1, padding=1,
                         dropout_prob=dropout_prob)

    def _stack(self, x_in, x):
        return torch.cat([x_in, x], 1)

    def forward(self, x):
        x_in = x
        x = super(_DenseLayer, self).forward(x)
        x = self._stack(x_in, x)
        return x


class _Transition(_CompositeLayer):
    def __init__(self, n_channels_in, n_channels_out, dropout_prob=0.):
        super().__init__(n_channels_in, n_channels_out,
                         kernel_size=1, stride=1, padding=0,
                         dropout_prob=dropout_prob)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = super().forward(x)
        x = self.pool(x)
        return x
