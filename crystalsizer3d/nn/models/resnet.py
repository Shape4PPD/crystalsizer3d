from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd.variable import Variable

from crystalsizer3d.nn.models.basenet import BaseNet, InputLayer, OutputLayer

RES_SHORTCUT_OPTIONS = ['id', 'conv']


class ResNet(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int],
            output_shape: Tuple[int],
            n_init_channels: int,
            block_config: Tuple[int],
            shortcut_type: str,
            use_bottlenecks: bool,
            dropout_prob: float = 0.,
            build_model: bool = True
    ):
        super(ResNet, self).__init__(input_shape, output_shape)

        assert shortcut_type in RES_SHORTCUT_OPTIONS
        self.n_init_channels = n_init_channels
        self.block_config = block_config
        self.shortcut_type = shortcut_type
        self.use_bottlenecks = use_bottlenecks
        self.dropout_prob = dropout_prob

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        bottleneck_variety = 'A' if not self.use_bottlenecks else 'B'
        return 'ResNet/{}i_{}_sc-{}_{}_d={}'.format(self.n_init_channels, ','.join(map(str, self.block_config)),
                                                    self.shortcut_type, bottleneck_variety, self.dropout_prob)

    def _build_model(self):
        # First convolution layer
        input_layer = InputLayer(self.input_shape[0], self.n_init_channels)

        # Residual components
        components, n_channels = self._build_model_components()

        # Add OutputLayer
        output_layer = OutputLayer(n_channels_in=n_channels, output_shape=self.output_shape)

        # Construct model
        self.model = nn.Sequential(OrderedDict([
            ('input_layer', input_layer),
            ('components', components),
            ('output_layer', output_layer),
        ]))

        return self.model

    def _build_model_components(self):
        n_channels = self.n_init_channels
        components = nn.Sequential()
        for i, n_layers in enumerate(self.block_config):
            n_channels_in = n_channels
            if i > 0:
                n_channels *= 2  # number of channels doubles after each set of blocks
            block = self._get_res_block(n_layers, n_channels_in, n_channels, i > 0)
            components.add_module('Block_%d' % (i + 1), block)
        return components, n_channels

    def _get_res_block(self, n_layers, n_channels_in, n_channels, downsample):
        return _ResBlock(
            n_layers=n_layers,
            n_channels_in=n_channels_in,
            n_channels=n_channels,
            downsample=downsample,
            shortcut_type=self.shortcut_type,
            use_bottlenecks=self.use_bottlenecks,
            dropout_prob=self.dropout_prob
        )


class _ResBlock(nn.Sequential):
    def __init__(self, n_layers, n_channels_in, n_channels, downsample, shortcut_type, use_bottlenecks=False,
                 dropout_prob=0., build_layers=True):
        super(_ResBlock, self).__init__()
        self.shortcut_type = shortcut_type
        self.use_bottlenecks = use_bottlenecks
        self.dropout_prob = dropout_prob
        if build_layers:
            for i in range(n_layers):
                n_channels_in = n_channels_in if i == 0 else n_channels
                downsample = downsample and i == 0  # downsample spatially at the start of each block
                layer = self._get_res_layer(n_channels_in, n_channels, downsample)
                self.add_module('Layer_%d' % (i + 1), layer)

    def _get_res_layer(self, n_channels_in, n_channels, downsample):
        Layer = _ResLayer if not self.use_bottlenecks else _ResBottleneckLayer
        return Layer(
            n_channels_in=n_channels_in,
            n_channels=n_channels,
            downsample=downsample,
            shortcut_type=self.shortcut_type,
            dropout_prob=self.dropout_prob
        )


class _ResLayer(nn.Module):
    def __init__(self, n_channels_in, n_channels, downsample=False, shortcut_type='id', dropout_prob=0.):
        super(_ResLayer, self).__init__()

        self.shortcut = _Shortcut(n_channels_in, n_channels, downsample, shortcut_type)

        conv1_stride = 2 if downsample else 1

        self.bn1 = nn.BatchNorm2d(n_channels_in)
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size=3, stride=conv1_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_channels)

        self.dropout = None
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x_in = self.shortcut(x)
        x = self.bn1(x)
        x = self.conv1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        return x + x_in


class _ResBottleneckLayer(nn.Module):
    def __init__(self, n_channels_in, n_channels, downsample=False, shortcut_type='id', dropout_prob=0.):
        super(_ResBottleneckLayer, self).__init__()

        self.shortcut = _Shortcut(n_channels_in, n_channels, downsample, shortcut_type)

        # todo: this differs from other implementations, but makes much more sense to me this way
        n_bottleneck_channels = n_channels // 4
        conv1_stride = 2 if downsample else 1

        self.bn1 = nn.BatchNorm2d(n_channels_in)
        self.conv1 = nn.Conv2d(n_channels_in, n_bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(n_bottleneck_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_bottleneck_channels, n_bottleneck_channels, kernel_size=3, stride=conv1_stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(n_bottleneck_channels, n_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(n_channels)

        self.dropout = None
        if dropout_prob > 0:
            self.dropout1 = nn.Dropout(dropout_prob)
            self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x_in = self.shortcut(x)
        x = self.bn1(x)
        x = self.conv1(x)
        if self.dropout:
            x = self.dropout1(x)
        x = self.bn2(x)
        x = self.relu1(x)
        x = self.conv2(x)
        if self.dropout:
            x = self.dropout2(x)
        x = self.bn3(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn4(x)
        return x + x_in


class _Shortcut(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, downsample, shortcut_type):
        super(_Shortcut, self).__init__()
        self.shortcut = nn.Sequential()

        # when number of channels changes we need to project input to new size
        if n_channels_in != n_channels_out:
            if shortcut_type == 'conv':
                stride = 2 if downsample else 1
                self.shortcut.add_module('bn', nn.BatchNorm2d(n_channels_in))
                self.shortcut.add_module('relu', nn.ReLU(inplace=True))
                self.shortcut.add_module('conv', nn.Conv2d(n_channels_in, n_channels_out, kernel_size=1, stride=stride,
                                                           bias=False))
            elif shortcut_type == 'id':
                if downsample:
                    self.shortcut.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))
                self.shortcut.add_module('zero_pad', _ZeroPadChannels(n_channels_out - n_channels_in))

    def forward(self, x):
        return self.shortcut(x)


class _ZeroPadChannels(nn.Module):
    def __init__(self, pad_size):
        super(_ZeroPadChannels, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        batch_size, _, spatial_size_1, spatial_size_2 = x.shape
        zeros = Variable(torch.zeros([batch_size, self.pad_size, spatial_size_1, spatial_size_2], device=x.device))
        return torch.cat([x, zeros], 1)
