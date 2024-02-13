import torch.nn as nn

from crystalsizer3d.nn.models.resnet import ResNet, _ResBlock


class PyramidNet(ResNet):
    def __init__(self, input_shape, output_shape, n_init_channels, block_config, shortcut_type, use_bottlenecks, alpha,
                 dropout_prob=0., build_model=True):
        super().__init__(input_shape, output_shape, n_init_channels, block_config, shortcut_type,
                         use_bottlenecks, dropout_prob, build_model=False)
        self.alpha = alpha

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        bottleneck_variety = 'A' if not self.use_bottlenecks else 'B'
        return 'PyramidNet/{}i_{}_a={}_sc-{}_{}_d={}'.format(
            self.n_init_channels, ','.join(map(str, self.block_config)),
            self.alpha, self.shortcut_type, bottleneck_variety, self.dropout_prob
        )

    def _build_model_components(self):
        total_res_layers = sum(self.block_config)
        add_rate = self.alpha // total_res_layers
        n_channels = self.n_init_channels

        components = nn.Sequential()
        for i, n_layers in enumerate(self.block_config):
            block = self._get_res_block(n_layers, n_channels, add_rate, i > 0)
            n_channels += add_rate * n_layers
            components.add_module('Block_%d' % (i + 1), block)

        return components, n_channels

    def _get_res_block(self, n_layers, n_channels_in, add_rate, downsample):
        return _PyramidResBlock(
            n_layers=n_layers,
            n_channels_in=n_channels_in,
            add_rate=add_rate,
            downsample=downsample,
            shortcut_type=self.shortcut_type,
            use_bottlenecks=self.use_bottlenecks,
            dropout_prob=self.dropout_prob
        )


class _PyramidResBlock(_ResBlock):
    def __init__(self, n_layers, n_channels_in, add_rate, downsample, shortcut_type, use_bottlenecks=False,
                 dropout_prob=0., build_layers=True):
        super().__init__(n_layers, n_channels_in, None, downsample, shortcut_type,
                         use_bottlenecks, dropout_prob, build_layers=False)
        if build_layers:
            for i in range(n_layers):
                n_channels = n_channels_in + add_rate
                downsample = downsample and i == 0  # downsample spatially at the start of each block
                layer = self._get_res_layer(n_channels_in, n_channels, downsample)
                self.add_module('Layer_%d' % (i + 1), layer)
                n_channels_in = n_channels
