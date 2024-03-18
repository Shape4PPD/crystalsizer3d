from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import resolve_model_data_config

from crystalsizer3d.nn.models.basenet import BaseNet
from crystalsizer3d.nn.models.fcnet import FCLayer


class TimmNet(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],
            model_name: str,
            classifier_hidden_layers: Tuple[int, ...],
            dropout_prob: float = 0.,
            droppath_prob: float = 0.,
            classifier_dropout_prob: float = 0.,
            build_model: bool = True
    ):
        super().__init__(input_shape, output_shape)
        self.model_name = model_name
        self.classifier_hidden_layers = classifier_hidden_layers
        self.dropout_prob = dropout_prob
        self.droppath_prob = droppath_prob
        self.classifier_dropout_prob = classifier_dropout_prob

        # Load pretrained model
        self.model = timm.create_model(
            model_name=self.model_name,
            pretrained=True,
            num_classes=0,
            drop_rate=self.dropout_prob,
            drop_path_rate=self.droppath_prob,
        )
        self.data_config = resolve_model_data_config(self.model)

        # Load the image means and stds as used for the pretrained model
        self.register_buffer('img_mean', torch.tensor(self.data_config['mean'])[None, :, None, None])
        self.register_buffer('img_std', torch.tensor(self.data_config['std'])[None, :, None, None])

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return f'timm/{self.model_name}'

    @torch.jit.ignore
    def get_n_classifier_params(self) -> int:
        return sum([p.data.nelement() for p in self.classifier.parameters()])

    def _build_model(self):
        in_size = self.model.num_features
        out_size = int(torch.prod(torch.tensor(self.output_shape)))

        # Build classifier
        self.classifier = nn.Sequential()
        size = in_size
        for i, n in enumerate(self.classifier_hidden_layers):
            if self.classifier_dropout_prob > 0:
                self.classifier.add_module(
                    f'HiddenLayer{i}_dropout',
                    nn.Dropout(self.classifier_dropout_prob)
                )
            self.classifier.add_module(
                f'HiddenLayer{i}',
                FCLayer(size, n, activation=i != 0)  # skip relu going into first layer
            )
            size = n
        self.classifier.add_module(
            'OutputLayer',
            nn.Linear(size, out_size)
        )

        # Add hook to store the last hidden state
        self.last_op = self.classifier.get_submodule('OutputLayer')
        self.last_op.register_forward_hook(self.latent_hook)

    def _init_params(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bs = x.shape[0]

        # Expand grayscale to RGB
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        # Resize to required input shape
        if x.shape[-1] != self.data_config['input_size'][-1]:
            x = F.interpolate(x, size=self.data_config['input_size'][-2:], mode='bilinear', align_corners=False)

        # Normalise
        x = (x - self.img_mean) / self.img_std

        # Get the image embedding
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)

        # Feed into classifier
        x = self.classifier(x)
        x = x.reshape(bs, self.output_shape[0])

        return x
