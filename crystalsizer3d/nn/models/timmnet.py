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
            resize_input: bool = True,
            build_model: bool = True
    ):
        super().__init__(input_shape, output_shape)
        self.model_name = model_name
        self.classifier_hidden_layers = classifier_hidden_layers
        self.dropout_prob = dropout_prob
        self.droppath_prob = droppath_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.resize_input = resize_input
        self.reshape_dim = None

        # Load pretrained model
        model_args = dict(
            model_name=self.model_name,
            pretrained=True,
            num_classes=0,
            drop_rate=self.dropout_prob,
            drop_path_rate=self.droppath_prob,
            img_size=input_shape[-1] if not self.resize_input else None,
        )
        try:
            self.model = timm.create_model(**model_args)
        except TypeError:
            del model_args['img_size']
            self.model = timm.create_model(**model_args)
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
        if 'mobilenet' in self.model_name:
            in_size = self.model.head_hidden_size
        else:
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

        # Check if reshape is required
        self._check_reshape_required()

    def _check_reshape_required(self):
        """
        Check if the model requires input to be reshaped to some power of two.
        """
        if self.resize_input:
            return

        x = torch.zeros(1, *self.input_shape)
        x_resized = x.clone()
        power = 2

        while power < 512:
            try:
                self.model.forward_features(x_resized)
                break
            except Exception as e:
                power *= 2
                if power < 256:
                    dim = int(round(self.input_shape[-1] / power) * power)
                    x_resized = F.interpolate(x, size=(dim, dim), mode='bilinear', align_corners=False)
                    self.reshape_dim = dim
                else:
                    raise e

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
        if self.resize_input and x.shape[-1] != self.data_config['input_size'][-1]:
            x = F.interpolate(x, size=self.data_config['input_size'][-2:], mode='bilinear', align_corners=False)
        elif self.reshape_dim:
            x = F.interpolate(x, size=(self.reshape_dim, self.reshape_dim), mode='bilinear', align_corners=False)

        # Normalise
        x = (x - self.img_mean) / self.img_std

        # Get the image embedding
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)

        # Feed into classifier
        x = self.classifier(x)
        x = x.reshape(bs, self.output_shape[0])

        return x
