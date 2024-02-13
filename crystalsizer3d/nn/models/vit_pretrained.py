from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel

from crystalsizer3d.nn.models.basenet import BaseNet
from crystalsizer3d.nn.models.fcnet import FCLayer


class ViTPretrainedNet(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],
            model_name: str,
            use_cls_token: bool,
            classifier_hidden_layers: Tuple[int, ...],
            vit_dropout_prob: float = 0.,
            classifier_dropout_prob: float = 0.,
            build_model: bool = True
    ):
        super().__init__(input_shape, output_shape)
        self.model_name = model_name
        self.use_cls_token = use_cls_token
        self.classifier_hidden_layers = classifier_hidden_layers
        self.vit_dropout_prob = vit_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob

        # Load the image means and stds as used for the pretrained model
        self.preprocessor = ViTImageProcessor.from_pretrained(self.model_name)
        self.register_buffer('img_mean', torch.tensor(self.preprocessor.image_mean)[None, :, None, None])
        self.register_buffer('img_std', torch.tensor(self.preprocessor.image_std)[None, :, None, None])

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return f'ViTNet/{self.model_name}'

    @torch.jit.ignore
    def get_n_classifier_params(self) -> int:
        return sum([p.data.nelement() for p in self.classifier.parameters()])

    def _build_model(self):
        # Load pretrained model
        self.model = ViTModel.from_pretrained(
            self.model_name,
            add_pooling_layer=False,
            hidden_dropout_prob=self.vit_dropout_prob
        )

        # Build classifier
        if self.use_cls_token:
            size = self.model.config.hidden_size
        else:
            image_size = self.model.config.image_size  # Image size (224)
            patch_size = self.model.config.patch_size  # Patch size (16)
            num_patches = (image_size // patch_size)**2  # Number of patches
            size = self.model.config.hidden_size * (1 + num_patches)
        self.classifier = nn.Sequential()
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
        out_size = int(torch.prod(torch.tensor(self.output_shape)))
        self.classifier.add_module(
            'OutputLayer',
            FCLayer(size, out_size, activation=False)
        )

        # Add hook to store the last hidden state
        self.last_op = self.classifier.get_submodule('OutputLayer').get_submodule('linear')
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
        if x.shape[-1] != self.model.config.image_size:
            x = F.interpolate(x, size=self.model.config.image_size, mode='bilinear', align_corners=False)

        # Normalise
        x = (x - self.img_mean) / self.img_std

        # Get the image embedding
        x = self.model(x)
        if self.use_cls_token:
            # Get the CLS token embedding from the ViT model
            x = x.last_hidden_state[:, 0, :]
        else:
            # Use the full embedding
            x = x.last_hidden_state.reshape(bs, -1)

        # Feed into classifier
        x = self.classifier(x)
        x = x.reshape(bs, self.output_shape[0])

        return x
