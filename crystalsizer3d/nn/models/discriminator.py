from typing import Tuple

from parti_pytorch.vit_vqgan import Discriminator as PartiDiscriminator, bce_discr_loss, bce_gen_loss, hinge_discr_loss, \
    hinge_gen_loss

from crystalsizer3d.nn.models.basenet import BaseNet


class Discriminator(BaseNet):
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],

            n_base_filters: int,
            n_layers: int = 4,
            loss_type: str = 'hinge',

            build_model: bool = True
    ):
        super().__init__(input_shape, output_shape)

        layer_mults = list(map(lambda t: 2**t, range(n_layers + 1)))
        self.layer_dims = [n_base_filters * mult for mult in layer_mults]

        # Get the loss functions
        if loss_type == 'hinge':
            self.discr_loss = hinge_discr_loss
            self.gen_loss = hinge_gen_loss
        elif loss_type == 'bce':
            self.discr_loss = bce_discr_loss
            self.gen_loss = bce_gen_loss

        if build_model:
            self._build_model()
            self._init_params()

    @property
    def id(self):
        return f'discriminator/l={self.latent_size}'

    def _build_model(self):
        self.model = PartiDiscriminator(
            dims=self.layer_dims,
            channels=1
        )

    def forward(self, x):
        x = self.model(x)

        # Convert to single logits
        x = x.mean(dim=(1, 2, 3))

        return x
