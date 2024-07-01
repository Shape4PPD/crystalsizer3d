from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path

from crystalsizer3d import DATA_PATH
from crystalsizer3d.args.base_args import BaseArgs


class DenoiserArgs(BaseArgs):
    def __init__(
            self,
            dn_model_name: str = 'MAGVIT2',
            dn_mv2_config_path: Path = DATA_PATH / 'MAGVIT2' / 'imagenet_lfqgan_256_B.yaml',
            dn_mv2_checkpoint_path: Path = DATA_PATH / 'MAGVIT2' / 'imagenet_256_B.ckpt',
            **kwargs
    ):
        self.dn_model_name = dn_model_name
        self.dn_mv2_config_path = dn_mv2_config_path
        self.dn_mv2_checkpoint_path = dn_mv2_checkpoint_path

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Denoiser Args')
        group.add_argument('--dn-model-name', type=str, default='MAGVIT2',
                           help='Name of the denoiser model to use.')
        group.add_argument('--dn-mv2-config-path', type=Path,
                           default=DATA_PATH / 'MAGVIT2' / 'imagenet_lfqgan_256_B.yaml',
                           help='Path to the MAGVIT2 config file to use.')
        group.add_argument('--dn-mv2-checkpoint-path', type=Path, default=DATA_PATH / 'MAGVIT2' / 'imagenet_256_B.ckpt',
                           help='Path to the MAGVIT2 model checkpoint to use.')

        return group
