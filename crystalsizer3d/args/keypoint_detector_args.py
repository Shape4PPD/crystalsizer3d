from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path

from crystalsizer3d import DATA_PATH
from crystalsizer3d.args.base_args import BaseArgs


class KeypointDetectorArgs(BaseArgs):
    def __init__(
            self,
            kd_model_name: str = 'MAGVIT2',
            kd_mv2_config_path: Path = DATA_PATH / 'MAGVIT2' / 'imagenet_lfqgan_256_B.yaml',
            kd_mv2_checkpoint_path: Path = DATA_PATH / 'MAGVIT2' / 'imagenet_256_B.ckpt',
            kd_focal_loss_alpha: float = 0.75,
            kd_focal_loss_gamma: float = 2.0,
            **kwargs
    ):
        self.kd_model_name = kd_model_name
        self.kd_mv2_config_path = kd_mv2_config_path
        self.kd_mv2_checkpoint_path = kd_mv2_checkpoint_path
        self.kd_focal_loss_alpha = kd_focal_loss_alpha
        self.kd_focal_loss_gamma = kd_focal_loss_gamma

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Keypoint Detector Args')
        group.add_argument('--kd-model-name', type=str, default='MAGVIT2',
                           help='Name of the denoiser model to use.')
        group.add_argument('--kd-mv2-config-path', type=Path,
                           default=DATA_PATH / 'MAGVIT2' / 'imagenet_lfqgan_256_B.yaml',
                           help='Path to the MAGVIT2 config file to use.')
        group.add_argument('--kd-mv2-checkpoint-path', type=Path, default=DATA_PATH / 'MAGVIT2' / 'imagenet_256_B.ckpt',
                           help='Path to the MAGVIT2 model checkpoint to use.')
        group.add_argument('--kd-focal-loss-alpha', type=float, default=0.75,
                           help='Alpha parameter for focal loss.')
        group.add_argument('--kd-focal-loss-gamma', type=float, default=2.0,
                           help='Gamma parameter for focal loss.')

        return group
