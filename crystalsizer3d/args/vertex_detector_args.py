from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path

from crystalsizer3d import DATA_PATH
from crystalsizer3d.args.base_args import BaseArgs


class VertexDetectorArgs(BaseArgs):
    def __init__(
            self,
            vd_model_name: str = 'MAGVIT2',
            vd_mv2_config_path: Path = DATA_PATH / 'MAGVIT2' / 'imagenet_lfqgan_256_B.yaml',
            vd_mv2_checkpoint_path: Path = DATA_PATH / 'MAGVIT2' / 'imagenet_256_B.ckpt',
            vd_focal_loss_alpha: float = 0.75,
            vd_focal_loss_gamma: float = 2.0,
            **kwargs
    ):
        self.vd_model_name = vd_model_name
        self.vd_mv2_config_path = vd_mv2_config_path
        self.vd_mv2_checkpoint_path = vd_mv2_checkpoint_path
        self.vd_focal_loss_alpha = vd_focal_loss_alpha
        self.vd_focal_loss_gamma = vd_focal_loss_gamma

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Vertex Detector Args')
        group.add_argument('--vd-model-name', type=str, default='MAGVIT2',
                           help='Name of the denoiser model to use.')
        group.add_argument('--vd-mv2-config-path', type=Path,
                           default=DATA_PATH / 'MAGVIT2' / 'imagenet_lfqgan_256_B.yaml',
                           help='Path to the MAGVIT2 config file to use.')
        group.add_argument('--vd-mv2-checkpoint-path', type=Path, default=DATA_PATH / 'MAGVIT2' / 'imagenet_256_B.ckpt',
                           help='Path to the MAGVIT2 model checkpoint to use.')
        group.add_argument('--vd-focal-loss-alpha', type=float, default=0.75,
                           help='Alpha parameter for focal loss.')
        group.add_argument('--vd-focal-loss-gamma', type=float, default=2.0,
                           help='Gamma parameter for focal loss.')

        return group
