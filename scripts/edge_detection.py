from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor

from crystalsizer3d import LOGS_PATH, ROOT_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.nn.models.rcf import RCF
from crystalsizer3d.util.utils import print_args, to_numpy

if USE_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            image_path: Path,
            model_path: Path = ROOT_PATH / 'tmp' / 'bsds500_pascal_model.pth',
            **kwargs
    ):
        assert image_path.exists(), f'Image path does not exist: {image_path}'
        self.image_path = image_path
        assert model_path.exists(), f'Model path does not exist: {model_path}'
        self.model_path = model_path

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Runtime Args')
        group.add_argument('--image-path', type=Path,
                           help='Path to the image to process.')
        group.add_argument('--model-path', type=Path, default=ROOT_PATH / 'tmp' / 'bsds500_pascal_model.pth',
                           help='Path to the edge detection model.')
        return group


def parse_arguments(printout: bool = True) -> RuntimeArgs:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Run edge detection on an image.')
    RuntimeArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    runtime_args = RuntimeArgs.from_args(args)

    return runtime_args


def load_rcf(model_path: Path) -> RCF:
    """
    Load the RCF edge finder model.
    """
    assert model_path.exists(), f'RCF checkpoint not found at {model_path}'
    logger.info(f'Loading RCF model checkpoint from {model_path}.')
    rcf = RCF()
    checkpoint = torch.load(model_path)
    rcf.load_state_dict(checkpoint, strict=False)
    rcf = torch.jit.script(rcf)
    rcf.to(device)
    rcf.eval()
    return rcf


def generate_edge_images():
    """
    Generate edge images for the given image.
    """
    args = parse_arguments()

    # Create an output directory
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_{args.image_path.name}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments to json file
    with open(save_dir / 'args.yml', 'w') as f:
        spec = args.to_dict()
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    # Save the input image
    img = Image.open(args.image_path)
    img.save(save_dir / 'input.png')

    # Detect edges
    rcf = load_rcf(args.model_path)
    img = to_tensor(img).to(device)[None, ...]
    feature_maps = rcf(img, apply_sigmoid=False)

    # Save the feature maps
    for i, feature_map in enumerate(feature_maps):
        feature_map = to_numpy(feature_map).squeeze()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        feature_map = (feature_map * 255).astype(np.uint8)
        if i == len(feature_maps) - 1:
            name = 'fused'
        else:
            name = f'feature_map_{i + 1}'
        Image.fromarray(feature_map).save(save_dir / f'{name}.png')


if __name__ == '__main__':
    generate_edge_images()
