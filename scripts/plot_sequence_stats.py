import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
from matplotlib import pyplot as plt

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.util.plots import make_3d_digital_crystal_image
from crystalsizer3d.util.utils import json_to_numpy, print_args, set_seed, to_dict


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to plot some sequence stats.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for the random number generator.')

    # Sequence
    parser.add_argument('--sf-path', type=Path, required=True,
                        help='Path to the sequence fitter directory.')
    parser.add_argument('--start-image', type=int, default=0,
                        help='Start with this image.')
    parser.add_argument('--end-image', type=int, default=-1,
                        help='End with this image.')
    parser.add_argument('--frame-idxs', type=lambda s: (int(i) for i in s.split(',')), default=(3, 100, 502),
                        help='Plot these frame idxs.')

    args = parser.parse_args()

    assert args.sf_path.exists(), f'Path does not exist: {args.sf_path}'

    # Set the random seed
    set_seed(args.seed)

    return args


def _init() -> Tuple[Namespace, Path]:
    """
    Initialise the dataset and get the command line arguments.
    """
    args = get_args()
    print_args(args)

    # Write the args to the output dir
    output_dir = LOGS_PATH / START_TIMESTAMP
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'args.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    return args, output_dir


def plot_sequence_errors():
    """
    Plot the sequence errors.
    """
    args, output_dir = _init()
    bs = 5
    N = 20

    # Load the losses
    with open(args.sf_path / 'losses.json', 'r') as f:
        losses = json_to_numpy(json.load(f))
    losses = losses['total/train']
    losses = losses[::len(losses) // N]

    # Sample some frames
    probabilities = losses / losses.sum()
    sampled_idxs = np.random.choice(len(losses), bs, p=probabilities, replace=False)

    # Set up the colours
    colours = ['lightgrey'] * len(losses)
    for idx in sampled_idxs:
        colours[idx] = 'darkred'

    fig, ax = plt.subplots(1, 1,
                           figsize=(0.85, 0.47),
                           gridspec_kw={'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1}
                           )
    # fig = plt.figure(figsize=(5, 3))
    ax.bar(range(len(losses)), losses, color=colours)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(output_dir / 'sequence_errors.svg')
    plt.show()


def plot_3d_crystals():
    """
    Plot the 3d crystals.
    """
    args, output_dir = _init()
    frame_idxs = args.frame_idxs

    for i, idx in enumerate(frame_idxs):
        logger.info(f'Plotting digital crystal for frame {idx} ({i + 1}/{len(frame_idxs)})')
        scene = Scene.from_yml(args.sf_path / 'scenes' / 'eval' / f'{idx:04d}.yml')
        crystal = scene.crystal
        dig_pred = make_3d_digital_crystal_image(
            crystal=crystal,
            res=1000,
            wireframe_radius_factor=0.03,
            azim=-10,
            elev=65,
            roll=-85,
            distance=4,
        )
        dig_pred.save(output_dir / f'digital_predicted_{idx:02d}.png')


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    # plot_sequence_errors()
    plot_3d_crystals()
