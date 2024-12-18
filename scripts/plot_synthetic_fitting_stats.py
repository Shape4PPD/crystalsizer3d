import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
from matplotlib import pyplot as plt

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.args.refiner_args import KEYPOINTS_ARG_NAMES, RefinerArgs
from crystalsizer3d.args.synthetic_fitter_args import SyntheticFitterArgs
from crystalsizer3d.util.utils import hash_data, json_to_numpy, print_args, set_seed, to_dict


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to plot some synthetic fitting stats.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for the random number generator.')

    # Synthetic
    parser.add_argument('--sf-path', type=Path, required=True,
                        help='Path to the synthetic fitter directory.')
    parser.add_argument('--include-checkpoints', type=lambda s: [int(item) for item in s.split(',')],
                        default=[],
                        help='Include these checkpoints in the plot. Leave empty to include all checkpoints.')

    args = parser.parse_args()

    assert args.sf_path.exists(), f'Path does not exist: {args.sf_path}'

    # Set the random seed
    set_seed(args.seed)

    return args


def _init() -> Tuple[Namespace, SyntheticFitterArgs, RefinerArgs, Namespace, Path]:
    """
    Initialise the args and output directory.
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

    # Load the args
    with open(args.sf_path / 'args_synthetic_fitter.yml', 'r') as f:
        sf_args = SyntheticFitterArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_refiner.yml', 'r') as f:
        ref_args = RefinerArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(args.sf_path / 'args_runtime.yml', 'r') as f:
        runtime_args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    return args, sf_args, ref_args, runtime_args, output_dir


def plot_fitting_losses():
    """
    Plot the fitting losses.
    """
    args, sf_args, ref_args, runtime_args, output_dir = _init()
    N = sf_args.n_samples

    # Load the ds idxs
    ds_idxs_path = args.sf_path / 'ds_idxs.json'
    with open(ds_idxs_path, 'r') as f:
        ds_idxs = json_to_numpy(json.load(f))

    # Load the losses
    losses_path = args.sf_path / 'losses.json'
    logger.info(f'Loading losses from {losses_path}.')
    with open(losses_path, 'r') as f:
        losses = json_to_numpy(json.load(f))

    # Determine the checkpoints
    checkpoints = [int(c) for c in list(losses['checkpoints'].keys())]
    checkpoints.sort()
    if len(args.include_checkpoints) > 0:
        checkpoints = [c for c in checkpoints if c in args.include_checkpoints]
        assert len(checkpoints) == len(args.include_checkpoints), \
            f'Not all requested checkpoints ({args.include_checkpoints}) found in {checkpoints}.'

    # Extract the losses
    L_x = np.zeros((1 + len(checkpoints), N))  # image loss
    L_k = np.zeros_like(L_x)  # keypoints loss
    L_d = np.zeros_like(L_x)  # distance loss
    L_v = np.zeros_like(L_x)  # vertex loss
    for i in range(1 + len(checkpoints)):
        if i == 0:
            L_x[i] = losses['init']['l2']
            L_k[i] = losses['init']['keypoints']
            L_d[i] = losses['init']['distances']
            L_v[i] = losses['init']['vertices']
        else:
            ck = str(checkpoints[i - 1])
            L_x[i] = losses['checkpoints'][ck]['l2']
            L_k[i] = losses['checkpoints'][ck]['keypoints']
            L_d[i] = losses['checkpoints'][ck]['distances']
            L_v[i] = losses['checkpoints'][ck]['vertices']

    # Load the keypoints
    kp_model_args = {k: getattr(ref_args, k) for k in KEYPOINTS_ARG_NAMES}
    kp_model_id = kp_model_args['keypoints_model_path'].stem[:4]
    keypoints_dir = args.sf_path.parent.parent / 'keypoints' / f'{kp_model_id}_{hash_data(kp_model_args)}' / 'keypoints'
    assert keypoints_dir.exists(), f'Keypoints cache dir does not exist: {keypoints_dir}'
    keypoints = []
    n_keypoints = []
    for i, idx in enumerate(ds_idxs):
        keypoints_path = keypoints_dir / f'{idx:010d}.json'
        assert keypoints_path.exists()
        with open(keypoints_path) as f:
            keypoints_i = json_to_numpy(json.load(f))
        keypoints.append(keypoints_i)
        n_keypoints.append(len(keypoints_i))

    # Exclude data where fewer than 5 keypoints were detected
    n_keypoints = np.array(n_keypoints)
    mask = n_keypoints >= 5
    logger.info(f'Excluding {np.sum(~mask)} samples with fewer than 5 keypoints detected.')
    L_x = L_x[:, mask]
    L_k = L_k[:, mask]
    L_d = L_d[:, mask]
    L_v = L_v[:, mask]

    # Calculate the mean and std of the relative loss changes
    data = np.zeros((9, L_x.shape[0]))
    data[0, 1:] = checkpoints
    headers = ['iterations']
    for i, (L, k) in enumerate(zip([L_x, L_k, L_d, L_v], ['image', 'keypoints', 'distances', 'vertices'])):
        L_rel = (L - L[0]) / L[0]
        L_rel_mean = np.mean(L_rel, axis=1)
        L_rel_std = np.std(L_rel, axis=1)
        data[i * 2 + 1] = L_rel_mean
        data[i * 2 + 2] = L_rel_std
        headers.extend([f'{k}_mean', f'{k}_std'])

    # Write data to csv file
    data_path = output_dir / 'fitting_losses.csv'
    logger.info(f'Writing data to {data_path}.')
    np.savetxt(data_path, data.T, delimiter=',', header=','.join(headers))

    # Set font sizes
    plt.rc('axes', titlesize=11, titlepad=6)  # fontsize of the title
    plt.rc('axes', labelsize=9, labelpad=4)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=7)  # fontsize of the legend

    # Set up the figure
    fig, axes = plt.subplots(
        2, 2,
        figsize=(7, 5.5),
        gridspec_kw=dict(
            top=0.95, bottom=0.1, right=0.99, left=0.08,
            hspace=0.5, wspace=0.3
        ),
    )

    x_vals = np.arange(1 + len(checkpoints)) + 1
    x_labels = ['Initial\nprediction'] + [f'{c}' for c in checkpoints]

    def plot_loss(ax_, L_, title_, y_label_):
        # Plot the violin plot
        ax_.violinplot(L_.T, widths=0.7, showmeans=False, showmedians=False)

        # Plot the mean and median
        ax_.plot(x_vals, np.mean(L_, axis=1), color='red', linestyle='--', label='Mean', marker='x', markersize=8)
        ax_.plot(x_vals, np.median(L_, axis=1), color='purple', linestyle='--', label='Median', marker='o',
                 markerfacecolor='none', markersize=7)

        ax_.set_yscale('log')
        ax_.set_xticks(x_vals)
        ax_.set_xticklabels(x_labels)
        ax_.set_title(title_)
        ax_.set_xlabel('Number of iterations')
        ax_.set_ylabel(y_label_, fontsize=13)
        ax_.legend(loc='lower left', markerscale=0.5)

    for i, (L, title, y_label) in enumerate(zip(
            [L_x, L_k, L_d, L_v],
            ['Rendered image loss', '2D keypoints loss', 'Face distances error', '3D vertices error'],
            ['$\mathcal{L}_X$', '$\mathcal{L}_K$', '$\mathcal{E}_d$', '$\mathcal{E}_v$']
    )):
        ax = axes.flatten()[i]
        plot_loss(ax, L, title, y_label)

    # Save the figure
    plt.savefig(output_dir / 'fitting_losses.svg', transparent=True)
    plt.show()


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    plot_fitting_losses()
