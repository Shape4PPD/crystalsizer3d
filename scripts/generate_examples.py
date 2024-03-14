import time
from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from ccdc.morphology import VisualHabitMorphology
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from trimesh import Trimesh

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.args.dataset_training_args import PREANGLES_MODE_AXISANGLE, PREANGLES_MODE_QUATERNION, PREANGLES_MODE_SINCOS
from crystalsizer3d.nn.manager import CCDC_AVAILABLE, Manager
from crystalsizer3d.util.utils import equal_aspect_ratio, normalise, print_args, str2bool, to_dict, to_numpy


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            model_path: Path,
            use_gpu: bool = False,
            batch_size: int = 4,
            vary_parameters: List[str] = [],
            n_vary_steps: int = 10,
            **kwargs
    ):
        assert model_path.exists(), f'Dataset path does not exist: {model_path}'
        assert model_path.suffix == '.json', f'Model path must be a json file: {model_path}'
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        assert len(vary_parameters) > 0, f'Vary parameters must be a non-empty list: {vary_parameters}'
        self.vary_parameters = vary_parameters
        self.n_vary_steps = n_vary_steps

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Runtime Args')
        group.add_argument('--model-path', type=Path, required=True,
                           help='Path to the model\'s json file.')
        parser.add_argument('--use-gpu', type=str2bool, default=USE_CUDA,
                            help='Use GPU. Defaults to environment setting.')
        parser.add_argument('--batch-size', type=int, default=4,
                            help='Batch size.')
        parser.add_argument('--vary-parameters', type=lambda s: [str(item) for item in s.split(',')], default='',
                            help='Parameter keys to vary.')
        parser.add_argument('--n-vary-steps', type=int, default=10,
                            help='Number of interpolation points to vary the parameters.')
        return group


def parse_arguments(printout: bool = True) -> RuntimeArgs:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Predict 3D morphology using a neural network.')
    RuntimeArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    runtime_args = RuntimeArgs.from_args(args)

    return runtime_args


def _build_crystal(
        Y: Dict[str, torch.Tensor],
        idx: int,
        manager: Manager,
) -> Tuple[VisualHabitMorphology, Trimesh]:
    if not CCDC_AVAILABLE:
        raise RuntimeError('CCDC unavailable.')

    # Normalise the distances
    distances = to_numpy(Y['distances'][idx]).astype(float)
    if manager.dataset_args.use_distance_switches:
        switches = to_numpy(Y['distance_switches'][idx])
        distances = np.where(switches < .5, 0, distances)
    distances[distances < 0] = 0

    if distances.max() < 1e-8:
        raise RuntimeError('Failed to build crystal:\nno positive distances.')

    # Build the crystal and plot the mesh
    distances /= distances.max()
    try:
        growth_rates = manager.crystal_generator.get_expanded_growth_rates(distances)
        morph = VisualHabitMorphology.from_growth_rates(manager.crystal_generator.crystal,
                                                        growth_rates)
        _, _, mesh = manager.crystal_generator.generate_crystal(rel_rates=distances,
                                                                validate=False)
        return morph, mesh
    except Exception as e:
        raise RuntimeError(f'Failed to build crystal:\n{e}')


def _plot_error(ax: Axes, err: str):
    txt = ax.text(
        0.5, 0.5, err,
        horizontalalignment='center',
        verticalalignment='center',
        wrap=True
    )
    txt._get_wrap_line_width = lambda: ax.bbox.width * 0.7
    ax.axis('off')


def _plot_image(ax: Axes, title: Optional[str], img: np.ndarray, font_size: int = 12):
    bbox = ax.get_tightbbox().transformed(ax.figure.transFigure.inverted())
    ax.set_position(bbox)
    if title is not None:
        ax.set_title(title, fontsize=font_size)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')


def _plot_3d(
        ax: Axes,
        title: Optional[str],
        morph: Optional[VisualHabitMorphology],
        mesh: Trimesh,
        colour: str
):
    if title is not None:
        ax.set_title(title)
    if morph is not None:
        for f in morph.facets:
            for edge in f.edges:
                coords = np.array(edge)
                ax.plot(*coords.T, c=colour)

    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        triangles=mesh.faces,
        Z=mesh.vertices[:, 2],
        color=colour,
        alpha=0.5
    )
    equal_aspect_ratio(ax, zoom=1.2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    bbox = ax.get_tightbbox().transformed(ax.figure.transFigure.inverted())
    ax.set_position(bbox)


def _plot_distances(
        ax: Axes,
        Y_pred: Dict[str, torch.Tensor],
        idx: int,
        manager: Manager,
        share_ax: Optional[Dict[str, Axes]] = None
):
    ax_pos = ax.get_position()
    ax.set_position([ax_pos.x0, ax_pos.y0 + 0.02, ax_pos.width, ax_pos.height - 0.02])
    d_pred = to_numpy(Y_pred['distances'][idx])
    d_pred = np.clip(d_pred, a_min=0, a_max=np.inf)
    if d_pred.max() > 1e-8:
        d_pred /= d_pred.max()

    locs = np.arange(len(d_pred))
    bar_width = 0.7
    ax.bar(locs, d_pred, bar_width, label='Predicted')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)

    if manager.dataset_args.use_distance_switches:
        s_pred = to_numpy(Y_pred['distance_switches'][idx])
        colours = []
        for i, sp in enumerate(s_pred):
            if sp > 0.5:
                ax.axvspan(i - bar_width / 2, i + bar_width / 2, alpha=0.1, color='green')
            colours.append('red' if sp < 0.5 else 'green')
        ax.scatter(locs, s_pred, color=colours, marker='+', s=100, label='Switches')

    ax.set_title('Distances')
    ax.set_xticks(locs)
    ax.set_xticklabels(manager.ds.labels_distances)
    ax.tick_params(axis='x', rotation=270)
    if share_ax is None or 'distances' not in share_ax:
        if share_ax is not None:
            share_ax['distances'] = ax
    else:
        ax.sharey(share_ax['distances'])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.autoscale()


def _plot_transformation(
        ax: Axes,
        Y_pred: Dict[str, torch.Tensor],
        idx: int,
        manager: Manager,
        share_ax: Optional[Dict[str, Axes]] = None
):
    t_pred = to_numpy(Y_pred['transformation'][idx])
    locs = np.arange(len(t_pred))
    bar_width = 0.7
    ax.bar(locs, t_pred, bar_width, label='Predicted')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    k = 3 / 4 * bar_width
    ax.axvspan(locs[0] - k, locs[2] + k, alpha=0.1, color='green')
    ax.axvspan(locs[3] - k, locs[3] + k, alpha=0.1, color='red')
    ax.axvspan(locs[4] - k, locs[-1] + k, alpha=0.1, color='blue')
    ax.set_title('Transformation')
    ax.set_xticks(locs)
    xlabels = manager.ds.labels_transformation.copy()
    if manager.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
        xlabels += manager.ds.labels_transformation_sincos
    elif manager.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
        xlabels += manager.ds.labels_transformation_quaternion
    else:
        assert manager.dataset_args.preangles_mode == PREANGLES_MODE_AXISANGLE
        xlabels += manager.ds.labels_transformation_axisangle
    ax.set_xticklabels(xlabels)
    if 'transformation' not in share_ax:
        # ax_.legend()
        share_ax['transformation'] = ax
    else:
        ax.sharey(share_ax['transformation'])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.autoscale()


def _plot_material(
        ax: Axes,
        Y_pred: Dict[str, torch.Tensor],
        idx: int,
        share_ax: Optional[Dict[str, Axes]] = None
):
    m_pred = to_numpy(Y_pred['material'][idx])
    locs = np.arange(len(m_pred))
    bar_width = 0.35
    ax.bar(locs, m_pred, bar_width, label='Predicted')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_title('Material')
    ax.set_xticks(locs)
    ax.set_xticklabels(['Brightness', 'IOR'])
    if 'material' not in share_ax:
        # ax_.legend()
        share_ax['material'] = ax
    else:
        ax.sharey(share_ax['material'])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.autoscale()


def _plot_light(
        ax: Axes,
        Y_pred: Dict[str, torch.Tensor],
        idx: int,
        manager: Manager,
        share_ax: Optional[Dict[str, Axes]] = None
):
    l_pred = to_numpy(Y_pred['light'][idx])
    locs = np.arange(len(l_pred))
    bar_width = 0.7
    ax.bar(locs, l_pred, bar_width, label='Predicted')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    k = 3 / 4 * bar_width
    ax.axvspan(locs[0] - k, locs[2] + k, alpha=0.1, color='green')
    ax.axvspan(locs[3] - k, locs[3] + k, alpha=0.1, color='red')
    ax.axvspan(locs[4] - k, locs[-1] + k, alpha=0.1, color='blue')
    ax.set_title('Light')
    ax.set_xticks(locs)
    xlabels = manager.ds.labels_light.copy()
    if manager.dataset_args.preangles_mode == PREANGLES_MODE_SINCOS:
        xlabels += manager.ds.labels_light_sincos
    elif manager.dataset_args.preangles_mode == PREANGLES_MODE_QUATERNION:
        xlabels += manager.ds.labels_light_quaternion
    else:
        assert manager.dataset_args.preangles_mode == PREANGLES_MODE_AXISANGLE
        xlabels += manager.ds.labels_light_axisangle
    ax.set_xticklabels(xlabels)
    if 'light' not in share_ax:
        # ax_.legend()
        share_ax['light'] = ax
    else:
        ax.sharey(share_ax['light'])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.autoscale()


def _plot_results(
        k: str,
        idx: int,
        p_val: float,
        Y: Dict[str, torch.Tensor],
        X: torch.Tensor,
        manager: Manager,
) -> Figure:
    """
    Plot the image and parameter predictions for parameter vector.
    """
    plt.rc('axes', labelsize=10)  # fontsize of the axes labels
    plt.rc('axes', titlesize=12, titlepad=4)  # fontsize of the axes labels
    plt.rc('xtick', labelsize=10)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=10)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=2, size=2)
    plt.rc('ytick.major', pad=2, size=2)

    fig = plt.figure(figsize=(7, 9))
    gs = GridSpec(
        nrows=3,
        ncols=2,
        wspace=0.2,
        hspace=0.3,
        height_ratios=[1.8, 1.2, 0.8],
        top=0.92,
        bottom=0.04,
        left=0.1,
        right=0.96
    )

    fig.suptitle(f'Varying "{k}": {p_val:.3f}', fontweight='bold')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    share_ax = {}

    # Plot the 3D morphology
    try:
        morph, mesh = _build_crystal(Y, idx, manager)
        _plot_3d(fig.add_subplot(gs[0, 0], projection='3d'), 'Digital crystal',
                 morph, mesh, default_colours[1])
    except Exception:
        _plot_error(fig.add_subplot(gs[0, 0]), 'Failed to build crystal.')

    # Plot the generated image
    img_i = to_numpy(X[idx]).squeeze()
    _plot_image(fig.add_subplot(gs[0, 1]), 'Generated image', img_i)

    # Plot the parameters
    _plot_distances(fig.add_subplot(gs[1, 0]), Y, idx, manager, share_ax)
    _plot_transformation(fig.add_subplot(gs[1, 1]), Y, idx, manager, share_ax)
    _plot_material(fig.add_subplot(gs[2, 0]), Y, idx, share_ax)
    _plot_light(fig.add_subplot(gs[2, 1]), Y, idx, manager, share_ax)

    # Highlight the parameter being varied
    for ax in fig.get_axes():
        for xtick in ax.get_xticklabels():
            if xtick._text == k:
                xtick.set_color('red')

    return fig


def _make_parameter_vector(
        manager: Manager,
        item: Dict[str, float],
        defaults: Optional[torch.Tensor] = None
):
    """
    Build a parameter vector.
    """
    ds = manager.ds
    ds_args = manager.dataset_args
    if defaults is None:
        Y = torch.zeros(len(ds.labels))
    else:
        Y = defaults.clone()
    idx = 0

    # Zingg parameters are always in [0, 1]
    if ds_args.train_zingg:
        if 'si' in item:
            assert 0 <= item['si'] <= 1, f'Zingg parameter si must be in [0, 1]. {item["si"]} received.'
            Y[0] = item['si']
        if 'il' in item:
            assert 0 <= item['il'] <= 1, f'Zingg parameter il must be in [0, 1]. {item["il"]} received.'
            Y[1] = item['il']
        idx += 2

    # Distances are always in [0, 1], where 0 indicates a collapsed face / missing distance
    if ds_args.train_distances:
        n_distances = len(ds.labels_distances)
        dists = Y[idx:idx + n_distances]
        for i, k in enumerate(ds.labels_distances):
            if k in item:
                dists[i] = item[k]

        if ds_args.use_distance_switches:
            switches = Y[idx + n_distances:idx + 2 * n_distances]
            for i, (k, k_d) in enumerate(zip(ds.labels_distance_switches, ds.labels_distances)):
                if k in item:
                    assert item[k] in [0, 1], f'Distance switch {k} must be in {0, 1}. {item[k]} received.'
                    switches[i] = item[k]
                if k_d in item and item[k_d] > 0:  # Always switch on if the item has requested a distance
                    switches[i] = 1
            switch_gates = switches > 0.5
            assert not torch.all(switch_gates == 0), 'Some distance switches must be non-zero.'
            dists *= switch_gates  # Apply the switches to the distances

        # Normalise the distances
        assert not torch.all(dists == 0), 'Some distances must be non-zero.'
        dists /= dists.max()
        try:
            rel_rates2, zingg, _ = manager.crystal_generator.generate_crystal(
                rel_rates=to_numpy(dists).astype(float),
                validate=False
            )
            dists = torch.from_numpy(rel_rates2)
            if ds_args.train_zingg:
                Y[:2] = torch.from_numpy(zingg)
        except Exception as e:
            logger.warning(f'Failed to generate crystal: {e}')

        # Add to parameter vector
        Y[idx:idx + n_distances] = dists
        idx += n_distances
        if ds_args.use_distance_switches:
            Y[idx:idx + n_distances] = switch_gates
            idx += n_distances

    # Transformation parameters are normalised by the dataset statistics
    if ds_args.train_transformation:
        # Position
        for xyz in 'xyz':
            if xyz in item:
                assert -1 <= item[xyz] <= 1, f'Transformation parameter {xyz} must be in [-1, 1]. {item[xyz]} received.'
                Y[idx] = item[xyz]
            idx += 1

        # Scale
        if 's' in item:
            assert -2 <= item['s'] <= 2, f'Transformation scale parameter must be in [-2, 2]. {item["s"]} received.'
            Y[idx] = item['s']
        idx += 1

        # Rotation
        assert ds_args.preangles_mode in [PREANGLES_MODE_QUATERNION, PREANGLES_MODE_AXISANGLE], \
            'Only quaternion or axis-angle pre-angles are supported.'
        if ds_args.preangles_mode == PREANGLES_MODE_QUATERNION:
            q = Y[idx:idx + 4]
            for i, k in enumerate(ds.labels_transformation_quaternion):
                if k in item:
                    if k != 'rw':
                        assert -1 <= item[k] <= 1, f'Rotation parameter {k} must be in [-1, 1]. {item[k]} received.'
                    q[i] = item[k]
            Y[idx:idx + 4] = normalise(q)
            idx += 4
        else:
            v = Y[idx:idx + 3]
            for i, k in enumerate(ds.labels_transformation_axisangle):
                v[i] = item[k]
            Y[idx:idx + 3] = v
            idx += 3

    # Material parameters are z-score standardised
    if ds_args.train_material:
        labels_material = ds.labels_material
        if ds_args.include_roughness:
            labels_material += ['r']
        for k in labels_material:
            if k in item:
                assert -1 <= item[k] <= 1, f'Material parameter {k} must be in [-2, 2]. {item[k]} received.'
                Y[idx] = item[k]
            idx += 1

    if ds_args.train_light:
        # Position
        for xyz in ['lx', 'ly', 'lz']:
            if xyz in item:
                assert -1 <= item[xyz] <= 1, f'Light parameter {xyz} must be in [-1, 1]. {item[xyz]} received.'
                Y[idx] = item[xyz]
            idx += 1

        # Energy
        if 'e' in item:
            assert -2 <= item['e'] <= 2, f'Light parameter energy must be in [-2, 2]. {item["e"]} received.'
            Y[idx] = item['e']
        idx += 1

        # Rotation
        assert ds_args.preangles_mode in [PREANGLES_MODE_QUATERNION, PREANGLES_MODE_AXISANGLE], \
            'Only quaternion or axis-angle pre-angles are supported.'
        if ds_args.preangles_mode == PREANGLES_MODE_QUATERNION:
            q = Y[idx:idx + 4]
            for i, k in enumerate(ds.labels_light_quaternion):
                if k in item:
                    if k != 'lrw':
                        assert -1 <= item[k] <= 1, \
                            f'Light rotation parameter {k} must be in [-1, 1]. {item[k]} received.'
                    q[i] = item[k]
            Y[idx:idx + 4] = normalise(q)
            idx += 4
        else:
            v = Y[idx:idx + 3]
            for i, k in enumerate(ds.labels_transformation_axisangle):
                v[i] = item[k]
            Y[idx:idx + 3] = v
            idx += 3

    return Y


def _make_default_parameter_vector(
        manager: Manager,
        use_zero_latent: bool = False
) -> torch.Tensor:
    """
    Build a parameter vector.
    """
    if use_zero_latent:
        Z0 = torch.zeros((1, manager.transcoder_args.tc_latent_size), device=manager.device)
        Y0 = manager.transcoder.to_parameters(Z0)[0]
        if manager.dataset_args.use_distance_switches:
            s0 = manager.ds.labels.index('ds0')
            s1 = s0 + len(manager.ds.labels_distance_switches)
            Y0[s0:s1] = torch.sigmoid(Y0[s0:s1])
    else:
        item = {
            'd4_111': 1,
            'ds4': 1,
            'rw': 1,
            'lrw': 1
        }
        Y0 = _make_parameter_vector(manager, item)

    return Y0


def generate_sweeps():
    """
    Generate outputs from parameter sweeps using a trained model.
    """
    args = parse_arguments()

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Create an output directory
    save_dir = LOGS_PATH / 'sweeps' / f'{START_TIMESTAMP}_{",".join(args.vary_parameters)}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments to json file
    with open(save_dir / 'options.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    # Instantiate the manager from the checkpoint json path
    manager = Manager.load(
        model_path=args.model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': args.use_gpu,
                'batch_size': args.batch_size
            },
        },
        save_dir=save_dir
    )

    # Put networks in eval mode
    manager.predictor.eval()
    manager.generator.eval()
    if manager.transcoder is not None:
        manager.transcoder.eval()

    # Make batches of varied parameters for each range and plot the results
    Y0 = _make_default_parameter_vector(manager, use_zero_latent=True)
    for k in args.vary_parameters:
        logger.info(f'Varying parameter: {k}')

        # Prepare the new batch of parameter vectors
        if k in ['si', 'il'] or k.startswith('d'):
            v_min = 0
            v_max = 1
        elif k in ['x', 'y', 'z', 'rx', 'ry', 'rz', 'lx', 'ly', 'lz', 'lrx', 'lry', 'lrz']:
            v_min = -1
            v_max = 1
        elif k in ['s', 'rw', 'b', 'ior', 'r', 'e', 'lrw']:
            v_min = -2
            v_max = 2
        else:
            raise RuntimeError(f'Unknown parameter key: {k}')
        p_vals = torch.linspace(v_min, v_max, args.n_vary_steps)
        Y_all = torch.zeros(args.n_vary_steps, len(Y0))
        for i, v in enumerate(p_vals):
            Y_all[i] = _make_parameter_vector(manager, {k: v}, defaults=Y0)
        Y_all = Y_all.to(manager.device)

        # Process in batches
        logger.info('Generating images from parameter vectors.')
        Y_batches = torch.split(Y_all, args.batch_size)
        p_vals_batches = torch.split(p_vals, args.batch_size)
        for i, Y in enumerate(Y_batches):
            Y = manager.prepare_parameter_dict(Y, apply_sigmoid_to_switches=False)

            # Generate the images from the parameter vectors
            X = manager.generate(Y)

            # Plot the results
            plot_dir = save_dir / k
            plot_dir.mkdir(exist_ok=True)
            for j, v in enumerate(p_vals_batches[i]):
                fig = _plot_results(k, j, v, Y, X, manager)
                plt.savefig(plot_dir / f'{i * args.batch_size + j:06d}.png')
                plt.close(fig)

    # Print how long this took - split into hours, minutes, seconds
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'Finished in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.')


if __name__ == '__main__':
    generate_sweeps()
