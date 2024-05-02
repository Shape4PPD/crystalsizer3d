import gc
import time
from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from ccdc.morphology import VisualHabitMorphology
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from timm.optim import create_optimizer_v2
from torch.utils.data import default_collate
from torchvision.transforms.functional import center_crop, to_tensor
from trimesh import Trimesh

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.args.dataset_training_args import ROTATION_MODE_AXISANGLE, ROTATION_MODE_QUATERNION, \
    ROTATION_MODE_SINCOS
from crystalsizer3d.crystal_renderer import render_from_parameters
from crystalsizer3d.nn.manager import CCDC_AVAILABLE, Manager
from crystalsizer3d.util.convergence_detector import ConvergenceDetector
from crystalsizer3d.util.utils import equal_aspect_ratio, print_args, str2bool, to_dict, to_numpy

PARAM_MODES = ['spectral', 'latent', 'parameters']


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            model_path: Path,
            image_path: Path,
            use_gpu: bool = False,
            log_every_n_steps: int = 1,
            plot_every_n_steps: int = 1,
            plot_top_k: int = 4,

            working_image_size: int = 512,
            param_mode: str = 'spectral',
            noise_level: float = 0.1,
            batch_size: int = 4,
            max_iterations: int = 100,

            algorithm: str = 'adam',
            weight_decay: float = 0.,
            lr: float = 1e-2,
            lrs_decay: float = 0.9,
            lrs_patience: int = 100,
            lrs_threshold: float = 0,
            lrs_burn_in: int = 200,
            convergence_tau_fast: int = 10,
            convergence_tau_slow: int = 100,
            convergence_threshold: float = 0.1,
            convergence_patience: int = 25,
            convergence_loss_target: float = 50.,

            **kwargs
    ):
        assert model_path.exists(), f'Dataset path does not exist: {model_path}'
        assert model_path.suffix == '.json', f'Model path must be a json file: {model_path}'
        self.model_path = model_path
        assert image_path.exists(), f'Image path does not exist: {image_path}'
        self.image_path = image_path
        self.use_gpu = use_gpu
        self.log_every_n_steps = log_every_n_steps
        self.plot_every_n_steps = plot_every_n_steps
        self.plot_top_k = plot_top_k

        # Model set up
        assert working_image_size > 64, f'Working image size must be > 64: {working_image_size}'
        self.working_image_size = working_image_size
        assert param_mode in PARAM_MODES, f'Parameter mode must be one of: {PARAM_MODES}'
        self.param_mode = param_mode
        self.noise_level = noise_level
        self.batch_size = batch_size
        self.max_iterations = max_iterations

        # Optimisation parameters
        self.algorithm = algorithm
        self.weight_decay = weight_decay
        self.lr = lr
        self.lrs_decay = lrs_decay
        self.lrs_patience = lrs_patience
        self.lrs_threshold = lrs_threshold
        self.lrs_burn_in = lrs_burn_in
        self.convergence_tau_fast = convergence_tau_fast
        self.convergence_tau_slow = convergence_tau_slow
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.convergence_loss_target = convergence_loss_target

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Runtime Args')
        group.add_argument('--model-path', type=Path, required=True,
                           help='Path to the model\'s json file.')
        group.add_argument('--image-path', type=Path, required=True,
                           help='Path to the image to process.')
        parser.add_argument('--use-gpu', type=str2bool, default=USE_CUDA,
                            help='Use GPU. Defaults to environment setting.')
        parser.add_argument('--log-every-n-steps', type=int, default=1,
                            help='Log every n steps.')
        parser.add_argument('--plot-every-n-steps', type=int, default=1,
                            help='Plot every n steps.')
        parser.add_argument('--plot-top-k', type=int, default=4,
                            help='Plot the top k solutions.')

        # Model
        parser.add_argument('--working-image-size', type=int, default=512,
                            help='Size of the working image. -1 to use the input image size.')
        parser.add_argument('--param-mode', type=str, default='spectral', choices=PARAM_MODES,
                            help='Parameter mode. Default=spectral.')
        parser.add_argument('--noise-level', type=float, default=0.1,
                            help='Either the Gaussian noise variance added to the latent/parameter space if param mode is "latent" or "parameters" '
                                 'or the variance multiplier for the latent noise in spectral space.')
        parser.add_argument('--batch-size', type=int, default=4,
                            help='Batch size.')
        parser.add_argument('--max-iterations', type=int, default=100,
                            help='Maximum number of iterations to run for.')

        # Optimisation parameters
        parser.add_argument('--algorithm', type=str, default='adam',
                            help='Optimisation algorithm. Default=adam.')
        parser.add_argument('--weight-decay', type=float, default=0.,
                            help='Weight decay (in the spectral parameter space).')
        parser.add_argument('--lr', type=float, default=1e-2,
                            help='Learning rate.')
        parser.add_argument('--lrs-decay', type=float, default=0.9,
                            help='Learning rate scheduler decay.')
        parser.add_argument('--lrs-patience', type=int, default=100,
                            help='Learning rate scheduler patience.')
        parser.add_argument('--lrs-threshold', type=float, default=0,
                            help='Learning rate scheduler threshold.')
        parser.add_argument('--lrs-burn-in', type=int, default=200,
                            help='Learning rate scheduler burn-in.')
        parser.add_argument('--convergence-tau-fast', type=int, default=10,
                            help='Convergence detector tau fast.')
        parser.add_argument('--convergence-tau-slow', type=int, default=100,
                            help='Convergence detector tau slow.')
        parser.add_argument('--convergence-threshold', type=float, default=0.1,
                            help='Convergence detector threshold.')
        parser.add_argument('--convergence-patience', type=int, default=25,
                            help='Convergence detector patience.')
        parser.add_argument('--convergence-loss-target', type=float, default=50.,
                            help='Convergence detector loss target.')

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
        _, _, mesh = manager.crystal_generator.generate_crystal(rel_distances=distances,
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
    equal_aspect_ratio(ax, zoom=1.25)
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
        # ax.legend()
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
    if manager.dataset_args.rotation_mode == ROTATION_MODE_SINCOS:
        xlabels += manager.ds.labels_transformation_sincos
    elif manager.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
        xlabels += manager.ds.labels_transformation_quaternion
    else:
        assert manager.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE
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
    if manager.dataset_args.rotation_mode == ROTATION_MODE_SINCOS:
        xlabels += manager.ds.labels_light_sincos
    elif manager.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
        xlabels += manager.ds.labels_light_quaternion
    else:
        assert manager.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE
        xlabels += manager.ds.labels_light_axisangle
    ax.set_xticklabels(xlabels)
    if 'light' not in share_ax:
        # ax_.legend()
        share_ax['light'] = ax
    else:
        ax.sharey(share_ax['light'])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.autoscale()


def _plot_single_prediction(
        manager: Manager,
        X_target: torch.Tensor,
        Y_pred: Dict[str, torch.Tensor],
        X_pred: torch.Tensor,
        idx: int = 0
):
    """
    Plot the image and parameter predictions.
    """
    ds_args = manager.dataset_args
    n_rows = 3 \
             + int(ds_args.train_distances) \
             + int(ds_args.train_transformation) \
             + int(ds_args.train_material) \
             + int(ds_args.train_light) \
             + int(ds_args.train_generator)

    height_ratios = [1.4, 1.4, 1]  # images and 3d plot
    if ds_args.train_distances:
        height_ratios.append(1)
    if ds_args.train_transformation:
        height_ratios.append(0.7)
    if ds_args.train_material:
        height_ratios.append(0.7)
    if ds_args.train_light:
        height_ratios.append(0.7)
    if ds_args.train_generator:
        height_ratios.insert(0, 1.4)

    fig = plt.figure(figsize=(4, n_rows * 2.4))
    gs = GridSpec(
        nrows=n_rows,
        ncols=1,
        wspace=0.06,
        hspace=0.4,
        width_ratios=[1],
        height_ratios=height_ratios,
        top=0.97,
        bottom=0.02,
        left=0.13,
        right=0.87
    )

    fig.suptitle(
        f'Predictions',
        fontweight='bold',
        y=0.995
    )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    row_idx = 0

    # Plot the input image
    img = to_numpy(X_target[idx]).squeeze()
    _plot_image(fig.add_subplot(gs[row_idx, 0]), 'Input', img)
    row_idx += 1

    # Plot the generated image
    if ds_args.train_generator:
        img = to_numpy(X_pred[idx]).squeeze()
        _plot_image(fig.add_subplot(gs[row_idx, 0]), 'Generated', img)
        row_idx += 1

    # Plot the 3D morphology
    mesh_pred = None
    try:
        morph_pred, mesh_pred = _build_crystal(Y_pred, idx, manager)
        _plot_3d(fig.add_subplot(gs[row_idx + 1, 0], projection='3d'), 'Predicted',
                 morph_pred, mesh_pred, default_colours[1])
    except Exception as e:
        _plot_error(fig.add_subplot(gs[row_idx + 1, 0]), str(e))
        row_idx += 1

    # Render the crystal
    if mesh_pred is not None:
        try:
            img = render_from_parameters(
                mesh=mesh_pred,
                settings_path=manager.ds.path / 'vcw_settings.json',
                r_params=manager.ds.denormalise_rendering_params(Y_pred, 0),
                attempts=1
            )
            _plot_image(fig.add_subplot(gs[row_idx, 0]), 'Render', img)
        except Exception as e:
            _plot_error(fig.add_subplot(gs[row_idx, 0]), f'Rendering failed:\n{e}')
    else:
        _plot_error(fig.add_subplot(gs[row_idx, 0]), 'No crystal to render.')

    row_idx += 2
    if ds_args.train_distances:
        _plot_distances(fig.add_subplot(gs[row_idx, 0]), Y_pred, idx, manager)
        row_idx += 1
    if ds_args.train_transformation:
        _plot_transformation(fig.add_subplot(gs[row_idx, 0]), Y_pred, idx, manager)
        row_idx += 1
    if ds_args.train_material:
        _plot_material(fig.add_subplot(gs[row_idx, 0]), Y_pred, idx)
        row_idx += 1
    if ds_args.train_light:
        _plot_light(fig.add_subplot(gs[row_idx, 0]), Y_pred, idx, manager)

    plt.show()


def _plot_batch_prediction(
        manager: Manager,
        step: int,
        lr: float,
        X_target: torch.Tensor,
        P: torch.Tensor,
        Y_pred: Dict[str, torch.Tensor],
        X_pred: torch.Tensor,
        losses: torch.Tensor,
        top_k: int = 3
) -> Figure:
    """
    Plot the image and parameter predictions for a batch of solutions.
    """
    # Sort the batch by loss
    losses = to_numpy(losses)
    idxs_by_rank = np.argsort(losses)
    idxs = idxs_by_rank[:top_k]
    n_cols = 2 + len(idxs)
    n_rows = 4

    plt.rc('axes', labelsize=10)  # fontsize of the axes labels
    plt.rc('axes', titlesize=12, titlepad=4)  # fontsize of the axes labels
    plt.rc('xtick', labelsize=10)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=10)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=2, size=2)
    plt.rc('ytick.major', pad=2, size=2)

    fig = plt.figure(figsize=(2 * n_cols, n_rows * 2))
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_cols,
        wspace=0.08,
        hspace=0.7,
        width_ratios=[1, 0.15] + [0.7, ] * len(idxs),
        height_ratios=[1, 0.7, 0.7, 0.7],
        top=0.95,
        bottom=0.07,
        left=0.06,
        right=0.99
    )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    share_ax = {}

    # Plot the input image - same for all batch items
    img = to_numpy(X_target[0]).squeeze()
    ax_img = fig.add_subplot(gs[0:2, 0])
    _plot_image(ax_img, None, img)
    ax_img.text(-0.1, 1.02, f'Step: {step}\nlr: {lr:.3E}', ha='left', va='bottom',
                fontsize=13, linespacing=1.3, fontweight='bold', transform=ax_img.transAxes)

    # Plot the (sorted) batch losses as a bar chart
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title('Batch errors')
    ax.grid(zorder=-10)
    losses_ranked = np.sort(losses)
    locs = np.arange(len(losses))
    bar_width = 0.7
    ax.bar(locs, losses_ranked, bar_width, zorder=10)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Solution #')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())

    # Find the clustering of solutions by calculating L2 distances and clustering the matrix
    P = to_numpy(P)
    dists = pdist(P)
    dists_sf = squareform(dists)
    L = linkage(dists, 'average', 'euclidean', optimal_ordering=True)
    clusters = fcluster(L, len(P), criterion='maxclust')
    block_idx = 0
    cluster_nums = np.unique(clusters)
    dists_reordered = np.zeros_like(dists_sf)
    all_idxs = []
    for cluster_num in cluster_nums:
        idxs_in_cluster = (clusters == cluster_num).nonzero()[0]
        cluster_size = len(idxs_in_cluster)
        dists_reordered[block_idx:(block_idx + cluster_size)] = dists_sf[idxs_in_cluster].copy()
        all_idxs.append(idxs_in_cluster)
        block_idx += cluster_size
    all_idxs = np.concatenate(all_idxs)
    dists_reordered[:, :] = dists_reordered[:, all_idxs]  # Reorder columns

    # Plot the reordered distance matrix
    ax = fig.add_subplot(gs[3, 0])
    ax.set_title('Solution distances')
    mat = ax.imshow(dists_reordered, cmap='Purples', aspect='auto')
    fig.colorbar(mat, format='%.2E', location='left', shrink=0.7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot the solution batch
    col_idx = 2
    for i, idx in enumerate(idxs):
        # Plot the generated image
        img_i = to_numpy(X_pred[idx]).squeeze()
        _plot_image(fig.add_subplot(gs[0, col_idx]), f'$\epsilon=${losses[idx]:.3E}', img_i)

        # Plot the 3D morphology
        try:
            morph_pred, mesh_pred = _build_crystal(Y_pred, idx, manager)
            _plot_3d(fig.add_subplot(gs[1, col_idx], projection='3d'), None,
                     morph_pred, mesh_pred, default_colours[1])
        except Exception:
            _plot_error(fig.add_subplot(gs[1, col_idx]), 'Failed to build crystal.')

        # Plot the parameters
        _plot_distances(fig.add_subplot(gs[2, col_idx]), Y_pred, idx, manager, share_ax)
        _plot_transformation(fig.add_subplot(gs[3, col_idx]), Y_pred, idx, manager, share_ax)

        col_idx += 1

    return fig


def predict():
    """
    Trains a network to estimate crystal growth parameters from images.
    """
    args = parse_arguments()

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Create an output directory
    save_dir = LOGS_PATH / 'single' / f'{args.image_path.name}_{START_TIMESTAMP}'
    plot_dir = save_dir / 'optimisation'
    plot_dir.mkdir(parents=True, exist_ok=True)

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

    # Load the target image
    img = Image.open(args.image_path).convert('L')
    img = to_tensor(img)
    X_target = default_collate([img, ])
    X_target = X_target.to(manager.device)

    # Crop and resize the image to the working image size
    X_target = center_crop(X_target, min(X_target.shape[-2:]))
    X_target = F.interpolate(
        X_target,
        size=manager.image_shape[-1],
        mode='bilinear',
        align_corners=False
    )

    # Get the PCA representation of the last mapping from latent to parameter space
    pca = PCA()
    pca.fit(to_numpy(manager.predictor.last_op.weight))
    pca_components = torch.from_numpy(pca.components_).to(manager.device)
    pca_mean = torch.from_numpy(pca.mean_).to(manager.device)
    pca_var = torch.from_numpy(pca.explained_variance_).to(manager.device)

    def transform(x):
        return torch.einsum('bi,ij->bj', x - pca_mean, pca_components.T)

    def inverse_transform(x):
        return torch.einsum('bj,ji->bi', x, pca_components) + pca_mean

    # Get the initial solution
    Y0 = manager.predict(X_target)
    if manager.transcoder_args.use_transcoder:
        Z0 = manager.transcoder.latents_in
    else:
        Z0 = manager.predictor.latent_state

    if args.param_mode == 'spectral':
        # Map the latent to the spectral embedding and add noise in the spectral space to give the initial batch
        Z0s = transform(Z0)
        Zs = Z0s.repeat(args.batch_size, 1)
        Zs += torch.normal(
            mean=torch.zeros_like(Zs),
            std=torch.sqrt(pca_var * args.noise_level)
        )
        Zs[0] = Z0s
        P = Zs
        Z = inverse_transform(Zs)
    elif args.param_mode == 'latent':
        # Add gaussian noise directly to the latent space to give the initial batch
        Z = Z0.repeat(args.batch_size, 1)
        Z += torch.randn_like(Z) * args.noise_level
        Z[0] = Z0
        P = Z
    else:
        # Add gaussian noise directly to the parameter space to give the initial batch
        Y = Y0.repeat(args.batch_size, 1)
        Y += torch.randn_like(Y) * args.noise_level
        Y[0] = Y0
        P = Y

    # Calculate the first batch of predictions
    if args.param_mode != 'parameters':
        Y = manager.predict(Z, latent_input=True)
    X_pred = manager.generate(Y)
    batch_losses = ((X_pred - X_target)**2).mean(dim=(-2, -1))[:, 0]

    # Plot the initial solution batch
    fig = _plot_batch_prediction(
        manager=manager,
        step=0,
        lr=args.lr,
        X_target=X_target,
        P=P,
        Y_pred=Y,
        X_pred=X_pred,
        losses=batch_losses,
        top_k=args.plot_top_k
    )
    plt.savefig(plot_dir / f'{0:06d}.png')
    plt.close(fig)

    # Set up parameters and optimiser
    P = torch.nn.Parameter(P, requires_grad=True)

    # Create the optimiser
    optimiser = create_optimizer_v2(
        model_or_params=[P, ],
        opt=args.algorithm,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=args.lrs_patience,
        factor=args.lrs_decay,
        threshold=args.lrs_threshold
    )

    # Set up the convergence detector
    convergence_detector = ConvergenceDetector(
        shape=(args.batch_size,),
        tau_fast=args.convergence_tau_fast,
        tau_slow=args.convergence_tau_slow,
        threshold=args.convergence_threshold,
        patience=args.convergence_patience
    )
    convergence_detector.to(manager.device)

    # Run optimisation
    logger.info('Starting optimisation.')
    losses = []
    for i in range(args.max_iterations):
        optimiser.zero_grad()

        # Generate the images from the parameters/latent vectors
        if args.param_mode != 'parameters':
            if args.param_mode == 'spectral':
                Z = inverse_transform(P)
            else:
                Z = P
            Y = manager.predict(Z, latent_input=True)
        else:
            Y = P
        X_pred = manager.generate(Y)

        # Calculate image losses
        # batch_losses = ((X_pred - X_target)**2).mean(dim=(-2, -1))[:, 0]
        batch_losses = ((X_pred - X_target).abs()).mean(dim=(-2, -1))[:, 0]
        loss = batch_losses.mean()
        losses.append(loss.item())

        # Gradient descent step
        loss.backward()
        optimiser.step()
        if i > args.lrs_burn_in:
            scheduler.step(loss)
        lr_i = optimiser.param_groups[0]['lr']

        # Log progress
        if (i + 1) % args.log_every_n_steps == 0:
            logger.info(
                f'Iteration: {i + 1} \t '
                f'Loss: {loss.item():.3E} \t '
                f'Learning rate: {lr_i:.3E}'
            )

        # Plot progress
        if args.plot_every_n_steps != -1 and (i + 1) % args.plot_every_n_steps == 0:
            fig = _plot_batch_prediction(
                manager=manager,
                step=i + 1,
                lr=lr_i,
                X_target=X_target,
                P=P,
                Y_pred=Y,
                X_pred=X_pred,
                losses=batch_losses,
                top_k=args.plot_top_k
            )
            plt.savefig(plot_dir / f'{i + 1:06d}.png')
            plt.close(fig)

        # Check for convergence
        convergence_detector.forward(batch_losses, first_val=i == 0)
        if convergence_detector.converged.all():
            logger.info(f'Converged after {i + 1} iterations.')
            break

        # Try to stop any memory leaks!
        if i % 10:
            gc.collect()

    # Plot final
    if args.plot_every_n_steps != -1 and (i + 1) % args.plot_every_n_steps != 0:
        fig = _plot_batch_prediction(
            manager=manager,
            step=i + 1,
            lr=lr_i,
            X_target=X_target,
            P=P,
            Y_pred=Y,
            X_pred=X_pred,
            losses=batch_losses,
            top_k=args.plot_top_k
        )
        plt.savefig(plot_dir / f'{i + 1:06d}.png')
        plt.close(fig)

    # Print how long this took - split into hours, minutes, seconds
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'Finished in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.')


if __name__ == '__main__':
    predict()
