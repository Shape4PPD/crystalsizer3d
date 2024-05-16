from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mayavi import mlab
from scipy.spatial.transform import Rotation

from crystalsizer3d import USE_MLAB
from crystalsizer3d.crystal import Crystal, ROTATION_MODE_QUATERNION
from crystalsizer3d.util.utils import equal_aspect_ratio, to_numpy, to_rgb

if TYPE_CHECKING:
    from crystalsizer3d.nn.manager import Manager

# Off-screen rendering
mlab.options.offscreen = True


def get_ax_size(ax: Axes) -> Tuple[float, float]:
    """
    Get the size of the axis in pixels.
    """
    fig = ax.get_figure()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


def plot_error(ax: Axes, err: str):
    """
    Plot an error message on the axis.
    """
    txt = ax.text(
        0.5, 0.5, err,
        horizontalalignment='center',
        verticalalignment='center',
        wrap=True
    )
    txt._get_wrap_line_width = lambda: ax.bbox.width * 0.7
    ax.axis('off')


def plot_image(ax: Axes, title: str, img: np.ndarray):
    """
    Plot an image on the axis.
    """
    ax.set_title(title)
    img = img.squeeze()
    if img.ndim == 2:
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    else:
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        ax.imshow(img)
    ax.axis('off')


def add_discriminator_value(ax: Axes, outputs: Dict[str, torch.Tensor], D_key: str, idx: int):
    """
    Add the discriminator value to the axis.
    """
    if D_key in outputs:
        d_val = outputs[D_key][idx].item()
        colour = 'red' if d_val < 0 else 'green'
        ax.text(
            0.99, -0.02, f'{d_val:.3E}',
            ha='right', va='top', transform=ax.transAxes,
            color=colour, fontsize=14
        )


def make_3d_digital_crystal_image(
        crystal: Crystal,
        res: int = 100,
        bg_col: float = 1.,
        wireframe_radius_factor: float = 0.1,
        surface_colour: str = 'skyblue',
        wireframe_colour: str = 'darkblue',
        opacity: float = 0.6,
        azim: float = 150,
        elev: float = 160,
        distance: Optional[float] = None,
        roll: float = 0
):
    """
    Make a 3D image of the crystal.
    """
    fig = mlab.figure(size=(res * 2, res * 2), bgcolor=(bg_col, bg_col, bg_col))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Add crystal mesh
    origin = crystal.origin.clone()
    crystal.origin.data = torch.zeros_like(origin)
    v, f = crystal.build_mesh()
    v, f = to_numpy(v), to_numpy(f)
    mlab.triangular_mesh(*v.T, f, figure=fig, color=to_rgb(surface_colour), opacity=opacity)
    for fv_idxs in crystal.faces.values():
        fv = to_numpy(crystal.vertices[fv_idxs])
        fv = np.vstack([fv, fv[0]])  # Close the loop
        mlab.plot3d(*fv.T, color=to_rgb(wireframe_colour),
                    tube_radius=crystal.distances[0].item() * wireframe_radius_factor)
    crystal.origin.data = origin

    # Render
    mlab.view(figure=fig, azimuth=azim, elevation=elev, distance=distance, roll=roll, focalpoint=np.zeros(3))

    # # Useful for getting the view parameters when recording from the gui:
    # mlab.show()
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [-1.0976718374293786, 0.5730634321110751, -4.126732879628852]
    # scene.scene.camera.focal_point = [0.0, 0.0, -1.862645149230957e-09]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [0.47541220253052663, -0.8452964994640355, -0.24383819569321213]
    # scene.scene.camera.clipping_range = [3.460626768194046, 5.386199822951965]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    # mlab.show()
    # exit()

    # fig.scene.render()
    image = mlab.screenshot(mode='rgb', antialiased=True, figure=fig)
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mlab.close()

    return image


def plot_3d(
        ax: Axes,
        title: str,
        crystal: Crystal,
        target_or_pred: str
):
    """
    Plot a 3d mesh on the axis.
    """
    ax.set_title(title)

    surf_col = 'orange' if target_or_pred == 'target' else 'skyblue'
    wire_col = 'red' if target_or_pred == 'target' else 'darkblue'

    if USE_MLAB:
        image = make_3d_digital_crystal_image(
            crystal,
            res=int(max(get_ax_size(ax)) * 2),
            surface_colour=surf_col,
            wireframe_colour=wire_col
        )
        ax.imshow(image)
        ax.axis('off')
    else:
        origin = crystal.origin.clone()
        crystal.origin.data = torch.zeros_like(origin)
        v, f = crystal.build_mesh()
        v, f = to_numpy(v), to_numpy(f)
        ax.plot_trisurf(
            v[:, 0],
            v[:, 1],
            triangles=f,
            Z=v[:, 2],
            color=surf_col,
            alpha=0.5
        )
        for fv_idxs in crystal.faces.values():
            fv = to_numpy(crystal.vertices[fv_idxs])
            fv = np.vstack([fv, fv[0]])  # Close the loop
            ax.plot(*fv.T, c=wire_col)
        crystal.origin.data = origin
        equal_aspect_ratio(ax)


def plot_zingg(
        ax: Axes,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Dict[str, torch.Tensor],
        idx: int,
        share_ax: Dict[str, Axes],
        **kwargs
):
    """
    Plot the Zingg diagram on the axis.
    """
    z_pred = to_numpy(Y_pred['zingg'][idx])
    z_target = to_numpy(Y_target['zingg'][idx])
    ax.scatter(z_target[0], z_target[1], c='r', marker='x', s=100, label='Target')
    ax.scatter(z_pred[0], z_pred[1], c='b', marker='o', s=100, label='Predicted')
    ax.set_title('Zingg')
    ax.set_xlabel('S/I', labelpad=-5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    if 'zingg' not in share_ax:
        ax.legend()
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_ylabel('I/L', labelpad=0)
        share_ax['zingg'] = ax
    else:
        ax.sharey(share_ax['zingg'])
        ax.yaxis.set_tick_params(labelleft=False)


def plot_distances(
        ax: Axes,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Dict[str, torch.Tensor],
        Y_pred2: Dict[str, torch.Tensor],
        idx: int,
        share_ax: Dict[str, Axes],
        manager: Manager
):
    """
    Plot the distances on the axis.
    """
    ax_pos = ax.get_position()
    ax.set_position([ax_pos.x0, ax_pos.y0 + 0.02, ax_pos.width, ax_pos.height - 0.02])
    d_pred = to_numpy(Y_pred['distances'][idx])
    d_target = to_numpy(Y_target['distances'][idx])
    if manager.dataset_args.train_generator:
        d_pred2 = to_numpy(Y_pred2['distances'][idx])

    # Clip predictions to -1 to avoid large negatives skewing the plots
    d_pred = np.clip(d_pred, a_min=-1, a_max=np.inf)

    locs = np.arange(len(d_target))
    if manager.dataset_args.train_generator:
        bar_width = 0.25
        offset = bar_width
        ax.bar(locs - offset, d_target, bar_width, label='Target')
        ax.bar(locs, d_pred, bar_width, label='Predicted')
        ax.bar(locs + offset, d_pred2, bar_width, label='Predicted2')
    else:
        bar_width = 0.35
        offset = bar_width / 2
        ax.bar(locs - offset, d_target, bar_width, label='Target')
        ax.bar(locs + offset, d_pred, bar_width, label='Predicted')

    if manager.dataset_args.use_distance_switches:
        s_pred = to_numpy(Y_pred['distance_switches'][idx])
        s_target = to_numpy(Y_target['distance_switches'][idx])
        k = 2.3
        colours = []
        for i, (sp, st) in enumerate(zip(s_pred, s_target)):
            if st > 0.5:
                ax.axvspan(i - k * offset, i, alpha=0.1, color='blue')
            if sp > 0.5:
                ax.axvspan(i, i + k * offset, alpha=0.1, color='red' if st < 0.5 else 'green')
            colours.append('red' if (st < 0.5 < sp) or (st > 0.5 > sp) else 'green')
        ax.scatter(locs + offset, s_pred, color=colours, marker='+', s=100, label='Switches')

    ax.set_title('Distances')
    ax.set_xticks(locs)
    ax.set_xticklabels(manager.ds.labels_distances_active)
    ax.tick_params(axis='x', rotation=270)
    if 'distances' not in share_ax:
        ax.legend()
        share_ax['distances'] = ax
    else:
        ax.sharey(share_ax['distances'])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.autoscale()


def plot_transformation(
        ax: Axes,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Dict[str, torch.Tensor],
        Y_pred2: Dict[str, torch.Tensor],
        idx: int,
        share_ax: Dict[str, Axes],
        manager: Manager
):
    """
    Plot the transformation parameters.
    """
    t_pred = to_numpy(Y_pred['transformation'][idx])
    t_target = to_numpy(Y_target['transformation'][idx])
    if manager.dataset_args.train_generator:
        t_pred2 = to_numpy(Y_pred2['transformation'][idx])

    # Adjust the target rotation to the best matching symmetry group
    sgi = manager.sym_group_idxs[idx]
    if manager.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
        t_target[4:] = to_numpy(Y_target['sym_rotations'][idx][sgi])
    else:
        R = to_numpy(Y_target['sym_rotations'][idx][sgi])
        t_target[4:] = Rotation.from_matrix(R).as_rotvec()

    locs = np.arange(len(t_target))

    if manager.dataset_args.train_generator:
        bar_width = 0.25
        offset = bar_width
        ax.bar(locs - offset, t_target, bar_width, label='Target')
        ax.bar(locs, t_pred, bar_width, label='Predicted')
        ax.bar(locs + offset, t_pred2, bar_width, label='Predicted2')
        k = 2 * offset
    else:
        bar_width = 0.35
        offset = bar_width / 2
        ax.bar(locs - offset, t_target, bar_width, label='Target')
        ax.bar(locs + offset, t_pred, bar_width, label='Predicted')
        k = 3 * offset
    ax.axvspan(locs[0] - k, locs[2] + k, alpha=0.1, color='green')
    ax.axvspan(locs[3] - k, locs[3] + k, alpha=0.1, color='red')
    ax.axvspan(locs[4] - k, locs[-1] + k, alpha=0.1, color='blue')
    ax.set_title('Transformation')
    ax.set_xticks(locs)
    xlabels = manager.ds.labels_transformation.copy()
    if manager.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
        xlabels += manager.ds.labels_rotation_quaternion
    else:
        xlabels += manager.ds.labels_rotation_axisangle
    ax.set_xticklabels(xlabels)
    if 'transformation' not in share_ax:
        ax.legend()
        share_ax['transformation'] = ax
    else:
        ax.sharey(share_ax['transformation'])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.autoscale()


def plot_material(
        ax: Axes,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Dict[str, torch.Tensor],
        Y_pred2: Dict[str, torch.Tensor],
        idx: int,
        share_ax: Dict[str, Axes],
        manager: Manager
):
    """
    Plot the material parameters.
    """
    m_pred = to_numpy(Y_pred['material'][idx])
    m_target = to_numpy(Y_target['material'][idx])
    if manager.dataset_args.train_generator:
        m_pred2 = to_numpy(Y_pred2['material'][idx])

    locs = np.arange(len(m_target))
    if manager.dataset_args.train_generator:
        bar_width = 0.25
        offset = bar_width
        ax.bar(locs - offset, m_target, bar_width, label='Target')
        ax.bar(locs, m_pred, bar_width, label='Predicted')
        ax.bar(locs + offset, m_pred2, bar_width, label='Predicted2')
    else:
        bar_width = 0.35
        offset = bar_width / 2
        ax.bar(locs - offset, m_target, bar_width, label='Target')
        ax.bar(locs + offset, m_pred, bar_width, label='Predicted')
    ax.set_title('Material')
    ax.set_xticks(locs)
    for i in range(len(locs) - 1):
        ax.axvline(locs[i] + .5, color='black', linestyle='--', linewidth=1)
    labels = []
    if 'ior' in manager.ds.labels_material_active:
        labels.append('IOR')
    if 'r' in manager.ds.labels_material_active:
        labels.append('Roughness')
    ax.set_xticklabels(labels)
    if 'material' not in share_ax:
        ax.legend()
        share_ax['material'] = ax
    else:
        ax.sharey(share_ax['material'])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.autoscale()


def plot_light(
        ax: Axes,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Dict[str, torch.Tensor],
        idx: int,
        share_ax: Dict[str, Axes],
        manager: Manager,
        **kwargs
):
    """
    Plot the light parameters.
    """
    l_pred = to_numpy(Y_pred['light'][idx])
    l_target = to_numpy(Y_target['light'][idx])
    locs = np.arange(len(l_target))
    bar_width = 0.35
    offset = bar_width / 2
    ax.bar(locs - offset, l_target, bar_width, label='Target')
    ax.bar(locs + offset, l_pred, bar_width, label='Predicted')
    ax.set_title('Light')
    ax.set_xticks(locs)

    xlabels = manager.ds.labels_light.copy()
    ax.set_xticklabels(xlabels)
    if 'light' not in share_ax:
        ax.legend()
        share_ax['light'] = ax
    else:
        ax.sharey(share_ax['light'])
        ax.yaxis.set_tick_params(labelleft=False)
        ax.autoscale()


def plot_training_samples(
        manager: Manager,
        data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
        outputs: Dict[str, Any],
        train_or_test: str,
        idxs: List[int],
) -> Figure:
    """
    Plot the image and parameter comparisons.
    """
    n_examples = len(idxs)
    metas, images, images_aug, Y_target = data
    Y_pred = outputs['Y_pred']
    if manager.dataset_args.train_generator:
        X_pred = outputs['X_pred']
        X_pred2 = outputs['X_pred2']
        Y_pred2 = outputs['Y_pred2']
    else:
        X_pred = None
        X_pred2 = None
        Y_pred2 = None
    n_rows = 4 \
             + int(manager.dataset_args.train_zingg) \
             + int(manager.dataset_args.train_distances) \
             + int(manager.dataset_args.train_transformation) \
             + int(manager.dataset_args.train_material and len(manager.ds.labels_material_active) > 0) \
             + int(manager.dataset_args.train_light) \
             + int(manager.dataset_args.train_generator) * 2

    height_ratios = [1.2, 1.2, 1, 1]  # images and 3d plots
    if manager.dataset_args.train_zingg:
        height_ratios.append(0.5)
    if manager.dataset_args.train_distances:
        height_ratios.append(1)
    if manager.dataset_args.train_transformation:
        height_ratios.append(0.7)
    if manager.dataset_args.train_material and len(manager.ds.labels_material_active) > 0:
        height_ratios.append(0.7)
    if manager.dataset_args.train_light:
        height_ratios.append(0.7)
    if manager.dataset_args.train_generator:
        height_ratios.insert(0, 1.2)
        height_ratios.insert(0, 1.2)

    fig = plt.figure(figsize=(n_examples * 2.6, n_rows * 2.4))
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_examples,
        wspace=0.06,
        hspace=0.4,
        width_ratios=[1] * n_examples,
        height_ratios=height_ratios,
        top=0.97,
        bottom=0.015,
        left=0.04,
        right=0.99
    )

    loss = getattr(manager.checkpoint, f'loss_{train_or_test}')
    fig.suptitle(
        f'epoch={manager.checkpoint.epoch}, '
        f'step={manager.checkpoint.step + 1}, '
        f'loss={loss:.4E}',
        fontweight='bold',
        y=0.995
    )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    share_ax = {}

    for i, idx in enumerate(idxs):
        meta = metas[idx]
        r_params_target = meta['rendering_parameters']
        r_params_pred = manager.ds.denormalise_rendering_params(Y_pred, idx, r_params_target)
        crystal_target = manager.ds.load_crystal(r_params=r_params_target, zero_origin=True)
        crystal_pred = manager.ds.load_crystal(r_params=r_params_pred, zero_origin=True)
        row_idx = 0

        # Plot the (possibly augmented) input image
        img = to_numpy(images_aug[idx]).squeeze()
        ax = fig.add_subplot(gs[row_idx, i])
        plot_image(ax, meta['image'].name, img)
        add_discriminator_value(ax, outputs, 'D_real', idx)
        row_idx += 1

        # Plot the rendering from the predicted parameters
        try:
            img = manager.crystal_renderer.render_from_parameters(r_params_pred)
            plot_image(fig.add_subplot(gs[row_idx, i]), 'Render', img)
        except Exception as e:
            plot_error(fig.add_subplot(gs[row_idx, i]), f'Rendering failed:\n{e}')
        row_idx += 1

        # Plot the generated image(s)
        if manager.dataset_args.train_generator:
            img = to_numpy(X_pred[idx]).squeeze()
            ax = fig.add_subplot(gs[row_idx, i])
            plot_image(ax, 'Generated', img)
            add_discriminator_value(ax, outputs, 'D_fake', idx)
            row_idx += 1
            img = to_numpy(X_pred2[idx]).squeeze()
            ax = fig.add_subplot(gs[row_idx, i])
            plot_image(ax, 'Generated2', img)
            row_idx += 1

        # Plot the 3d digital crystals
        projection_kwargs = {} if USE_MLAB else {'projection': '3d'}
        plot_3d(fig.add_subplot(gs[row_idx, i], **projection_kwargs), 'Morphology',
                crystal_target, 'target')
        row_idx += 1
        plot_3d(fig.add_subplot(gs[row_idx, i], **projection_kwargs), 'Predicted',
                crystal_pred, 'predicted')
        row_idx += 1

        # Plot the parameters
        shared_args = dict(
            Y_pred=Y_pred,
            Y_target=Y_target,
            Y_pred2=Y_pred2,
            idx=idx,
            share_ax=share_ax,
            manager=manager
        )
        if manager.dataset_args.train_zingg:
            plot_zingg(fig.add_subplot(gs[row_idx, i]), **shared_args)
            row_idx += 1
        if manager.dataset_args.train_distances:
            plot_distances(fig.add_subplot(gs[row_idx, i]), **shared_args)
            row_idx += 1
        if manager.dataset_args.train_transformation:
            plot_transformation(fig.add_subplot(gs[row_idx, i]), **shared_args)
            row_idx += 1
        if manager.dataset_args.train_material and len(manager.ds.labels_material_active) > 0:
            plot_material(fig.add_subplot(gs[row_idx, i]), **shared_args)
            row_idx += 1
        if manager.dataset_args.train_light:
            plot_light(fig.add_subplot(gs[row_idx, i]), **shared_args)

    return fig


def _plot_vaetc_examples(
        self,
        data: Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
        outputs: Dict[str, Any],
        train_or_test: str,
        idxs: np.ndarray
):
    """
    Plot some VAE transcoder examples.
    """
    n_examples = min(self.runtime_args.plot_n_examples, self.runtime_args.batch_size)
    metas, images, images_aug, params = data
    Yr_noisy = outputs['Yr_mu']
    Yr_clean = outputs['Yr_mu_clean']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    n_rows = int(self.dataset_args.train_zingg) \
             + int(self.dataset_args.train_distances) \
             + int(self.dataset_args.train_transformation) \
             + int(self.dataset_args.train_material and len(self.ds.labels_material_active) > 0) \
             + int(self.dataset_args.train_light)

    # Plot properties
    bar_width = 0.3
    sep = 0.05

    height_ratios = []
    if self.dataset_args.train_zingg:
        height_ratios.append(0.5)
    if self.dataset_args.train_distances:
        height_ratios.append(1)
    if self.dataset_args.train_transformation:
        height_ratios.append(0.8)
    if self.dataset_args.train_material and len(self.ds.labels_material_active) > 0:
        height_ratios.append(0.7)
    if self.dataset_args.train_light:
        height_ratios.append(0.8)

    fig = plt.figure(figsize=(n_examples * 2.6, n_rows * 2.3))
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_examples,
        wspace=0.06,
        hspace=0.4,
        width_ratios=[1] * n_examples,
        height_ratios=height_ratios,
        top=0.95,
        bottom=0.04,
        left=0.05,
        right=0.99
    )

    loss = getattr(self.checkpoint, f'loss_{train_or_test}')
    fig.suptitle(
        f'epoch={self.checkpoint.epoch}, '
        f'step={self.checkpoint.step + 1}, '
        f'loss={loss:.4E}',
        fontweight='bold',
        y=0.995
    )
    share_ax = {}

    def plot_zingg(ax_, idx_):
        z_noisy = to_numpy(Yr_noisy['zingg'][idx_])
        z_clean = to_numpy(Yr_clean['zingg'][idx_])
        z_target = to_numpy(params['zingg'][idx_])
        ax_.scatter(z_target[0], z_target[1], c=default_colours[0], marker='o', s=100, label='Target')
        ax_.scatter(z_noisy[0], z_noisy[1], c=default_colours[1], marker='+', s=100, label='Sample')
        ax_.scatter(z_noisy[0], z_clean[1], c=default_colours[2], marker='x', s=100, label='Clean')
        ax_.set_title('Zingg')
        ax_.set_xlabel('S/I', labelpad=-5)
        ax_.set_xlim(0, 1)
        ax_.set_xticks([0, 1])
        if 'zingg' not in share_ax:
            ax_.legend()
            ax_.set_ylim(0, 1)
            ax_.set_yticks([0, 1])
            ax_.set_ylabel('I/L', labelpad=0)
            share_ax['zingg'] = ax_
        else:
            ax_.sharey(share_ax['zingg'])
            ax_.yaxis.set_tick_params(labelleft=False)

    def _plot_bar_chart(key_, idx_, ax_, labels_, title_):
        noisy_ = to_numpy(Yr_noisy[key_][idx_])
        clean_ = to_numpy(Yr_clean[key_][idx_])
        target_ = to_numpy(params[key_][idx_])
        locs = np.arange(len(target_))
        ax_.bar(locs - bar_width, target_, bar_width - sep / 2, label='Target')
        ax_.bar(locs, noisy_, bar_width - sep / 2, label='Noisy')
        ax_.bar(locs + bar_width, clean_, bar_width - sep / 2, label='Clean')
        ax_.set_title(title_)
        ax_.set_xticks(locs)
        ax_.set_xticklabels(labels_)
        if key_ not in share_ax:
            ax_.legend()
            share_ax[key_] = ax_
        else:
            ax_.sharey(share_ax[key_])
            ax_.yaxis.set_tick_params(labelleft=False)
            ax_.autoscale()
        return locs

    def plot_distances(ax_, idx_):
        ax_pos = ax_.get_position()
        ax_.set_position([ax_pos.x0, ax_pos.y0 + 0.02, ax_pos.width, ax_pos.height - 0.02])
        _plot_bar_chart('distances', idx_, ax_, self.ds.labels_distances_active, 'Distances')
        ax_.tick_params(axis='x', rotation=270)

        if self.dataset_args.use_distance_switches:
            s_noisy = to_numpy(Yr_noisy['distance_switches'][idx_])
            s_clean = to_numpy(Yr_clean['distance_switches'][idx_])
            s_target = to_numpy(params['distance_switches'][idx_])
            locs = np.arange(len(s_target))
            colours = []
            for i, (sn, sc, st) in enumerate(zip(s_noisy, s_clean, s_target)):
                if st > 0.5:
                    ax_.axvspan(i - 3 / 2 * bar_width, i - 1 / 2 * bar_width, alpha=0.1, color='blue')
                if sn > 0.5:
                    ax_.axvspan(i - 1 / 2 * bar_width, i + 1 / 2 * bar_width, alpha=0.1,
                                color='red' if sn < 0.5 else 'green')
                if sc > 0.5:
                    ax_.axvspan(i + 1 / 2 * bar_width, i + 3 / 2 * bar_width, alpha=0.1,
                                color='red' if st < 0.5 else 'green')
                colours.append('red' if (st < 0.5 < sc) or (st > 0.5 > sc) else 'green')
            ax_.scatter(locs, s_noisy, color=colours, marker='+', s=30, label='Switches')
            ax_.scatter(locs + bar_width, s_clean, color=colours, marker='+', s=30)

    def plot_transformation(ax_, idx_):
        xlabels = self.ds.labels_transformation.copy()
        if self.ds.ds_args.rotation_mode == ROTATION_MODE_QUATERNION:
            xlabels += self.ds.labels_rotation_quaternion
        else:
            xlabels += self.ds.labels_rotation_axisangle
        _plot_bar_chart('transformation', idx_, ax_, xlabels, 'Transformation')
        locs = np.arange(len(xlabels))
        offset = 1.6 * bar_width
        ax_.axvspan(locs[0] - offset, locs[2] + offset, alpha=0.1, color='green')
        ax_.axvspan(locs[3] - offset, locs[3] + offset, alpha=0.1, color='red')
        ax_.axvspan(locs[4] - offset, locs[-1] + offset, alpha=0.1, color='blue')

    def plot_material(ax_, idx_):
        labels = []
        if 'b' in self.ds.labels_material_active:
            labels.append('Brightness')
        if 'ior' in self.ds.labels_material_active:
            labels.append('IOR')
        if 'r' in self.ds.labels_material_active:
            labels.append('Roughness')
        locs = _plot_bar_chart('material', idx_, ax_, labels, 'Material')
        for i in range(len(locs) - 1):
            ax_.axvline(locs[i] + .5, color='black', linestyle='--', linewidth=1)

    def plot_light(ax_, idx_):
        xlabels = self.ds.labels_light.copy()
        locs = _plot_bar_chart('light', idx_, ax_, xlabels, 'Light')
        if not self.ds.dataset_args.transmission_mode:
            offset = 1.6 * bar_width
            ax_.axvspan(locs[0] - offset, locs[2] + offset, alpha=0.1, color='green')
            ax_.axvspan(locs[3] - offset, locs[3] + offset, alpha=0.1, color='red')
            ax_.axvspan(locs[4] - offset, locs[-1] + offset, alpha=0.1, color='blue')

    for i, idx in enumerate(idxs):
        row_idx = 0
        if self.dataset_args.train_zingg:
            plot_zingg(fig.add_subplot(gs[row_idx, i]), idx)
            row_idx += 1
        if self.dataset_args.train_distances:
            plot_distances(fig.add_subplot(gs[row_idx, i]), idx)
            row_idx += 1
        if self.dataset_args.train_transformation:
            plot_transformation(fig.add_subplot(gs[row_idx, i]), idx)
            row_idx += 1
        if self.dataset_args.train_material and len(self.ds.labels_material_active) > 0:
            plot_material(fig.add_subplot(gs[row_idx, i]), idx)
            row_idx += 1
        if self.dataset_args.train_light:
            plot_light(fig.add_subplot(gs[row_idx, i]), idx)

    self._save_plot(fig, 'vaetc', train_or_test)
