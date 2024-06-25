from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mayavi import mlab
from torch import Tensor

from crystalsizer3d import USE_MLAB
from crystalsizer3d.crystal import Crystal, ROTATION_MODE_AXISANGLE, ROTATION_MODE_QUATERNION
from crystalsizer3d.util.geometry import get_closest_rotation
from crystalsizer3d.util.utils import equal_aspect_ratio, to_numpy, to_rgb

if TYPE_CHECKING:
    from crystalsizer3d.nn.manager import Manager

# Off-screen rendering
mlab.options.offscreen = True


def _load_single_parameter(
        Y: Union[None, Dict[str, Tensor]],
        key: str,
        idx: Optional[int] = 0
) -> Union[None, np.ndarray]:
    """
    Load a single parameter from the dictionary and convert it to a numpy array.
    """
    if Y is None:
        return None
    val = to_numpy(Y[key])
    if val.ndim == 2:
        val = val[idx]
    return val


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


def add_discriminator_value(ax: Axes, outputs: Dict[str, Tensor], D_key: str, idx: int):
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
        crystal_comp: Optional[Crystal] = None,
        res: int = 100,
        bg_col: float = 1.,
        wireframe_radius_factor: float = 0.1,
        surface_colour: str = 'skyblue',
        wireframe_colour: str = 'cornflowerblue',
        surface_colour_comp: str = 'orange',
        wireframe_colour_comp: str = 'darkorange',
        opacity: float = 0.6,
        azim: float = 150,
        elev: float = 160,
        distance: Optional[float] = None,
        roll: float = 0
) -> Image:
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
    def _add_crystal_mesh(crystal_: Crystal, surface_colour_: str, wireframe_colour_: str):
        origin = crystal_.origin.clone()
        crystal_.origin.data = torch.zeros_like(origin)
        v, f = crystal_.build_mesh()
        v, f = to_numpy(v), to_numpy(f)
        mlab.triangular_mesh(*v.T, f, figure=fig, color=to_rgb(surface_colour_), opacity=opacity)
        tube_radius = max(0.0001, crystal_.distances[0].item() * wireframe_radius_factor)
        for fv_idxs in crystal_.faces.values():
            fv = to_numpy(crystal_.vertices[fv_idxs])
            fv = np.vstack([fv, fv[0]])  # Close the loop
            mlab.plot3d(*fv.T, color=to_rgb(wireframe_colour_), tube_radius=tube_radius)
        crystal_.origin.data = origin

    # Add crystal(s)
    _add_crystal_mesh(crystal, surface_colour, wireframe_colour)
    if crystal_comp is not None:
        _add_crystal_mesh(crystal_comp, surface_colour_comp, wireframe_colour_comp)

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
    frame = mlab.screenshot(mode='rgba', antialiased=True, figure=fig)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = Image.fromarray((frame * 255).astype(np.uint8), 'RGBA')

    mlab.close()

    return img


def make_error_image(
        img_target: np.ndarray,
        img_pred: np.ndarray,
        loss_type: str = 'l2'
) -> Image:
    """
    Make an error image.
    """
    img_target = np.array(Image.fromarray(img_target).convert('L')).astype(np.float32) / 255
    img_pred = np.array(Image.fromarray(img_pred).convert('L')).astype(np.float32) / 255
    if loss_type == 'l2':
        err = (img_target - img_pred)**2
    elif loss_type == 'l1':
        err = np.abs(img_target - img_pred)
    else:
        raise NotImplementedError()
    if err.ndim == 3:
        err = np.mean(err, axis=-1)
    err = err / err.max()
    err_pos, err_neg = err.copy(), err.copy()
    err_pos[img_target > img_pred] = 0
    err_neg[img_target < img_pred] = 0
    img = np.ones((*img_pred.shape, 4))
    img[:, :, 0] = 1 - err_neg
    img[:, :, 1] = 1 - err_pos - err_neg
    img[:, :, 2] = 1 - err_pos
    img[:, :, 3] = np.where(err > 1e-3, 1, 0)
    img = Image.fromarray((img * 255).astype(np.uint8))
    return img


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
        Y_pred: Dict[str, Tensor],
        Y_target: Dict[str, Tensor],
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


def _add_bars(
        ax: Axes,
        pred: np.ndarray,
        target: Optional[np.ndarray] = None,
        pred2: Optional[np.ndarray] = None,
        colour_pred: str = 'cornflowerblue',
        colour_target: str = 'darkorange',
        colour_pred2: str = 'green',
) -> Tuple[np.ndarray, float, float]:
    """
    Add bar chart data to the axis.
    """
    locs = np.arange(len(pred))
    if pred2 is not None:
        bar_width = 0.25
        offset = bar_width
        ax.bar(locs - offset, target, bar_width, color=colour_target, label='Target')
        ax.bar(locs, pred, bar_width, color=colour_pred, label='Predicted')
        ax.bar(locs + offset, pred2, bar_width, color=colour_pred2, label='Predicted2')
    elif target is not None:
        bar_width = 0.35
        offset = bar_width / 2
        ax.bar(locs - offset, target, bar_width, color=colour_target, label='Target')
        ax.bar(locs + offset, pred, bar_width, color=colour_pred, label='Predicted')
    else:
        bar_width = 0.7
        offset = 0
        ax.bar(locs, pred, bar_width, color=colour_pred, label='Predicted')

    return locs, bar_width, offset


def _shared_ax_legend(share_ax: Dict[str, Axes], ax: Axes, key: str):
    """
    Add a shared legend to the axis.
    """
    if share_ax is None:
        ax.legend()
    else:
        if key not in share_ax:
            ax.legend()
            share_ax[key] = ax
        else:
            ax.sharey(share_ax[key])
            ax.yaxis.set_tick_params(labelleft=False)
            ax.autoscale()


def plot_distances(
        ax: Axes,
        manager: Manager,
        Y_pred: Dict[str, Tensor],
        Y_target: Optional[Dict[str, Tensor]] = None,
        Y_pred2: Optional[Dict[str, Tensor]] = None,
        idx: int = 0,
        colour_pred: str = 'cornflowerblue',
        colour_target: str = 'darkorange',
        colour_pred2: str = 'green',
        share_ax: Optional[Dict[str, Axes]] = None
):
    """
    Plot the distances on the axis.
    """
    ds = manager.ds
    d_pred = _load_single_parameter(Y_pred, 'distances', idx)
    d_target = _load_single_parameter(Y_target, 'distances', idx)
    d_pred2 = _load_single_parameter(Y_pred2, 'distances', idx)

    # Prepare the distances
    d_pred = ds.prep_distances(torch.from_numpy(d_pred))
    if d_target is not None:
        d_target = ds.prep_distances(torch.from_numpy(d_target))
    if d_pred2 is not None:
        d_pred2 = ds.prep_distances(torch.from_numpy(d_pred2))

    # Group asymmetric distances by face group
    distance_groups = {}
    if ds.dataset_args.asymmetry is not None:
        grouped_order_pred = []
        grouped_order_target = []
        grouped_order_pred2 = []
        for i, hkl in enumerate(ds.dataset_args.miller_indices):
            group_idxs = (manager.crystal.symmetry_idx == i).nonzero().squeeze()
            distance_groups[hkl] = group_idxs
            dpi = d_pred[group_idxs].argsort()
            grouped_order_pred.append(group_idxs[dpi])
            if d_target is not None:
                dti = d_target[group_idxs].argsort()
                grouped_order_target.append(group_idxs[dti])
            if d_pred2 is not None:
                dp2i = d_pred2[group_idxs].argsort()
                grouped_order_pred2.append(group_idxs[dp2i])
        grouped_order_pred = torch.cat(grouped_order_pred)
        d_pred = d_pred[grouped_order_pred]
        if d_target is not None:
            grouped_order_target = torch.cat(grouped_order_target)
            d_target = d_target[grouped_order_target]
        if d_pred2 is not None:
            grouped_order_pred2 = torch.cat(grouped_order_pred2)
            d_pred2 = d_pred2[grouped_order_pred2]

    # Add bar chart data
    locs, bar_width, offset = _add_bars(
        ax=ax,
        pred=d_pred,
        target=d_target,
        pred2=d_pred2,
        colour_pred=colour_pred,
        colour_target=colour_target,
        colour_pred2=colour_pred2,
    )

    # Reorder labels for asymmetric distances
    if ds.dataset_args.asymmetry is not None:
        xlabels = []
        for i, (hkl, g) in enumerate(distance_groups.items()):
            group_labels = [''] * len(g)
            group_labels[len(g) // 2] = '(' + ''.join(map(str, hkl)) + ')'
            xlabels.extend(group_labels)

            # Add vertical separator lines between face groups
            if i < len(distance_groups) - 1:
                ax.axvline(locs[len(xlabels) - 1] + 0.5, color='black', linestyle='--', linewidth=1)
    else:
        xlabels = ['(' + ''.join(list(l[3:])) + ')' for l in ds.labels_distances]

    # Replace -X with \bar{X} in labels
    xlabels = [re.sub(r'-(\d)', r'$\\bar{\1}$', label) for label in xlabels]

    if manager.dataset_args.use_distance_switches:
        s_pred = _load_single_parameter(Y_pred, 'distance_switches', idx)
        s_target = _load_single_parameter(Y_target, 'distance_switches', idx)
        if ds.dataset_args.asymmetry is not None:
            s_pred = s_pred[grouped_order_pred]
            s_target = s_target[grouped_order_target]
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
    ax.set_xticklabels(xlabels)
    if len(xlabels) > 5:
        ax.tick_params(axis='x', labelsize='small')
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '', '1'])
    _shared_ax_legend(share_ax, ax, 'distances')


def plot_areas(
        ax: Axes,
        manager: Manager,
        crystal_pred: Crystal,
        Y_pred: Dict[str, Tensor],
        crystal_target: Optional[Crystal] = None,
        Y_target: Optional[Dict[str, Tensor]] = None,
        idx: int = 0,
        colour_pred: str = 'cornflowerblue',
        colour_target: str = 'darkorange',
        share_ax: Optional[Dict[str, Axes]] = None,
        **kwargs
):
    """
    Plot the face areas on the axis.
    """
    ds = manager.ds

    # Load the area values from the crystal
    a_pred = np.array(list(crystal_pred.areas.values()))
    if crystal_target is not None:
        a_target = np.array(list(crystal_target.areas.values()))
    else:
        a_target = None

    # Load the distances for sorting the areas in the same way
    d_pred = _load_single_parameter(Y_pred, 'distances', idx)
    d_target = _load_single_parameter(Y_target, 'distances', idx)
    d_pred = ds.prep_distances(torch.from_numpy(d_pred))
    if d_target is not None:
        d_target = ds.prep_distances(torch.from_numpy(d_target))

    # Group asymmetric distances by face group
    distance_groups = {}
    if ds.dataset_args.asymmetry is not None:
        grouped_order_pred = []
        grouped_order_target = []
        for i, hkl in enumerate(ds.dataset_args.miller_indices):
            group_idxs = (manager.crystal.symmetry_idx == i).nonzero().squeeze()
            distance_groups[hkl] = group_idxs
            dpi = d_pred[group_idxs].argsort()
            grouped_order_pred.append(group_idxs[dpi])
            if d_target is not None:
                dti = d_target[group_idxs].argsort()
                grouped_order_target.append(group_idxs[dti])
        grouped_order_pred = torch.cat(grouped_order_pred)
        a_pred = a_pred[grouped_order_pred]
        if d_target is not None:
            grouped_order_target = torch.cat(grouped_order_target)
            a_target = a_target[grouped_order_target]
    else:
        a_pred = a_pred[:len(d_pred)]
        if a_target is not None:
            a_target = a_target[:len(d_target)]

    # Add bar chart data
    locs, bar_width, offset = _add_bars(
        ax=ax,
        pred=a_pred,
        target=a_target,
        colour_pred=colour_pred,
        colour_target=colour_target,
    )

    # Reorder labels for asymmetric distances
    if ds.dataset_args.asymmetry is not None:
        xlabels = []
        for i, (hkl, g) in enumerate(distance_groups.items()):
            group_labels = [''] * len(g)
            group_labels[len(g) // 2] = '(' + ''.join(map(str, hkl)) + ')'
            xlabels.extend(group_labels)

            # Add vertical separator lines between face groups
            if i < len(distance_groups) - 1:
                ax.axvline(locs[len(xlabels) - 1] + 0.5, color='black', linestyle='--', linewidth=1)
    else:
        xlabels = ['(' + l[3:] + ')' for l in ds.labels_distances]

    # Replace -X with \bar{X} in labels
    xlabels = [re.sub(r'-(\d)', r'$\\bar{\1}$', label) for label in xlabels]

    ax.set_title('Face areas')
    ax.set_xticks(locs)
    ax.set_xticklabels(xlabels)
    if len(xlabels) > 5:
        ax.tick_params(axis='x', labelsize='small')
    _shared_ax_legend(share_ax, ax, 'areas')


def plot_transformation(
        ax: Axes,
        manager: Manager,
        Y_pred: Dict[str, Tensor],
        Y_target: Optional[Dict[str, Tensor]] = None,
        Y_pred2: Optional[Dict[str, Tensor]] = None,
        idx: int = 0,
        colour_pred: str = 'cornflowerblue',
        colour_target: str = 'darkorange',
        colour_pred2: str = 'green',
        share_ax: Optional[Dict[str, Axes]] = None
):
    """
    Plot the transformation parameters.
    """
    t_pred = _load_single_parameter(Y_pred, 'transformation', idx)
    t_target = _load_single_parameter(Y_target, 'transformation', idx)
    t_pred2 = _load_single_parameter(Y_pred2, 'transformation', idx)

    # Adjust the target rotation to the best matching symmetry group
    if (Y_target is not None
            and manager.dataset_args.rotation_mode in [ROTATION_MODE_QUATERNION, ROTATION_MODE_AXISANGLE]
            and 'sym_rotations' in Y_target):
        sym_rotations = Y_target['sym_rotations']
        if isinstance(sym_rotations, list):
            sym_rotations = sym_rotations[idx]
        t_target[4:] = get_closest_rotation(t_pred[4:], sym_rotations)

    # Add bar chart data
    locs, bar_width, offset = _add_bars(
        ax=ax,
        pred=t_pred,
        target=t_target,
        pred2=t_pred2,
        colour_pred=colour_pred,
        colour_target=colour_target,
        colour_pred2=colour_pred2,
    )

    # Add highlight spans
    if t_pred2 is not None:
        k = 2 * offset
    elif t_target is not None:
        k = 3 / 2 * bar_width
    else:
        k = 3 / 4 * bar_width
    ax.axvspan(locs[0] - k, locs[2] + k, alpha=0.1, color='green')
    ax.axvspan(locs[3] - k, locs[3] + k, alpha=0.1, color='red')
    ax.axvspan(locs[4] - k, locs[-1] + k, alpha=0.1, color='blue')

    ax.set_title('Transformation')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    lbls = []
    for l in manager.ds.labels_transformation_active:
        if l in ['x', 'z', 'rw', 'ry', 'rz', 'rax', 'raz']:
            lbls.append('')
        elif l == 'y':
            lbls.append('Position')
        elif l == 's':
            lbls.append('Scale')
        elif l in ['rx', 'ray']:
            lbls.append('Rotation')
    ax.set_xticks(locs)
    ax.set_xticklabels(lbls)
    y_extent = max(1, max(abs(y) for y in ax.get_ylim()))
    ax.set_ylim(-y_extent, y_extent)
    ylabels = [str(int(y)) if y == int(y) else '' for y in ax.get_yticks()]
    ax.set_yticks(ax.get_yticks())  # Suppress warning
    ax.set_yticklabels(ylabels)
    _shared_ax_legend(share_ax, ax, 'transformation')


def plot_material(
        ax: Axes,
        manager: Manager,
        Y_pred: Dict[str, Tensor],
        Y_target: Optional[Dict[str, Tensor]] = None,
        Y_pred2: Optional[Dict[str, Tensor]] = None,
        idx: int = 0,
        colour_pred: str = 'cornflowerblue',
        colour_target: str = 'darkorange',
        colour_pred2: str = 'green',
        share_ax: Optional[Dict[str, Axes]] = None
):
    """
    Plot the material parameters.
    """
    m_pred = _load_single_parameter(Y_pred, 'material', idx)
    m_target = _load_single_parameter(Y_target, 'material', idx)
    m_pred2 = _load_single_parameter(Y_pred2, 'material', idx)

    # Add bar chart data
    locs, bar_width, offset = _add_bars(
        ax=ax,
        pred=m_pred,
        target=m_target,
        pred2=m_pred2,
        colour_pred=colour_pred,
        colour_target=colour_target,
        colour_pred2=colour_pred2,
    )

    ax.set_title('Material')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_xticks(locs)
    for i in range(len(locs) - 1):
        ax.axvline(locs[i] + .5, color='black', linestyle='--', linewidth=1)
    labels = []
    if 'ior' in manager.ds.labels_material_active:
        labels.append('IOR')
    if 'r' in manager.ds.labels_material_active:
        labels.append('Roughness')
    ax.set_xticklabels(labels)
    _shared_ax_legend(share_ax, ax, 'material')


def plot_light(
        ax: Axes,
        Y_pred: Dict[str, Tensor],
        Y_target: Optional[Dict[str, Tensor]] = None,
        Y_pred2: Optional[Dict[str, Tensor]] = None,
        idx: int = 0,
        colour_pred: str = 'cornflowerblue',
        colour_target: str = 'darkorange',
        colour_pred2: str = 'green',
        share_ax: Optional[Dict[str, Axes]] = None,
        **kwargs
):
    """
    Plot the light parameters.
    """
    l_pred = _load_single_parameter(Y_pred, 'light', idx)
    l_target = _load_single_parameter(Y_target, 'light', idx)
    l_pred2 = _load_single_parameter(Y_pred2, 'light', idx)

    # Add bar chart data
    locs, bar_width, offset = _add_bars(
        ax=ax,
        pred=l_pred,
        target=l_target,
        pred2=l_pred2,
        colour_pred=colour_pred,
        colour_target=colour_target,
        colour_pred2=colour_pred2,
    )

    ax.set_title('Light')
    ax.set_xticks(locs)
    ax.set_xticklabels(['R', 'G', 'B'])
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    _shared_ax_legend(share_ax, ax, 'light')


def plot_training_samples(
        manager: Manager,
        data: Tuple[dict, Tensor, Tensor, Tensor, Dict[str, Tensor]],
        outputs: Dict[str, Any],
        train_or_test: str,
        idxs: List[int],
) -> Figure:
    """
    Plot the image and parameter comparisons.
    """
    n_examples = len(idxs)
    metas, images, images_aug, images_clean, Y_target = data
    Y_pred = outputs['Y_pred']
    dsa = manager.dataset_args
    if dsa.train_generator:
        X_pred = outputs['X_pred']
        X_pred2 = outputs['X_pred2']
        Y_pred2 = outputs['Y_pred2']
    else:
        X_pred = None
        X_pred2 = None
        Y_pred2 = None
    n_rows = 4 \
             + int(dsa.train_zingg) \
             + 2 * int(dsa.train_distances) \
             + int(dsa.train_transformation) \
             + int(dsa.train_material and len(manager.ds.labels_material_active) > 0) \
             + int(dsa.train_light) \
             + int(dsa.train_generator) * 2

    height_ratios = [1.3, 1.3, 1.1, 1.1]  # images and 3d plots
    if dsa.train_zingg:
        height_ratios.append(0.5)
    if dsa.train_distances:
        height_ratios.append(0.8)
        height_ratios.append(0.7)
    if dsa.train_transformation:
        height_ratios.append(0.7)
    if dsa.train_material and len(manager.ds.labels_material_active) > 0:
        height_ratios.append(0.7)
    if dsa.train_light:
        height_ratios.append(0.7)
    if dsa.train_generator:
        height_ratios.insert(0, 1.3)
        height_ratios.insert(0, 1.3)

    fig = plt.figure(figsize=(n_examples * 2.7, n_rows * 2.3))
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
        if dsa.train_generator:
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
        if dsa.train_zingg:
            plot_zingg(fig.add_subplot(gs[row_idx, i]), **shared_args)
            row_idx += 1
        if dsa.train_distances:
            plot_distances(fig.add_subplot(gs[row_idx, i]), **shared_args)
            row_idx += 1
            plot_areas(fig.add_subplot(gs[row_idx, i]),
                       crystal_pred=crystal_pred, crystal_target=crystal_target, **shared_args)
            row_idx += 1
        if dsa.train_transformation:
            plot_transformation(fig.add_subplot(gs[row_idx, i]), **shared_args)
            row_idx += 1
        if dsa.train_material and len(manager.ds.labels_material_active) > 0:
            plot_material(fig.add_subplot(gs[row_idx, i]), **shared_args)
            row_idx += 1
        if dsa.train_light:
            plot_light(fig.add_subplot(gs[row_idx, i]), **shared_args)

    return fig


def plot_generator_samples(
        manager: Manager,
        data: Tuple[dict, Tensor, Tensor, Tensor, Dict[str, Tensor]],
        outputs: Dict[str, Any],
        train_or_test: str,
        idxs: List[int],
) -> Figure:
    """
    Plot the image and generator output.
    """
    n_examples = len(idxs)
    metas, images, images_aug, images_clean, Y_target = data
    X_pred = outputs['X_pred']
    n_rows = 3
    fig = plt.figure(figsize=(n_examples * 2.7, n_rows * 3))
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_examples,
        wspace=0.06,
        hspace=0.03,
        top=0.95,
        bottom=0.004,
        left=0.01,
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

    for i, idx in enumerate(idxs):
        meta = metas[idx]

        # Plot the (possibly augmented) input image
        img = to_numpy(images_aug[idx]).squeeze()
        ax = fig.add_subplot(gs[0, i])
        plot_image(ax, meta['image'].name, img)

        # Plot the clean image
        img = to_numpy(images_clean[idx]).squeeze()
        ax = fig.add_subplot(gs[1, i])
        plot_image(ax, 'Clean target', img)
        add_discriminator_value(ax, outputs, 'D_real', idx)

        # Plot the generated image
        img = to_numpy(X_pred[idx]).squeeze()
        ax = fig.add_subplot(gs[2, i])
        plot_image(ax, 'Generated', img)
        add_discriminator_value(ax, outputs, 'D_fake', idx)

    return fig


def _plot_vaetc_examples(
        self,
        data: Tuple[dict, Tensor, Tensor, Dict[str, Tensor]],
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
