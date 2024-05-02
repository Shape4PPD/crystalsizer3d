import time
from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# Required to stop some import errors (?!)
from ccdc.io import EntryReader  # type: ignore
from mayavi import mlab
# --
from ccdc.morphology import VisualHabitMorphology
from kornia.geometry import axis_angle_to_rotation_matrix, quaternion_to_rotation_matrix, rotation_matrix_to_axis_angle, \
    rotation_matrix_to_quaternion
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from torch.utils.data import default_collate
from torchvision.transforms.functional import center_crop, crop, to_tensor
from trimesh import Trimesh

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.nn.manager import CCDC_AVAILABLE, Manager
from crystalsizer3d.util.utils import geodesic_distance, print_args, to_dict, to_numpy, to_rgb

# Off-screen rendering
mlab.options.offscreen = True


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            model_path: Path,
            image_path: Optional[Path] = None,
            ds_idx: int = 0,

            img_size_3d: int = 400,
            wireframe_r_factor: float = 0.005,
            surface_colour_target: str = 'orange',
            wireframe_colour_target: str = 'red',
            surface_colour_pred: str = 'skyblue',
            wireframe_colour_pred: str = 'darkblue',
            azim: float = 0,
            elev: float = 0,
            roll: float = 0,
            distance: float = 10,

            plot_colour_target: str = 'red',
            plot_colour_pred: str = 'darkblue',

            **kwargs
    ):
        assert model_path.exists(), f'Dataset path does not exist: {model_path}'
        assert model_path.suffix == '.json', f'Model path must be a json file: {model_path}'
        self.model_path = model_path
        if image_path is not None:
            assert image_path.exists(), f'Image path does not exist: {image_path}'
        self.image_path = image_path
        self.ds_idx = ds_idx

        # Digital crystal image
        self.img_size_3d = img_size_3d
        self.wireframe_r_factor = wireframe_r_factor
        self.surface_colour_target = surface_colour_target
        self.wireframe_colour_target = wireframe_colour_target
        self.surface_colour_pred = surface_colour_pred
        self.wireframe_colour_pred = wireframe_colour_pred
        self.azim = azim
        self.elev = elev
        self.roll = roll
        self.distance = distance

        # Parameter plots
        self.plot_colour_target = plot_colour_target
        self.plot_colour_pred = plot_colour_pred

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Runtime Args')
        group.add_argument('--model-path', type=Path, required=True,
                           help='Path to the model\'s json file.')
        group.add_argument('--image-path', type=Path,
                           help='Path to the image to process. If set, will override the dataset entry.')
        group.add_argument('--ds-idx', type=int, default=0,
                           help='Index of the dataset entry to use.')

        # Digital crystal image
        group.add_argument('--img-size-3d', type=int, default=400,
                           help='Size of the 3D digital crystal image.')
        group.add_argument('--wireframe-r-factor', type=float, default=0.005,
                           help='Wireframe radius factor, multiplied by the maximum dimension of the bounding box to calculate the final edge tube radius.')
        group.add_argument('--surface-colour-target', type=str, default='orange',
                           help='Target mesh surface colour.')
        group.add_argument('--wireframe-colour-target', type=str, default='darkorange',
                           help='Target mesh wireframe colour.')
        group.add_argument('--surface-colour-pred', type=str, default='skyblue',
                           help='Predicted mesh surface colour.')
        group.add_argument('--wireframe-colour-pred', type=str, default='cornflowerblue',
                           help='Predicted mesh wireframe colour.')
        group.add_argument('--azim', type=float, default=50,
                           help='Azimuthal angle of the camera.')
        group.add_argument('--elev', type=float, default=50,
                           help='Elevation angle of the camera.')
        group.add_argument('--roll', type=float, default=-120,
                           help='Roll angle of the camera.')
        group.add_argument('--distance', type=float, default=100,
                           help='Camera distance.')

        # Parameter plots
        group.add_argument('--plot-colour-target', type=str, default='darkorange',
                           help='Target parameters plot colour.')
        group.add_argument('--plot-colour-pred', type=str, default='cornflowerblue',
                           help='Predicted parameters plot colour.')

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


def _prep_distances(
        manager: Manager,
        distance_vals: np.ndarray
) -> np.ndarray:
    """
    Prepare the distances to fill zeros and apply constraints.
    """
    ds = manager.ds
    crystal_generator = manager.crystal_generator
    distances = np.zeros(len(ds.labels_distances))
    pos_active = [ds.labels_distances.index(k) for k in ds.labels_distances_active]
    for i, pos in enumerate(pos_active):
        distances[pos] = distance_vals[i]
    if crystal_generator.constraints is not None:
        largest_hkl = ''.join([str(hkl) for hkl in crystal_generator.constraints[0]])
        largest_pos = [d[-3:] for d in ds.labels_distances].index(largest_hkl)
        distances[largest_pos] = 1
    return distances


def _calculate_sf(
        mesh1: Trimesh,
        mesh2: Trimesh
) -> float:
    """
    Calculate the scale factor needed to match the custom implementation with the CSD one.
    """
    sf = mesh1.bounding_box.vertices / mesh2.bounding_box.vertices
    assert np.allclose(sf, sf[0, 0])
    return float(sf[0, 0])


def _build_crystal_csd(
        Y: Dict[str, torch.Tensor],
        manager: Manager,
) -> Tuple[VisualHabitMorphology, Trimesh]:
    """
    Build a digital crystal from the parameters using VisualHabit.
    """
    if not CCDC_AVAILABLE:
        raise RuntimeError('CCDC unavailable.')

    # Normalise the distances
    distances = to_numpy(Y['distances']).astype(float)
    distances = _prep_distances(manager, distances)
    if manager.dataset_args.use_distance_switches:
        switches = to_numpy(Y['distance_switches'])
        distances = np.where(switches < .5, 0, distances)
    distances[distances < 0] = 0

    if distances.max() < 1e-8:
        raise RuntimeError('Failed to build crystal:\nno positive distances.')

    # Build the crystal morphology and mesh
    distances /= distances.max()
    growth_rates = manager.crystal_generator.get_expanded_growth_rates(distances)
    morph = VisualHabitMorphology.from_growth_rates(manager.crystal_generator.crystal, growth_rates)
    build_attempts = 0
    max_attempts = 10
    while True:
        try:
            _, _, mesh = manager.crystal_generator.generate_crystal(rel_distances=distances, validate=False)
            break
        except AssertionError as e:
            if build_attempts < max_attempts and str(e) == 'Mesh is not watertight!':
                build_attempts += 1
            else:
                raise e

    return morph, mesh


def _build_crystal_custom(
        Y: Dict[str, torch.Tensor],
        manager: Manager,
        apply_origin: bool = True,
        apply_scale: bool = True,
        apply_rotation: bool = True,
        scale_to_csd: bool = True
) -> Crystal:
    """
    Generate a beta-form LGA crystal.
    """
    reader = EntryReader()
    crystal = reader.crystal(manager.ds.dataset_args.crystal_id)
    lattice_unit_cell = [crystal.cell_lengths[0], crystal.cell_lengths[1], crystal.cell_lengths[2]]
    lattice_angles = [crystal.cell_angles[0], crystal.cell_angles[1], crystal.cell_angles[2]]
    point_group_symbol = '222'  # crystal.spacegroup_symbol - this doesn't work

    # Get the required miller indices from the constraints
    miller_indices = manager.crystal_generator.constraints[:-1]

    # Extract the parameters
    device = Y['distances'].device

    # Normalise the distances
    distances = to_numpy(Y['distances']).astype(float)
    distances = _prep_distances(manager, distances)
    if manager.dataset_args.use_distance_switches:
        switches = to_numpy(Y['distance_switches'])
        distances = np.where(switches < .5, 0, distances)
    distances[distances < 0] = 0
    distances /= distances.max()

    growth_rates = manager.crystal_generator.get_expanded_growth_rates(distances)
    # morph = VisualHabitMorphology.from_growth_rates(crystal, growth_rates)

    miller_indices = []
    distances = []
    for (mi, d) in growth_rates:
        miller_indices.append(mi.hkl)
        distances.append(d * 100)

    # Build the initial (non-transformed or scaled) crystal
    crystal = Crystal(
        lattice_unit_cell=lattice_unit_cell,
        lattice_angles=lattice_angles,
        miller_indices=miller_indices,
        point_group_symbol=point_group_symbol,
        distances=distances,
        rotation_mode=manager.dataset_args.rotation_mode,
    )
    crystal.to(device)

    # # Apply scaling to match csd
    # if scale_to_csd:
    #     v, f = crystal.build_mesh()
    #     mesh_custom = Trimesh(vertices=to_numpy(v), faces=to_numpy(f))
    #     _, mesh_csd = _build_crystal_csd(Y, manager)
    #     sf = _calculate_sf(mesh_csd, mesh_custom)
    #     crystal.distances.data *= sf

    # Apply transformation
    r_params = manager.ds.denormalise_rendering_params(Y)
    if apply_origin:
        crystal.origin.data = torch.tensor(r_params['location'], device=device)
    if apply_scale:
        crystal.distances.data *= r_params['scale']
    if apply_rotation:
        crystal.rotation.data = Y['transformation'][4:].clone().detach()

    # Rebuild the mesh
    crystal.build_mesh()
    # v2, f2 = crystal.build_mesh()
    # mesh2 = Trimesh(vertices=to_numpy(v2), faces=to_numpy(f2))

    return crystal


def _check_custom_crystal(
        Y: Dict[str, torch.Tensor],
        manager: Manager,
        threshold: float = 0.9
):
    """
    Check if we can use the custom mesh in place of the CSD one.
    """
    _, mesh1 = _build_crystal_csd(Y, manager)

    # Build the custom mesh
    crystal = _build_crystal_custom(Y, manager, apply_origin=False, apply_scale=False, apply_rotation=False,
                                    scale_to_csd=True)
    v2, f2 = crystal.build_mesh()
    mesh2 = Trimesh(vertices=to_numpy(v2), faces=to_numpy(f2))

    # Check the meshes are close
    intersection = mesh1.intersection(mesh2)
    union = mesh1.union(mesh2)
    iou = intersection.volume / union.volume

    assert iou > threshold, f'IOU too low: {iou:.2f} < {threshold:.2f}.'


def _make_3d_crystal_image(
        manager: Manager,
        args: RuntimeArgs,
        target_or_pred: str,
        Y: Dict[str, torch.Tensor],
        Y_comp: Optional[Dict[str, torch.Tensor]] = None,
        sym_idx: int = 0,
        csd_or_custom: str = 'csd',
) -> Image:
    """
    Make a 3D digital crystal plot.
    """
    assert target_or_pred in ['target', 'pred']
    assert csd_or_custom in ['csd', 'custom']

    # Set up mlab figure
    fig = mlab.figure(size=(args.img_size_3d * 2, args.img_size_3d * 2))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    def _add_crystal_mesh(
            Y_: Dict[str, torch.Tensor],
            target_or_pred_: str,
    ):
        """
        Add a crystal mesh to the figure.
        """
        if csd_or_custom == 'csd':
            morph, mesh = _build_crystal_csd(Y_, manager)
        else:
            crystal = _build_crystal_custom(Y_, manager, apply_origin=False)
            v, f = crystal.build_mesh()
            mesh = Trimesh(vertices=to_numpy(v), faces=to_numpy(f))

        wireframe_r = np.max(np.ptp(mesh.bounding_box.vertices, axis=0)) * args.wireframe_r_factor
        surf_col = args.surface_colour_target if target_or_pred_ == 'target' else args.surface_colour_pred
        wire_col = args.wireframe_colour_target if target_or_pred_ == 'target' else args.wireframe_colour_pred

        # Get the scale factor and rotation matrix
        scale = manager.ds.denormalise_rendering_params(Y_)['scale']
        rot = Y_['transformation'][4:][None, ...]
        if target_or_pred_ == 'target' and 'sym_rotations' in Y_:
            R = Y_['sym_rotations'][sym_idx][None, ...]
        elif manager.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
            R = quaternion_to_rotation_matrix(rot)
        elif manager.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE:
            R = axis_angle_to_rotation_matrix(rot)
        else:
            raise NotImplementedError()
        R = to_numpy(R[0])

        # Add crystal surface mesh
        if csd_or_custom == 'csd':
            v = R @ mesh.vertices.T * scale
        else:
            v = mesh.vertices.T
        mlab.triangular_mesh(*v, mesh.faces, figure=fig, color=to_rgb(surf_col), opacity=0.7)

        # Add crystal wireframe
        if csd_or_custom == 'csd':
            for f in morph.facets:
                fv = np.array(f.coordinates)
                fv = np.vstack([fv, fv[0]])  # Close the loop
                fv = R @ fv.T * scale
                mlab.plot3d(*fv, color=to_rgb(wire_col), tube_radius=wireframe_r)
        else:
            for fv_idxs in crystal.faces.values():
                fv = to_numpy(crystal.vertices[fv_idxs])
                fv = np.vstack([fv, fv[0]])  # Close the loop
                mlab.plot3d(*fv.T, color=to_rgb(wire_col), tube_radius=wireframe_r)

    # Add the crystal(s)
    _add_crystal_mesh(Y, target_or_pred)
    if Y_comp is not None:
        _add_crystal_mesh(Y_comp, 'pred' if target_or_pred == 'target' else 'target')

    # Render
    mlab.view(figure=fig, azimuth=args.azim, elevation=args.elev, distance=args.distance, roll=args.roll,
              focalpoint=np.zeros(3))

    # # Useful for getting the view parameters when recording from the gui:
    # # mlab.show()
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [42.40655283143192, 48.852597501236176, 47.97853006317138]
    # scene.scene.camera.focal_point = [0.0, 1.2434497875801753e-14, 4.2521541843143495e-13]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [-0.40972639568048425, -0.4329104804459839, 0.8029400952765448]
    # scene.scene.camera.clipping_range = [47.868405842182995, 121.86303485516211]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    frame = mlab.screenshot(mode='rgba', antialiased=True, figure=fig)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = Image.fromarray((frame * 255).astype(np.uint8), 'RGBA')
    mlab.close()

    return img


def _render_crystal(
        manager: Manager,
        Y: Dict[str, torch.Tensor],
        default_rendering_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Render a 3D crystal from the predicted parameters.
    """
    crystal = _build_crystal_custom(Y, manager)
    img = render_crystal_scene(crystal, spp=256, res=manager.ds.image_size)

    return img


def _make_error_image(
        img_target: np.ndarray,
        img_pred: np.ndarray,
        loss_type: str = 'l2'
) -> Image:
    """
    Make an error image.
    """
    img_pred2 = np.array(Image.fromarray(img_pred).convert('L')).astype(np.float32) / 255
    if loss_type == 'l2':
        err = (img_target - img_pred2)**2
    elif loss_type == 'l1':
        err = np.abs(img_target - img_pred2)
    else:
        raise NotImplementedError()
    err = err / err.max()
    err_pos, err_neg = err.copy(), err.copy()
    err_pos[img_target > img_pred2] = 0
    err_neg[img_target < img_pred2] = 0
    img = np.ones((*img_pred2.shape, 4))
    img[:, :, 0] = 1 - err_neg
    img[:, :, 1] = 1 - err_pos - err_neg
    img[:, :, 2] = 1 - err_pos
    img[:, :, 3] = np.where(err > 1e-3, 1, 0)
    img = Image.fromarray((img * 255).astype(np.uint8))
    return img


def _plot_distances(
        ax: Axes,
        manager: Manager,
        args: RuntimeArgs,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Plot the distances.
    """
    ax_pos = ax.get_position()
    ax.set_position([ax_pos.x0, ax_pos.y0 + 0.02, ax_pos.width, ax_pos.height - 0.02])
    d_pred = to_numpy(Y_pred['distances'])
    d_pred = np.clip(d_pred, a_min=0, a_max=np.inf)
    d_target = to_numpy(Y_target['distances']) if Y_target is not None else None
    locs = np.arange(len(d_pred))
    xlabels = ['(' + ','.join(list(l[3:])) + ')' for l in manager.ds.labels_distances_active]
    if d_target is not None:
        bar_width = 0.35
        ax.bar(locs - bar_width / 2, d_target, bar_width, color=args.plot_colour_target, label='Target')
        ax.bar(locs + bar_width / 2, d_pred, bar_width, color=args.plot_colour_pred, label='Predicted')
    else:
        bar_width = 0.7
        ax.bar(locs, d_pred, bar_width, color=args.plot_colour_pred, label='Predicted')

    if manager.ds.dataset_args.distance_constraints is not None:
        locs = np.concatenate([[-1], locs])
        ax.bar(-1, 1, bar_width, color='purple', label='Constraint')
        largest_hkl = '(' + ','.join([str(hkl) for hkl in manager.crystal_generator.constraints[0]]) + ')'
        xlabels = [largest_hkl] + xlabels

    ax.set_title('Distances')
    ax.set_xticks(locs)
    ax.set_xticklabels(xlabels)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '', '1'])
    ax.set_ylim(0, 1)
    ax.legend()


def _get_closest_target_rotation(
        manager: Manager,
        r_pred: torch.Tensor,
        Y_target: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Get the closest target rotation to the predicted rotation, accounting for symmetries.
    """
    sym_rotations = Y_target['sym_rotations']
    if manager.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
        raise NotImplementedError('Needs testing!')
        v_norms = r_pred.norm(dim=-1, keepdim=True)
        r_pred = r_pred / v_norms
        q_pred = r_pred[None, ...]
        dot_product = torch.sum(sym_rotations * q_pred, dim=-1)
        dot_product = torch.clamp(dot_product, -0.99, 0.99)  # Ensure valid input for arccos
        angular_differences = 2 * torch.acos(dot_product)
        min_idx = int(torch.argmin(angular_differences))
        r_target = rotation_matrix_to_quaternion(sym_rotations)[min_idx]

    else:
        assert manager.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE
        R_pred = axis_angle_to_rotation_matrix(torch.from_numpy(r_pred)[None, ...])
        R_pred = R_pred.expand(len(sym_rotations), -1, -1)
        angular_differences = geodesic_distance(R_pred, sym_rotations)
        min_idx = int(torch.argmin(angular_differences))
        r_target = rotation_matrix_to_axis_angle(sym_rotations)[min_idx]

    return r_target


def _plot_transformation(
        ax: Axes,
        manager: Manager,
        args: RuntimeArgs,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Plot the transformation parameters.
    """
    t_pred = to_numpy(Y_pred['transformation'])
    t_target = to_numpy(Y_target['transformation']) if Y_target is not None else None
    locs = np.arange(len(t_pred))
    if t_target is not None:
        # Adjust the target rotation to the best matching symmetry group
        if manager.dataset_args.rotation_mode in [ROTATION_MODE_QUATERNION, ROTATION_MODE_AXISANGLE] \
                and 'sym_rotations' in Y_target:
            t_target[4:] = _get_closest_target_rotation(manager, t_pred[4:], Y_target)
        bar_width = 0.35
        k = 3 / 2 * bar_width
        ax.bar(locs - bar_width / 2, t_target, bar_width, color=args.plot_colour_target, label='Target')
        ax.bar(locs + bar_width / 2, t_pred, bar_width, color=args.plot_colour_pred, label='Predicted')
    else:
        bar_width = 0.7
        k = 3 / 4 * bar_width
        ax.bar(locs, t_pred, bar_width, color=args.plot_colour_pred, label='Predicted')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.axvspan(locs[0] - k, locs[2] + k, alpha=0.1, color='green')
    ax.axvspan(locs[3] - k, locs[3] + k, alpha=0.1, color='red')
    ax.axvspan(locs[4] - k, locs[-1] + k, alpha=0.1, color='blue')
    ax.set_title('Transformation')
    ax.set_xticks(locs)
    lbls = []
    for l in manager.ds.labels_transformation_active:
        if l in ['x', 'z', 'rax', 'raz']:
            lbls.append('')
        elif l == 'y':
            lbls.append('Position')
        elif l == 's':
            lbls.append('Scale')
        elif l == 'ray':
            lbls.append('Rotation')
    ax.set_xticklabels(lbls)
    y_extent = max(1, max(abs(y) for y in ax.get_ylim()))
    ax.set_ylim(-y_extent, y_extent)
    ylabels = [str(int(y)) if y == int(y) else '' for y in ax.get_yticks()]
    ax.set_yticks(ax.get_yticks())  # Suppress warning
    ax.set_yticklabels(ylabels)
    # ax.set_yticklabels([])


def _plot_material(
        ax: Axes,
        manager: Manager,
        args: RuntimeArgs,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Plot the material parameters.
    """
    m_pred = to_numpy(Y_pred['material'])
    m_target = to_numpy(Y_target['material']) if Y_target is not None else None
    locs = np.arange(len(m_pred))
    if m_target is not None:
        bar_width = 0.35
        ax.bar(locs - bar_width / 2, m_target, bar_width, color=args.plot_colour_target, label='Target')
        ax.bar(locs + bar_width / 2, m_pred, bar_width, color=args.plot_colour_pred, label='Predicted')
    else:
        bar_width = 0.7
        ax.bar(locs, m_pred, bar_width, color=args.plot_colour_pred, label='Predicted')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_title('Material')
    ax.set_xticks(locs)
    for i in range(len(locs) - 1):
        ax.axvline(locs[i] + .5, color='black', linestyle='--', linewidth=1)
    labels = []
    if 'b' in manager.ds.labels_material_active:
        labels.append('Brightness')
    if 'ior' in manager.ds.labels_material_active:
        labels.append('IOR')
    if 'r' in manager.ds.labels_material_active:
        labels.append('Roughness')
    ax.set_xticklabels(labels)
    y_extent = max(1, max(abs(y) for y in ax.get_ylim()))
    ax.set_ylim(-y_extent, y_extent)
    ylabels = [str(int(y)) if y == int(y) else '' for y in ax.get_yticks()]
    ax.set_yticks(ax.get_yticks())  # Suppress warning
    ax.set_yticklabels(ylabels)
    # ax.set_yticklabels([])


def _plot_light(
        ax: Axes,
        manager: Manager,
        args: RuntimeArgs,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Plot the light parameters.
    """
    l_pred = to_numpy(Y_pred['light'])
    l_target = to_numpy(Y_target['light']) if Y_target is not None else None
    locs = np.arange(len(l_pred))
    if l_target is not None:
        bar_width = 0.35
        ax.bar(locs - bar_width / 2, l_target, bar_width, color=args.plot_colour_target, label='Target')
        ax.bar(locs + bar_width / 2, l_pred, bar_width, color=args.plot_colour_pred, label='Predicted')
    else:
        bar_width = 0.7
        ax.bar(locs, l_pred, bar_width, color=args.plot_colour_pred, label='Predicted')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    if not manager.ds.renderer_args.transmission_mode:
        k = 3 / 4 * bar_width
        ax.axvspan(locs[0] - k, locs[2] + k, alpha=0.1, color='green')
        ax.axvspan(locs[3] - k, locs[3] + k, alpha=0.1, color='red')
        ax.axvspan(locs[4] - k, locs[-1] + k, alpha=0.1, color='blue')
    ax.set_title('Light')
    ax.set_xticks(locs)
    if len(manager.ds.labels_light_active) == 1:
        ax.set_xticklabels(['Intensity'])
    else:
        ax.set_xticklabels(manager.ds.labels_light_active)
    y_extent = max(1, max(abs(y) for y in ax.get_ylim()))
    ax.set_ylim(-y_extent, y_extent)
    ylabels = [str(int(y)) if y == int(y) else '' for y in ax.get_yticks()]
    ax.set_yticks(ax.get_yticks())  # Suppress warning
    ax.set_yticklabels(ylabels)
    # ax.set_yticklabels([])


def _plot_parameters(
        manager: Manager,
        args: RuntimeArgs,
        Y_pred: Dict[str, torch.Tensor],
        Y_target: Optional[Dict[str, torch.Tensor]] = None,
) -> Figure:
    """
    Plot the image and parameter predictions.
    """
    ds_args = manager.dataset_args

    fig = plt.figure(figsize=(5, 4))
    gs = GridSpec(
        nrows=2,
        ncols=2,
        wspace=0.2,
        hspace=0.4,
        top=0.93,
        bottom=0.08,
        left=0.07,
        right=0.98
    )

    shared_args = (manager, args, Y_pred, Y_target)
    if ds_args.train_distances:
        _plot_distances(fig.add_subplot(gs[0, 0]), *shared_args)
    if ds_args.train_transformation:
        _plot_transformation(fig.add_subplot(gs[0, 1]), *shared_args)
    if ds_args.train_material:
        _plot_material(fig.add_subplot(gs[1, 0]), *shared_args)
    if ds_args.train_light:
        _plot_light(fig.add_subplot(gs[1, 1]), *shared_args)

    return fig
    # plt.show()
    # exit()


def plot_prediction(args: Optional[RuntimeArgs] = None):
    """
    Plot the predicted parameters for a given image.
    """
    if args is None:
        args = parse_arguments()

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Create an output directory
    if args.image_path is None:
        target_str = str(args.ds_idx)
    else:
        target_str = args.image_path.stem
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_{args.model_path.stem[:4]}_{target_str}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments to json file
    with open(save_dir / 'args.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    # Instantiate the manager from the checkpoint json path
    manager = Manager.load(
        model_path=args.model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': USE_CUDA,
                'batch_size': 1
            },
        },
        save_dir=save_dir
    )

    # Put networks in eval mode
    manager.predictor.eval()
    if manager.generator is not None:
        manager.generator.eval()
    if manager.transcoder is not None:
        manager.transcoder.eval()

    # Load the input image (and parameters if loading from the dataset)
    if args.image_path is None:
        metas, X_target, Y_target = manager.ds.load_item(args.ds_idx)
        X_target = to_tensor(X_target)
        Y_target = {
            k: torch.from_numpy(v).to(torch.float32).to(manager.device)
            for k, v in Y_target.items()
        }
        default_rendering_params = metas['rendering_parameters']

        # _check_custom_crystal(Y_target, manager)
        # exit()

    else:
        X_target = to_tensor(Image.open(args.image_path).convert('L'))
        Y_target = None
        default_rendering_params = None  # todo!

        # Crop and resize the image to the working image size
        # X_target = center_crop(X_target, min(X_target.shape[-2:]))
        d = min(X_target.shape[-2:])
        X_target = crop(X_target, top=0, left=X_target.shape[-1] - d, height=d, width=d)
        X_target = F.interpolate(
            X_target[None, ...],
            size=manager.image_shape[-1],
            mode='bilinear',
            align_corners=False
        )[0]
    X_target = default_collate([X_target, ])
    X_target = X_target.to(manager.device)

    # Predict parameters
    logger.info('Predicting parameters.')
    Y_pred = manager.predict(X_target)

    # Strip batch dimensions
    X_target = X_target[0]
    Y_pred = {k: v[0] for k, v in Y_pred.items()}

    # Check the custom crystal meshes are close enough to the CSD ones
    _check_custom_crystal(Y_pred, manager)
    if Y_target is not None:
        _check_custom_crystal(Y_target, manager)

    # Plot the digital crystals
    logger.info('Plotting digital crystals.')
    try:
        dig_pred_A = _make_3d_crystal_image(manager, args, target_or_pred='pred', Y=Y_pred)
        dig_pred_A.save(save_dir / 'digital_predicted_A.png')
        dig_pred_B = _make_3d_crystal_image(manager, args, target_or_pred='pred', Y=Y_pred, csd_or_custom='custom')
        dig_pred_B.save(save_dir / 'digital_predicted_B.png')
    except Exception as e:
        logger.warning(f'Failed to plot predicted digital crystal: {e}')
    if Y_target is not None:
        dig_target_A = _make_3d_crystal_image(manager, args, target_or_pred='target', Y=Y_target)
        dig_target_A.save(save_dir / 'digital_target_A.png')
        dig_target_B = _make_3d_crystal_image(manager, args, target_or_pred='target', Y=Y_target,
                                              csd_or_custom='custom')
        dig_target_B.save(save_dir / 'digital_target_B.png')
        # try:
        #     dig_combined_A = _make_3d_crystal_image(manager, args, target_or_pred='pred', Y=Y_pred, Y_comp=Y_target)
        #     dig_combined_A.save(save_dir / 'digital_combined_A.png')
        #     dig_combined_B = _make_3d_crystal_image(manager, args, target_or_pred='pred', Y=Y_pred, Y_comp=Y_target, csd_or_custom='custom')
        #     dig_combined_B.save(save_dir / 'digital_combined_B.png')
        # except Exception as e:
        #     logger.warning(f'Failed to plot combined digital crystals: {e}')
        #     raise e
    exit()

    # Save the original image
    logger.info('Saving rendered images.')
    img_target = to_numpy(X_target).squeeze()
    Image.fromarray((img_target * 255).astype(np.uint8)).save(save_dir / 'target.png')

    # Re-render using the custom mesh and mitsuba (pipeline B)
    if Y_target is not None:
        img_target2 = _render_crystal(manager, Y_target, default_rendering_params, pipeline='mitsuba')
        Image.fromarray(img_target2).convert('L').save(save_dir / 'target2.png')

    # Render the predicted crystal
    img_pred = _render_crystal(manager, Y_pred, default_rendering_params)
    Image.fromarray(img_pred).convert('L').save(save_dir / 'predicted.png')

    # Create error images
    img_l2 = _make_error_image(img_target, img_pred, loss_type='l2')
    img_l2.save(save_dir / 'error_l2.png')
    img_l1 = _make_error_image(img_target, img_pred, loss_type='l1')
    img_l1.save(save_dir / 'error_l1.png')

    # Plot the parameter values
    fig = _plot_parameters(manager, args, Y_pred, Y_target)
    fig.savefig(save_dir / 'parameters.svg', transparent=True)

    # Print how long this took
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logger.info(f'Finished in {int(minutes):02d}:{int(seconds):02d}.')


if __name__ == '__main__':
    plot_prediction()

    # # Iterate over all images in the image_path directory
    # args_ = parse_arguments()
    # img_dir = args_.image_path.parent
    # for img_path in img_dir.iterdir():
    #     args_.image_path = img_path
    #     try:
    #         plot_prediction(args_)
    #     except Exception as e:
    #         logger.error(f'Failed to plot prediction for image {img_path}: {e}')
    #         continue

    # Plot the first 20 images in the dataset
    # for i in range(20):
    #     args_.ds_idx = i
    #     try:
    #         plot_prediction(args_)
    #     except Exception as e:
    #         logger.error(f'Failed to plot prediction for index {i}: {e}')
    #         continue
# --image-path=/home/tom0/projects/CrystalSizer_v2/logs/transmission_46/LGA_000040.jpg
