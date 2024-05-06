import math
from typing import List, Optional, Tuple

import mitsuba as mi
import numpy as np
import torch
from scipy.spatial import ConvexHull
from trimesh import Trimesh
from trimesh.collision import CollisionManager

from crystalsizer3d import USE_CUDA
from crystalsizer3d.args.dataset_synthetic_args import ROTATION_MODE_AXISANGLE
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.scene_components.bubble import Bubble
from crystalsizer3d.scene_components.utils import RenderError, build_crystal_mesh, project_to_image
from crystalsizer3d.util.geometry import normalise
from crystalsizer3d.util.utils import to_numpy

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
else:
    mi.set_variant('llvm_ad_rgb')

from mitsuba import ScalarTransform4f as T


class Scene:
    SHAPE_NAME = 'crystal'
    VERTEX_KEY = SHAPE_NAME + '.vertex_positions'
    FACES_KEY = SHAPE_NAME + '.faces'
    BSDF_KEY = SHAPE_NAME + '.bsdf'
    COLOUR_KEY = BSDF_KEY + '.reflectance.value'

    def __init__(
            self,
            crystal: Optional[Crystal] = None,
            bubbles: Optional[List[Bubble]] = None,

            spp: int = 256,
            res: int = 400,

            integrator_max_depth: int = 64,
            integrator_rr_depth: int = 5,

            camera_distance: float = 100.,
            camera_fov: float = 25.,
            focus_distance: float = 90.,
            aperture_radius: float = 0.5,
            sampler_type: str = 'stratified',

            light_z_position: float = -10.,
            light_scale: float = 50.,
            light_radiance: Tuple[float, float, float] = (0.5, 0.5, 0.5),

            cell_z_positions: List[float] = [-10., 0., 50., 60.],
            cell_surface_scale: float = 100.,

            **kwargs
    ):
        """
        Create a scene with the given crystal and bubbles.
        """
        self.crystal = crystal
        if bubbles is None:
            bubbles = []
        self.bubbles = bubbles

        # Rendering settings
        self.spp = spp
        self.res = res

        # Integrator parameters
        self.integrator_max_depth = integrator_max_depth
        self.integrator_rr_depth = integrator_rr_depth

        # Sensor parameters
        self.camera_distance = camera_distance
        self.camera_fov = camera_fov
        self.focus_distance = focus_distance
        self.aperture_radius = aperture_radius
        self.sampler_type = sampler_type

        # Light parameters
        self.light_z_position = light_z_position
        self.light_scale = light_scale
        self.light_radiance = light_radiance

        # Growth cell parameters
        self.cell_z_positions = cell_z_positions
        self.cell_surface_scale = cell_surface_scale

        self._build_mi_scene()

    def to_dict(self) -> dict:
        """
        Convert all the relevant scene parameters to a dictionary.
        """
        spec = {
            'spp': self.spp,
            'res': self.res,

            'integrator_max_depth': self.integrator_max_depth,
            'integrator_rr_depth': self.integrator_rr_depth,

            'camera_distance': self.camera_distance,
            'camera_fov': self.camera_fov,
            'focus_distance': self.focus_distance,
            'aperture_radius': self.aperture_radius,
            'sampler_type': self.sampler_type,

            'light_z_position': self.light_z_position,
            'light_scale': self.light_scale,
            'light_radiance': self.light_radiance,

            'cell_z_positions': self.cell_z_positions,
            'cell_surface_scale': self.cell_surface_scale,
        }

        if self.crystal is not None:
            spec['crystal'] = {
                'scale': self.crystal.scale.item(),
                'distances': self.crystal.distances.tolist(),
                'origin': self.crystal.origin.tolist(),
                'rotation': self.crystal.rotation.tolist(),
                'material_roughness': self.crystal.material_roughness.item(),
                'material_ior': self.crystal.material_ior.item(),
                'use_bumpmap': self.crystal.use_bumpmap,
            }

        if len(self.bubbles) > 0:
            spec['bubbles'] = []
            for bubble in self.bubbles:
                spec['bubbles'].append({
                    'origin': bubble.origin.tolist(),
                    'scale': bubble.scale.item(),
                    'roughness': bubble.roughness.item(),
                    'ior': bubble.ior.item(),
                })

        return spec

    @property
    def integrator_params(self) -> dict:
        return {
            'type': 'prb_projective',
            'max_depth': self.integrator_max_depth,
            'rr_depth': self.integrator_rr_depth,
            'sppi': 0,
            # 'guiding': 'grid',
            # 'guiding_rounds': 10
        }

    @property
    def sensor_params(self) -> dict:
        return {
            'type': 'thinlens',
            'aperture_radius': self.aperture_radius,
            'focus_distance': self.focus_distance,
            'fov': self.camera_fov,
            'to_world': T.look_at(
                origin=[0, 0, self.camera_distance],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': self.sampler_type,
                'sample_count': self.spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.res,
                'height': self.res,
                'filter': {'type': 'gaussian'},
                'sample_border': True,
            },
        }

    @property
    def light_params(self) -> dict:
        return {
            'type': 'rectangle',
            'to_world': T.translate([0, 0, self.light_z_position]) @ T.scale(self.light_scale),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': self.light_radiance
                }
            },
        }

    @property
    def crystal_material_bsdf(self) -> dict:
        return {
            'type': 'roughdielectric',
            'distribution': 'beckmann',
            'alpha': mi.ScalarFloat(self.crystal.material_roughness),
            'int_ior': mi.ScalarFloat(self.crystal.material_ior),
            'ext_ior': 'water',  # 1.333
        }

    @property
    def growth_cell_params(self) -> dict:
        cell = {}
        for i, z in enumerate(self.cell_z_positions):
            cell[f'cell_glass_{i}'] = {
                'type': 'rectangle',
                'to_world': T.translate([0, 0, z]) @ T.scale(self.cell_surface_scale),
                'material': {
                    'type': 'thindielectric',
                    'int_ior': 'acrylic glass',  # 1.49
                    'ext_ior': 'water',  # 1.333
                },
            }
        return cell

    def _build_mi_scene(self):
        """
        Build the Mitsuba scene.
        """
        scene_dict = {
            'type': 'scene',
            'integrator': self.integrator_params,
            'sensor': self.sensor_params,
            'light': self.light_params,
        }
        scene_dict.update(self.growth_cell_params)

        # Add crystal
        if self.crystal is not None:
            scene_dict[self.SHAPE_NAME] = build_crystal_mesh(
                crystal=self.crystal,
                material_bsdf=self.crystal_material_bsdf,
                shape_name=self.SHAPE_NAME,
                bsdf_key=self.BSDF_KEY,
            )

        # Add bubbles
        for bubble in self.bubbles:
            scene_dict[bubble.SHAPE_NAME] = bubble.build_mesh()

        self.mi_scene_dict = scene_dict
        self.mi_scene: mi.Scene = mi.load_dict(scene_dict)

    def render(self, seed: int = 0) -> np.ndarray:
        """
        Render the scene.
        """
        img = mi.render(self.mi_scene, seed=seed)
        img = np.array(mi.util.convert_to_bitmap(img))
        return img

    def get_crystal_image_coords(self) -> torch.Tensor:
        """
        Get the coordinates of the crystal vertices in the image.
        """
        assert self.crystal is not None, 'No crystal object provided.'
        self.crystal.build_mesh()
        uv_points = project_to_image(self.mi_scene, self.crystal.vertices)
        return uv_points

    def place_crystal(
            self,
            min_area: float = 0.1,
            max_area: float = 0.5,
            centre_crystal: bool = False,
            min_x: float = -50.,
            max_x: float = 50.,
            min_y: float = -50.,
            max_y: float = 50.,
    ):
        """
        Place the crystal in the scene.
        """
        assert self.crystal is not None, 'No crystal object provided.'
        device = self.crystal.origin.device

        # Keep trying to place the crystal until the projected area is close to the target
        is_placed = False
        n_placement_attempts = 0
        while not is_placed and n_placement_attempts < 1000:
            n_placement_attempts += 1

            # Sample a target area for how much of the image should be covered by the crystal
            target_area = np.random.uniform(min_area, max_area)

            # Place the crystal
            if centre_crystal:
                self.crystal.origin.data.zero_()
            else:
                self.crystal.origin.data = torch.tensor([
                    min_x + torch.rand(1, device=device) * (max_x - min_x),
                    min_y + torch.rand(1, device=device) * (max_y - min_y),
                    0
                ], device=device)

            # Apply random rotation
            if self.crystal.rotation_mode == ROTATION_MODE_AXISANGLE:
                self.crystal.rotation.data = (normalise(torch.rand(3, device=device))
                                              * torch.rand(1, device=device) * 2 * math.pi)
            else:
                self.crystal.rotation.data = normalise(torch.rand(4, device=device) * 2 * math.pi)

            # Adjust scaling until the projected area is close to the target
            projected_area = 0
            n_scale_attempts = 0
            oob = False
            while abs(projected_area - target_area) > target_area * 0.1 and n_scale_attempts < 1000:
                if n_scale_attempts != 0:
                    if projected_area > target_area:
                        self.crystal.scale.data *= 0.9
                    else:
                        self.crystal.scale.data *= 1.1

                # Adjust the position so the bottom of the crystal is on the growth cell surface
                v, f = self.crystal.build_mesh()
                self.crystal.origin.data[2] -= v[:, 2].min() - self.cell_z_positions[1]

                # Project the crystal vertices onto the image plane
                uv_points = self.get_crystal_image_coords()
                v = self.crystal.vertices

                # Check that the crystal is within the image bounds and the growth cell
                if (uv_points[:, 0].min() < 0 or uv_points[:, 0].max() > self.res or
                        uv_points[:, 1].min() < 0 or uv_points[:, 1].max() > self.res or
                        v[:, 2].min() < self.cell_z_positions[1] or v[:, 2].max() > self.cell_z_positions[2]
                ):
                    if n_scale_attempts < 5:
                        self.crystal.scale.data /= 2
                        n_scale_attempts += 1
                        continue
                    else:
                        oob = True
                        break

                # Check the area of the convex hull of the projected points
                ch = ConvexHull(to_numpy(uv_points))
                projected_area = ch.volume / self.res**2  # Note - volume not area (area is perimeter)

                n_scale_attempts += 1
            if projected_area < min_area or projected_area > max_area or oob:
                # Failed to scale crystal to meet the target area so try a new orientation
                continue

            # Placed successfully!
            is_placed = True

        if not is_placed:
            raise RenderError('Failed to place crystal!')

        # Rebuild the scene with the new crystal position
        self._build_mi_scene()

    def place_bubbles(
            self,
            min_x: float,
            max_x: float,
            min_y: float,
            max_y: float,
            min_scale: float,
            max_scale: float,
    ):
        """
        Place the bubbles in the scene.
        """
        if len(self.bubbles) == 0:
            return
        device = self.bubbles[0].origin.device

        # Get the channels where the bubbles may appear
        z_pos = self.cell_z_positions
        h_below = z_pos[1] - z_pos[0]
        h_above = z_pos[3] - z_pos[2]
        collision_managers = {'bottom': CollisionManager(), 'top': CollisionManager()}

        for bubble in self.bubbles:
            # Keep trying to place the bubble until it appears in the image and doesn't intersect with anything else in the scene
            is_placed = False
            n_placement_attempts = 0
            while not is_placed and n_placement_attempts < 100:
                n_placement_attempts += 1

                # Pick whether to place the bubble above or below of the cell
                if np.random.uniform() < h_below / (h_below + h_above):
                    channel = 'bottom'
                    z = z_pos[0] + torch.rand(1, device=device) * (z_pos[1] - z_pos[0])
                else:
                    channel = 'top'
                    z = z_pos[2] + torch.rand(1, device=device) * (z_pos[3] - z_pos[2])

                # Place and scale the bubble
                bubble.origin.data = torch.tensor([
                    min_x + torch.rand(1, device=device) * (max_x - min_x),
                    min_y + torch.rand(1, device=device) * (max_y - min_y),
                    z
                ], device=device)
                bubble.scale.data = min_scale + torch.rand(1, device=device) * (max_scale - min_scale)
                bubble.build_mesh()

                # Check for intersections with the surfaces
                if (channel == 'bottom'
                        and (bubble.vertices[:, 2].min() < z_pos[0] or bubble.vertices[:, 2].max() > z_pos[1])
                        or channel == 'top'
                        and (bubble.vertices[:, 2].min() < z_pos[2] or bubble.vertices[:, 2].max() > z_pos[3])):
                    continue

                # Check for intersections with the other bubbles
                mesh = Trimesh(vertices=to_numpy(bubble.vertices), faces=to_numpy(bubble.faces))
                cm = collision_managers[channel]
                if cm.in_collision_single(mesh):
                    continue

                # Check that some of the bubble appears in the image
                uv_points = project_to_image(self.mi_scene, bubble.vertices)
                if uv_points[:, 0].max() < 0 or uv_points[:, 0].min() > self.res or \
                        uv_points[:, 1].max() < 0 or uv_points[:, 1].min() > self.res:
                    continue

                # Placed successfully!
                is_placed = True
                cm.add_object(bubble.SHAPE_NAME, mesh)

            if not is_placed:
                raise RenderError('Failed to place bubble!')

        # Rebuild the scene with the new bubble positions
        self._build_mi_scene()

    def clear_bubbles_and_bumpmap(self):
        """
        Clear the bubbles and bumpmap from the scene.
        """
        self.bubbles = []
        if self.crystal is not None:
            self.crystal.use_bumpmap = False
            self.crystal.bumpmap = None
        self._build_mi_scene()
