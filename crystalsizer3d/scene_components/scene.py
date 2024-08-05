import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mitsuba as mi
import numpy as np
import torch
import yaml
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from torch import Tensor

from crystalsizer3d import MI_CPU_VARIANT, USE_CUDA
from crystalsizer3d.args.dataset_synthetic_args import ROTATION_MODE_AXISANGLE
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.scene_components.bubble import Bubble
from crystalsizer3d.scene_components.textures import NoiseTexture, NormalMapNoiseTexture
from crystalsizer3d.scene_components.utils import RenderError, build_crystal_mesh, project_to_image
from crystalsizer3d.util.geometry import normalise
from crystalsizer3d.util.utils import hash_data, init_tensor, to_numpy

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant(MI_CPU_VARIANT)
    device = torch.device('cpu')

from mitsuba import ScalarTransform4f as T


class Scene:
    SHAPE_NAME = 'crystal'
    VERTEX_KEY = SHAPE_NAME + '.vertex_positions'
    FACES_KEY = SHAPE_NAME + '.faces'
    BSDF_KEY = SHAPE_NAME + '.bsdf'
    COLOUR_KEY = BSDF_KEY + '.reflectance.value'
    ETA_KEY = BSDF_KEY + '.eta'
    ROUGHNESS_KEY = BSDF_KEY + '.alpha.value'
    RADIANCE_KEY = 'light.emitter.radiance.value'
    mi_scene_dict: dict
    mi_scene: mi.Scene
    hash_id: str
    xy_bounds = {}

    def __init__(
            self,
            crystal: Optional[Crystal] = None,
            crystal_seed: Optional[Crystal] = None,
            bubbles: Optional[List[Bubble]] = None,

            spp: int = 256,
            res: int = 400,
            remesh_max_edge: Optional[float] = None,

            integrator_max_depth: int = 64,
            integrator_rr_depth: int = 5,

            camera_type: str = 'perspective',
            camera_distance: float = 100.,
            focus_distance: float = 90.,
            focal_length: Optional[float] = None,
            camera_fov: Optional[float] = 25.,
            aperture_radius: float = 0.5,
            sampler_type: str = 'stratified',

            light_z_position: float = -10.1,
            light_scale: float = 50.,
            light_radiance: Union[float, Tuple[float, float, float]] = (0.5, 0.5, 0.5),
            light_st_texture: Optional[Union[NoiseTexture, Tensor]] = None,

            cell_z_positions: List[float] = [-10., 0., 50., 60.],
            cell_surface_scale: float = 100.,
            cell_bumpmap: Optional[Union[NormalMapNoiseTexture, Tensor]] = None,
            cell_bumpmap_idx: int = 1,
            cell_render_blanks: bool = False,

            **kwargs
    ):
        """
        Create a scene with the given crystal and bubbles.
        """
        self.crystal = crystal
        self.crystal_seed = crystal_seed
        if bubbles is None:
            bubbles = []
        self.bubbles = bubbles

        # Rendering settings
        self.spp = spp
        self.res = res
        self.remesh_max_edge = remesh_max_edge

        # Integrator parameters
        self.integrator_max_depth = integrator_max_depth
        self.integrator_rr_depth = integrator_rr_depth

        # Sensor parameters
        self.camera_type = camera_type
        self.camera_distance = camera_distance
        self.focus_distance = focus_distance
        assert focal_length is not None or camera_fov is not None, 'Either focal length or camera FOV must be provided.'
        if focal_length is not None and camera_fov is not None:
            camera_fov = None  # Focal length takes precedence
        self.focal_length = focal_length
        self.camera_fov = camera_fov
        self.aperture_radius = aperture_radius
        self.sampler_type = sampler_type

        # Light parameters
        self.light_z_position = light_z_position
        self.light_scale = light_scale
        if isinstance(light_radiance, float):
            light_radiance = (light_radiance, light_radiance, light_radiance)
        self.light_radiance = init_tensor(light_radiance)
        self.light_st_texture = light_st_texture

        # Growth cell parameters
        self.cell_z_positions = cell_z_positions
        self.cell_surface_scale = cell_surface_scale
        self.cell_bumpmap = cell_bumpmap
        self.cell_bumpmap_idx = cell_bumpmap_idx
        self.cell_render_blanks = cell_render_blanks

        # Build the scene
        self.build_mi_scene()

    @property
    def device(self) -> torch.device:
        return device

    def to_dict(self) -> dict:
        """
        Convert all the relevant scene parameters to a dictionary.
        """
        spec = {
            'spp': self.spp,
            'res': self.res,
            'remesh_max_edge': self.remesh_max_edge,

            'integrator_max_depth': self.integrator_max_depth,
            'integrator_rr_depth': self.integrator_rr_depth,

            'camera_type': self.camera_type,
            'camera_distance': self.camera_distance,
            'focus_distance': self.focus_distance,
            'focal_length': self.focal_length,
            'camera_fov': self.camera_fov,
            'aperture_radius': self.aperture_radius,
            'sampler_type': self.sampler_type,

            'light_z_position': self.light_z_position,
            'light_scale': self.light_scale,
            'light_radiance': self.light_radiance.tolist(),
            'light_st_texture': self.light_st_texture.to_dict()
            if isinstance(self.light_st_texture, NoiseTexture) else None,

            'cell_z_positions': self.cell_z_positions,
            'cell_surface_scale': self.cell_surface_scale,
            'cell_bumpmap': self.cell_bumpmap.to_dict()
            if isinstance(self.cell_bumpmap, NormalMapNoiseTexture) else None,
            'cell_bumpmap_idx': self.cell_bumpmap_idx,
            'cell_render_blanks': self.cell_render_blanks,
        }

        if self.crystal is not None:
            spec['crystal'] = self.crystal.to_dict(include_buffers=False)
        if self.crystal_seed is not None:
            spec['crystal_seed'] = self.crystal_seed.to_dict(include_buffers=False)

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

    @classmethod
    def from_dict(cls, spec: dict) -> 'Scene':
        """
        Create a scene from a dictionary.
        """
        args = {k: v for k, v in spec.items() if k not in ['crystal', 'crystal_seed', 'bubbles']}

        if 'crystal' in spec:
            if isinstance(spec['crystal'], Crystal):
                args['crystal'] = spec['crystal']
            else:
                args['crystal'] = Crystal(**spec['crystal'])

        if 'crystal_seed' in spec:
            args['crystal_seed'] = Crystal(**spec['crystal_seed'])

        if 'bubbles' in spec:
            bubbles = []
            for bubble_spec in spec['bubbles']:
                bubble = Bubble(**bubble_spec)
                bubbles.append(bubble)
            args['bubbles'] = bubbles

        return cls(**args)

    def to_yml(self, path: Path, overwrite: bool = False):
        """
        Save the scene to a yaml file.
        """
        if not overwrite:
            assert not path.exists(), f'YAML file already exists at {path}'
        data = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(data, f)

    @classmethod
    def from_yml(cls, path: Path) -> 'Scene':
        """
        Instantiate a scene from a YAML file.
        """
        assert path.exists(), f'YAML file not found at {path}'
        with open(path, 'r') as f:
            scene_args = yaml.load(f)
        return cls.from_dict(scene_args)

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
        type_params = {
            'type': self.camera_type
        }
        if self.camera_type == 'thinlens':
            type_params['aperture_radius'] = self.aperture_radius
        return {
            **type_params,
            'focus_distance': self.focus_distance,
            'focal_length' if self.focal_length is not None else 'fov':
                f'{self.focal_length}mm' if self.focal_length is not None else self.camera_fov,
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
        radiance = self.light_radiance.tolist() if isinstance(self.light_radiance, Tensor) else self.light_radiance
        light_params = {
            'light': {
                'type': 'rectangle',
                'to_world': T.translate([0, 0, self.light_z_position]) @ T.scale(self.light_scale),
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'rgb',
                        'value': radiance
                    }
                },
            }
        }
        if self.light_st_texture is not None:
            if isinstance(self.light_st_texture, NoiseTexture):
                tex = self.light_st_texture.build(device=device)
            elif isinstance(self.light_st_texture, Tensor):
                tex = self.light_st_texture
            else:
                raise ValueError(f'Invalid light specular transmittance texture type: {type(self.light_st_texture)}')
            assert tex.ndim == 3, 'Light specular transmittance texture must be 3D.'
            assert tex.shape[-1] == 3, 'Light specular transmittance texture must have 3 channels.'
            light_params['diffuser'] = {
                'type': 'rectangle',
                'to_world': T.translate([0, 0, self.light_z_position + 0.01]) @ T.scale(self.light_scale),
                'material': {
                    'type': 'roughdielectric',
                    'specular_transmittance': {
                        'type': 'bitmap',
                        'bitmap': mi.Bitmap(mi.TensorXf(tex)),
                        'wrap_mode': 'clamp',
                        'raw': True,
                    }
                }
            }
        return light_params

    @property
    def crystal_material_bsdf(self) -> dict:
        return {
            'type': 'roughdielectric',
            'distribution': 'beckmann',
            'alpha': mi.ScalarFloat(self.crystal.material_roughness),
            'int_ior': mi.ScalarFloat(self.crystal.material_ior),
            'ext_ior': 1.333,  # water
        }

    @property
    def crystal_seed_material_bsdf(self) -> dict:
        return {
            'type': 'dielectric',
            'int_ior': 'water',  # 1.333
            'ext_ior': 'water',  # 1.333
        }

    @property
    def growth_cell_params(self) -> dict:
        cell = {}
        for i, z in enumerate(self.cell_z_positions):
            material_bsdf = {
                'type': 'thindielectric',
                'int_ior': 'acrylic glass',  # 1.49
                'ext_ior': 'water',  # 1.333
            }

            if self.cell_bumpmap is not None and i == self.cell_bumpmap_idx:
                if isinstance(self.cell_bumpmap, NormalMapNoiseTexture):
                    tex = self.cell_bumpmap.build(device=device)
                elif isinstance(self.cell_bumpmap, Tensor):
                    tex = self.cell_bumpmap
                else:
                    raise ValueError(f'Invalid cell bumpmap texture type: {type(self.cell_bumpmap)}')
                assert tex.ndim == 3, 'Cell bumpmap texture must be 2D.'
                bsdf = {
                    'type': 'normalmap',
                    'normalmap': {
                        'type': 'bitmap',
                        'bitmap': mi.Bitmap(mi.TensorXf(tex)),
                        'wrap_mode': 'clamp',
                        'raw': True,
                    },
                    'bsdf': material_bsdf
                }
            elif self.cell_render_blanks:
                bsdf = material_bsdf
            else:
                continue

            cell[f'cell_glass_{i}'] = {
                'type': 'rectangle',
                'to_world': T.translate([0, 0, z]) @ T.scale(self.cell_surface_scale),
                'material': bsdf,
            }

        return cell

    def build_mi_scene(self):
        """
        Build the Mitsuba scene.
        """
        scene_dict = {
            'type': 'scene',
            'integrator': self.integrator_params,
            'sensor': self.sensor_params,
        }
        scene_dict.update(self.light_params)
        scene_dict.update(self.growth_cell_params)

        # Add crystal
        if self.crystal is not None:
            scene_dict[self.SHAPE_NAME] = build_crystal_mesh(
                crystal=self.crystal,
                material_bsdf=self.crystal_material_bsdf,
                shape_name=self.SHAPE_NAME,
                bsdf_key=self.BSDF_KEY,
                remesh_max_edge=self.remesh_max_edge
            )

        # Add crystal seed
        if self.crystal_seed is not None:
            seed_name = self.SHAPE_NAME + '_seed'
            scene_dict[seed_name] = build_crystal_mesh(
                crystal=self.crystal_seed,
                material_bsdf=self.crystal_seed_material_bsdf,
                shape_name=seed_name,
                bsdf_key=seed_name + '_seed',
                remesh_max_edge=self.remesh_max_edge
            )

        # Add bubbles
        for bubble in self.bubbles:
            scene_dict[bubble.SHAPE_NAME] = bubble.build_mitsuba_mesh()

        self.mi_scene_dict = scene_dict
        self.mi_scene = mi.load_dict(scene_dict)
        self.hash_id = hash_data(self.to_dict())

    def render(self, seed: int = 0) -> np.ndarray:
        """
        Render the scene.
        """
        img = mi.render(self.mi_scene, seed=seed)
        img = np.array(mi.util.convert_to_bitmap(img))
        return img

    def get_xy_bounds(
            self,
            z: float = 0,
            z_precision: float = 1e-2,
            verify: bool = True
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get the 3D x and y coordinates that map to the edges of the image at the given z position.
        """
        z = round(z / z_precision) * z_precision
        if self.hash_id not in self.xy_bounds or z not in self.xy_bounds[self.hash_id]:
            pts = torch.tensor([[-1, -1, z], [1, 1, z]], device=device)
            uv_pts = project_to_image(self.mi_scene, pts)
            zoom = torch.abs(uv_pts[0] - uv_pts[1]) / self.res
            bounds = torch.tensor([[-1 / zoom[0], -1 / zoom[1], z], [1 / zoom[0], 1 / zoom[1], z]], device=device)

            # To verify, check that the projected bound points appear at the corners of the image
            if verify:
                uv_pts2 = project_to_image(self.mi_scene, bounds)
                assert torch.allclose(uv_pts2, torch.tensor([[0., self.res], [self.res, 0.]], device=device), atol=1e-3)

            min_x, max_x = bounds[0, 0].item(), bounds[1, 0].item()
            min_y, max_y = bounds[0, 1].item(), bounds[1, 1].item()

            if self.hash_id not in self.xy_bounds:
                self.xy_bounds[self.hash_id] = {}
            self.xy_bounds[self.hash_id][z] = (min_x, max_x), (min_y, max_y)
        return self.xy_bounds[self.hash_id][z]

    def get_crystal_image_coords(self) -> Tensor:
        """
        Get the coordinates of the crystal vertices in the image.
        """
        assert self.crystal is not None, 'No crystal object provided.'
        uv_points = project_to_image(self.mi_scene, self.crystal.vertices)
        return uv_points

    def place_crystal(
            self,
            min_area: float = 0.1,
            max_area: float = 0.5,
            centre_crystal: bool = False,
            max_placement_attempts: int = 1000,
            rebuild_scene: bool = True,
    ):
        """
        Place the crystal in the scene.
        """
        assert self.crystal is not None, 'No crystal object provided.'
        device_og = self.crystal.origin.device
        self.crystal.to(torch.device('cpu'))
        self.crystal.origin.data.zero_()

        # Keep rotating and scaling until the projected area is close to the target
        is_scaled = False
        n_placement_attempts = 0
        while not is_scaled and n_placement_attempts < max_placement_attempts:
            n_placement_attempts += 1

            # Apply random rotation
            if self.crystal.rotation_mode == ROTATION_MODE_AXISANGLE:
                self.crystal.rotation.data = normalise(torch.rand(3)) * torch.rand(1) * 2 * math.pi
            else:
                self.crystal.rotation.data = normalise(torch.rand(4) * 2 * math.pi)

            # Sample a target area for how much of the image should be covered by the crystal
            target_area = np.random.uniform(min_area, max_area)

            # Optimise the crystal scaling to match the target area
            is_scaled = self._optimise_crystal_scaling(target_area)

        # Try again with a slightly smaller target area
        if not is_scaled:
            if min_area > 0.01:
                self.crystal.to(device_og)
                return self.place_crystal(
                    min_area=min_area - 0.01,
                    max_area=max_area - 0.01,
                    centre_crystal=centre_crystal,
                    max_placement_attempts=max_placement_attempts,
                    rebuild_scene=rebuild_scene,
                )
            raise RenderError('Failed to place crystal!')

        # Place the crystal
        if not centre_crystal:
            self.crystal.build_mesh(update_uv_map=False)
            uv_points = project_to_image(self.mi_scene, self.crystal.vertices)
            min_uv = uv_points.argmin(dim=0)
            max_uv = uv_points.argmax(dim=0)
            z = torch.cat([self.crystal.vertices[min_uv, 2], self.crystal.vertices[max_uv, 2]]).amax()
            (min_x, max_x), (min_y, max_y) = self.get_xy_bounds(z=z.item())
            min_x -= self.crystal.vertices[:, 0].min()
            max_x -= self.crystal.vertices[:, 0].max()
            min_y -= self.crystal.vertices[:, 1].min()
            max_y -= self.crystal.vertices[:, 1].max()
            self.crystal.origin.data[:2] = torch.tensor([
                min_x + torch.rand(1) * (max_x - min_x),
                min_y + torch.rand(1) * (max_y - min_y),
            ])

        # Rebuild the scene with the new crystal position
        self.crystal.build_mesh()
        # uv_points = project_to_image(self.mi_scene, self.crystal.vertices)
        # assert not (uv_points[:, 0].min() < 0 or uv_points[:, 0].max() > self.res or
        #             uv_points[:, 1].min() < 0 or uv_points[:, 1].max() > self.res or
        #             self.crystal.vertices[:, 2].min() < self.cell_z_positions[1] or
        #             self.crystal.vertices[:, 2].max() > self.cell_z_positions[2]), 'Crystal out of bounds!'

        # Update the crystal seed
        if self.crystal_seed is not None:
            seed_device_og = self.crystal_seed.origin.device
            self.crystal_seed.to(torch.device('cpu'))
            self.crystal_seed.scale.data *= self.crystal.scale
            self.crystal_seed.origin.data = self.crystal.origin + self.crystal_seed.origin.data * self.crystal.scale
            self.crystal_seed.rotation.data = self.crystal.rotation
            self.crystal_seed.build_mesh()
            self.crystal_seed.to(seed_device_og)

        # Rebuild the scene
        self.crystal.to(device_og)
        if rebuild_scene:
            self.build_mi_scene()

    def _optimise_crystal_scaling(
            self,
            target_area: float,
            tol: float = 1e-3,
    ):
        """
        Optimise the crystal scaling.
        """
        assert torch.allclose(self.crystal.origin, torch.zeros(3))
        self.crystal.build_mesh(update_uv_map=False)
        scale_og = self.crystal.scale.clone().detach()
        scale_new = scale_og.clone().detach().clamp(min=1e-3)
        origin_new = self.crystal.origin.clone().detach()
        vertices_og = self.crystal.vertices.clone().detach()
        vertices_new = vertices_og.clone()

        def _update_vertices():
            nonlocal vertices_new
            vertices_new = vertices_og / scale_og * scale_new
            z_adj = self.cell_z_positions[1] - vertices_new[:, 2].min()
            vertices_new[:, 2] += z_adj
            origin_new[2] = z_adj

        def _in_bounds(img_points):
            return not (img_points[:, 0].min() < 0 or img_points[:, 0].max() > self.res or
                        img_points[:, 1].min() < 0 or img_points[:, 1].max() > self.res or
                        vertices_new[:, 2].max() > self.cell_z_positions[2])

        def _projected_area(img_points):
            # Check the area of the convex hull of the projected points
            pts = to_numpy(img_points)
            if pts.ptp(axis=0).min() < 1e-3:
                return 0
            ch = ConvexHull(to_numpy(img_points))
            return ch.volume / self.res**2  # Note - volume not area (area is perimeter)

        # Ensure the initial guess falls in the bounds by scaling until it must
        oob = True
        n_fails = 0
        while oob:
            _update_vertices()
            uv_points = project_to_image(self.mi_scene, vertices_new)
            if _in_bounds(uv_points):
                oob = False
            else:
                scale_new *= 0.9
                n_fails += 1
                if n_fails > 100:
                    return False
        assert _in_bounds(uv_points)

        # Set up parameters
        x0 = np.array([scale_new, ])

        def _loss(x):
            scale_new.data = torch.tensor(x[0], dtype=self.crystal.scale.dtype)
            scale_new.clamp_(min=1e-3)
            _update_vertices()
            uv_points = project_to_image(self.mi_scene, vertices_new)
            projected_area = _projected_area(uv_points)
            loss = (projected_area - target_area)**2
            if not _in_bounds(uv_points):
                loss += 100
            return loss

        # Find optimal scaling
        res = minimize(
            _loss,
            x0=x0,
            method='COBYLA',
            options={
                'maxiter': 1000,
                'tol': tol
            }
        )

        if res.success and res.fun < tol:
            self.crystal.scale.data = scale_new
            self.crystal.origin.data = origin_new
            return True
        return False

    def place_bubbles(
            self,
            min_scale: float,
            max_scale: float,
            rebuild_scene: bool = True,
    ):
        """
        Place the bubbles in the scene.
        """
        if len(self.bubbles) == 0:
            return
        device_og = self.bubbles[0].origin.device

        # Get the channels where the bubbles may appear
        z_pos = self.cell_z_positions
        h_below = z_pos[1] - z_pos[0]
        h_above = z_pos[3] - z_pos[2]

        # Get the xy bounds
        bounds = {}
        for i, c in enumerate(['bottom', 'top']):
            xb, yb = self.get_xy_bounds(z=(z_pos[2 * i] + z_pos[2 * i + 1]) / 2)
            bounds[c] = {
                'x': xb,
                'y': yb,
                'z': [z_pos[2 * i], z_pos[2 * i + 1]],
            }

        placed_bubbles = []
        for bubble in self.bubbles:
            # Keep trying to place the bubble until it appears in the image and doesn't intersect with anything else in the scene
            bubble = bubble.to(torch.device('cpu'))
            is_placed = False
            n_placement_attempts = 0
            while not is_placed and n_placement_attempts < 100:
                n_placement_attempts += 1

                # Pick whether to place the bubble above or below of the cell
                if np.random.uniform() < h_below / (h_below + h_above):
                    channel = 'bottom'
                else:
                    channel = 'top'
                bounds_c = bounds[channel]
                min_x_c, max_x_c = bounds_c['x']
                min_y_c, max_y_c = bounds_c['y']
                min_z_c, max_z_c = bounds_c['z']

                # Sample the bubble scale and update the z limits
                bubble.scale.data = min_scale + torch.rand(1) * (max_scale - min_scale)
                min_z_c += bubble.scale.item()
                max_z_c -= bubble.scale.item()

                # Place and scale the bubble
                bubble.origin.data = torch.tensor([
                    min_x_c + torch.rand(1) * (max_x_c - min_x_c),
                    min_y_c + torch.rand(1) * (max_y_c - min_y_c),
                    min_z_c + torch.rand(1) * (max_z_c - min_z_c)
                ])
                bubble.build_mesh()

                # Check for intersections with the surfaces
                if (channel == 'bottom'
                        and (bubble.vertices[:, 2].min() < z_pos[0] or bubble.vertices[:, 2].max() > z_pos[1])
                        or channel == 'top'
                        and (bubble.vertices[:, 2].min() < z_pos[2] or bubble.vertices[:, 2].max() > z_pos[3])):
                    continue

                # Check that some of the bubble appears in the image
                uv_points = project_to_image(self.mi_scene, bubble.vertices)
                if uv_points[:, 0].max() < 0 or uv_points[:, 0].min() > self.res or \
                        uv_points[:, 1].max() < 0 or uv_points[:, 1].min() > self.res:
                    continue

                # Check for intersections with the other bubbles
                for placed_bubble in placed_bubbles:
                    origin_dist = (placed_bubble['origin'] - bubble.origin).norm()
                    if origin_dist < placed_bubble['scale'] + bubble.scale:
                        continue

                # Placed successfully!
                is_placed = True
                placed_bubbles.append({
                    'origin': bubble.origin.clone(),
                    'scale': bubble.scale.clone(),
                })

            if not is_placed:
                raise RenderError('Failed to place bubble!')
            bubble.to(device_og)

        # Rebuild the scene with the new bubble positions
        if rebuild_scene:
            self.build_mi_scene()

    def clear_interference(self):
        """
        Clear the textures, bumpmaps, bubbles and defects from the scene.
        """
        self.light_st_texture = None
        self.cell_bumpmap = None
        if self.crystal is not None:
            self.crystal.use_bumpmap = False
            self.crystal.bumpmap = None
        self.crystal_seed = None
        self.bubbles = []
        self.build_mi_scene()
