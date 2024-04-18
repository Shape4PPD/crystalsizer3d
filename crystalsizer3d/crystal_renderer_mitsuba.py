import csv
import json
import math
import shutil
from multiprocessing import Lock
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
from PIL import Image
from ccdc.io import EntryReader
from pytorch3d.utils import ico_sphere
from scipy.spatial import ConvexHull
from torch import nn
from trimesh import Trimesh
from trimesh.collision import CollisionManager

from crystalsizer3d import USE_CUDA, logger
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs, ROTATION_MODE_AXISANGLE
from crystalsizer3d.args.renderer_args import RendererArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.crystal_renderer import CrystalRenderer, RenderError, append_json
from crystalsizer3d.util.utils import SEED, normalise, to_numpy

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('llvm_ad_rgb')
    device = torch.device('cpu')

from mitsuba import ScalarTransform4f as T


class Bubble(nn.Module):
    vertices: torch.Tensor
    faces: torch.Tensor

    def __init__(
            self,
            shape_name: str = 'bubble',
            origin: List[float] = [0, 0, 0],
            scale: float = 1.0,
            colour: List[float] = [1, 1, 1],
            roughness: float = 0.05,
            ior: float = 1.5,
    ):
        """
        Create a sphere with the given scale, origin, and rotation.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=True)
        self.origin = nn.Parameter(torch.tensor(origin, dtype=torch.float32), requires_grad=True)
        self.colour = nn.Parameter(torch.tensor(colour, dtype=torch.float32), requires_grad=True)
        self.roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float32), requires_grad=True)
        self.ior = nn.Parameter(torch.tensor(ior, dtype=torch.float32), requires_grad=True)
        self.register_buffer('vertices', torch.empty(0))
        self.register_buffer('faces', torch.empty(0))

        # Parameter keys
        self.SHAPE_NAME = shape_name
        self.VERTEX_KEY = self.SHAPE_NAME + '.vertex_positions'
        self.FACES_KEY = self.SHAPE_NAME + '.faces'
        self.BSDF_KEY = self.SHAPE_NAME + '.bsdf'
        self.COLOUR_KEY = self.BSDF_KEY + '.reflectance.value'

    def build_mesh(self) -> mi.Mesh:
        """
        Create a sphere mesh with the given origin and radius.
        """
        # Build basic sphere
        sphere = ico_sphere(level=5, device=self.origin.device)
        vertices = sphere.verts_packed()
        faces = sphere.faces_packed()

        # Apply scaling
        vertices = vertices * self.scale

        # Apply translation
        vertices = vertices + self.origin

        # Store the pytorch vertices and faces
        self.vertices = vertices.clone()
        self.faces = faces.clone()

        # Convert to a mitsuba mesh
        nv, nf = len(vertices), len(faces)
        vertices = mi.TensorXf(vertices)
        faces = mi.TensorXi64(faces)

        # Set up the material properties
        bsdf = {
            'type': 'roughdielectric',
            'distribution': 'beckmann',
            'alpha': self.roughness.item(),
            'int_ior': self.ior.item(),
        }
        props = mi.Properties()
        props[self.BSDF_KEY] = mi.load_dict(bsdf)

        # Construct the mitsuba mesh and set the vertex positions and faces
        mesh = mi.Mesh(
            self.SHAPE_NAME,
            vertex_count=nv,
            face_count=nf,
            has_vertex_normals=False,
            has_vertex_texcoords=False,
            props=props
        )
        mesh_params = mi.traverse(mesh)
        mesh_params['vertex_positions'] = dr.ravel(vertices)
        mesh_params['faces'] = dr.ravel(faces)

        return mesh


class Scene:
    SHAPE_NAME = 'crystal'
    VERTEX_KEY = SHAPE_NAME + '.vertex_positions'
    FACES_KEY = SHAPE_NAME + '.faces'
    BSDF_KEY = SHAPE_NAME + '.bsdf'
    COLOUR_KEY = BSDF_KEY + '.reflectance.value'

    def __init__(
            self,
            crystal: Optional[Crystal] = None,
            bubbles: List[Bubble] = [],

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
            scene_dict[self.SHAPE_NAME] = self._build_crystal_mesh()

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
        uv_points = self._project_to_image(self.crystal.vertices)
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
        while not is_placed and n_placement_attempts < 100:
            n_placement_attempts += 1

            # Sample a target area for how much of the image should be covered by the crystal
            target_area = np.random.uniform(min_area, max_area)

            # Place the crystal
            if centre_crystal:
                self.crystal.origin.data.zero_()
            else:
                self.crystal.origin.data = torch.tensor([
                    min_x + torch.rand() * (max_x - min_x),
                    min_y + torch.rand() * (max_y - min_y),
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
            while abs(projected_area - target_area) > target_area * 0.1 and n_scale_attempts < 100:
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
                uv_points = self._project_to_image(bubble.vertices)
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

    def _build_crystal_mesh(self) -> mi.Mesh:
        """
        Convert the Crystal object into a Mitsuba mesh.
        """
        assert self.crystal is not None, 'No crystal object provided.'

        # Build the mesh in pytorch and convert the parameters to Mitsuba format
        vertices, faces = self.crystal.build_mesh()
        nv, nf = len(vertices), len(faces)
        vertices = mi.TensorXf(vertices)
        faces = mi.TensorXi64(faces)

        # Set up the material properties
        if self.crystal.use_bumpmap:
            bsdf = {
                'type': 'bumpmap',
                'texture': {
                    'type': 'bitmap',
                    'bitmap': mi.Bitmap(mi.TensorXf(self.crystal.bumpmap)),
                    'wrap_mode': 'clamp',
                    'raw': True
                },
                'bsdf': self.crystal_material_bsdf
            }
        else:
            bsdf = self.crystal_material_bsdf
        props = mi.Properties()
        props[self.BSDF_KEY] = mi.load_dict(bsdf)

        # Construct the mitsuba mesh and set the vertex positions and faces
        mesh = mi.Mesh(
            self.SHAPE_NAME,
            vertex_count=nv,
            face_count=nf,
            has_vertex_normals=False,
            has_vertex_texcoords=self.crystal.use_bumpmap,
            props=props
        )
        mesh_params = mi.traverse(mesh)
        mesh_params['vertex_positions'] = dr.ravel(vertices)
        mesh_params['faces'] = dr.ravel(faces)

        # Update the texture coordinates if a bumpmap is used
        if self.crystal.use_bumpmap:
            tex_coords = mi.TensorXf(self.crystal.uv_map)
            mesh_params['vertex_texcoords'] = dr.ravel(tex_coords)

        return mesh

    def _project_to_image(self, points: torch.Tensor) -> torch.Tensor:
        """
        Project 3D points to the image plane.
        """
        params = mi.traverse(self.mi_scene)
        sensor = self.mi_scene.sensors()[0]
        film = sensor.film()

        # Create the projection matrix
        prj = mi.perspective_projection(film.size(), film.crop_size(), film.crop_offset(), params['sensor.x_fov'],
                                        sensor.near_clip(), sensor.far_clip())

        # Get the inverse camera world transform
        wti = sensor.world_transform().inverse()

        # Project the points
        if points.ndim == 1:
            points = points[None, :]
        points_homogeneous = torch.cat([points, torch.ones(len(points), 1, device=points.device)], dim=1)
        hpp = points_homogeneous @ torch.tensor((prj @ wti).matrix, device=device)[0].T
        pp = hpp[:, :3] / hpp[:, 3][:, None]
        dim = torch.tensor(film.crop_size(), device=device)[None, :]
        uv = pp[:, :2] * dim
        return uv


def build_mitsuba_mesh(crystal: Crystal) -> mi.Mesh:
    """
    Convert a Crystal object into a Mitsuba mesh.
    """
    # Build the mesh in pytorch and convert the parameters to Mitsuba format
    vertices, faces = crystal.build_mesh()
    nv, nf = len(vertices), len(faces)
    vertices = mi.TensorXf(vertices)
    faces = mi.TensorXi64(faces)

    # Set up the material properties
    material_bsdf = {
        'type': 'roughdielectric',
        'distribution': 'beckmann',
        'alpha': mi.ScalarFloat(crystal.material_roughness),
        'int_ior': mi.ScalarFloat(crystal.material_ior),
    }
    if crystal.use_bumpmap:
        bsdf = {
            'type': 'bumpmap',
            'texture': {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(mi.TensorXf(crystal.bumpmap)),
                'wrap_mode': 'clamp',
                'raw': True
            },
            'bsdf': material_bsdf
        }
    else:
        bsdf = material_bsdf
    props = mi.Properties()
    props[Scene.BSDF_KEY] = mi.load_dict(bsdf)

    # Construct the mitsuba mesh and set the vertex positions and faces
    mesh = mi.Mesh(
        Scene.SHAPE_NAME,
        vertex_count=nv,
        face_count=nf,
        has_vertex_normals=False,
        has_vertex_texcoords=crystal.use_bumpmap,
        props=props
    )
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['faces'] = dr.ravel(faces)

    # Update the texture coordinates if a bumpmap is used
    if crystal.use_bumpmap:
        tex_coords = mi.TensorXf(crystal.uv_map)
        mesh_params['vertex_texcoords'] = dr.ravel(tex_coords)

    return mesh


def render_crystal_scene(
        crystal: Optional[Crystal] = None,
        scene: Optional[Scene] = None,
        **kwargs
) -> np.ndarray:
    """
    Render a crystal scene.
    """
    assert not (crystal is None and scene is None), 'Either a crystal or scene must be provided.'
    assert not (crystal is not None and scene is not None), 'Only one of a crystal or scene can be provided.'
    if crystal is not None:
        scene = Scene(crystal=crystal, **kwargs)
    img = scene.render()
    return img


def _render_batch(
        batch_idx: int,
        img_start_idx: int,
        obj_path: Path,
        n_imgs: int,
        settings_path: Path,
        tmp_dir: Path,
        images_dir: Path,
        lock: Lock,
        quiet_render: bool = False
):
    """
    Render a batch of crystals to images.
    """
    tmp_dir = tmp_dir / f'tmp_{batch_idx:010d}'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    assert images_dir.exists(), f'Images dir does not exist! ({images_dir})'

    # Copy over the settings file and update the parameters
    shutil.copy(settings_path, tmp_dir / 'vcw_settings.json')
    settings_path = tmp_dir / 'vcw_settings.json'
    settings = CrystalWellSettings()
    settings.from_json(settings_path)
    settings.path = settings_path
    settings.settings_dict['number_images'] = n_imgs
    settings.settings_dict['crystal_import_path'] = str(obj_path.absolute())
    settings.settings_dict['output_path'] = str(tmp_dir.absolute())
    settings.write_json()

    # Render
    logger.info(f'Batch {batch_idx}: Rendering {n_imgs} crystals from {obj_path} to {tmp_dir}.')
    try:
        blender_render(settings, attempts=1, quiet=quiet_render, seed=SEED + batch_idx)
    except RenderError as e:
        logger.warning(f'Rendering failed! {e}')
        logger.info(f'Adding idx={str(img_start_idx + e.idx)} details to {str(tmp_dir.parent / "errored.json")}')
        append_json_mp(
            tmp_dir.parent.parent / 'errored.json',
            {img_start_idx + e.idx: {
                'batch_idx': batch_idx,
                'obj_idx': e.idx,
                'obj_file': str(obj_path.name),
                'img_file': f'{img_start_idx + e.idx:010d}.png'
            }}, lock)
        logger.info(f'Added idx={str(img_start_idx + e.idx)} details to {str(tmp_dir.parent / "errored.json")}')
        raise e

    # Rename the images and json files to start at the correct index
    logger.info(f'Batch {batch_idx}: Collating results.')
    image_files = list(tmp_dir.glob('*.png'))
    image_files = sorted(image_files)
    json_files = list(tmp_dir.glob('*.png.json'))
    json_files = sorted(json_files)
    n_files = len(image_files)
    assert len(json_files) == n_files, \
        f'Batch {batch_idx}: Number of images and json files do not match! ({len(image_files)} vs {len(json_files)})'
    for i, (img, j) in enumerate(zip(image_files, json_files)):
        img.rename(images_dir / f'{img_start_idx + i:010d}.png')  # Move images into the images directory
        j.rename(tmp_dir / f'{img_start_idx + i:010d}.png.json2')

    # Combine the json parameter files
    json_files = list(tmp_dir.glob('*.json2'))
    json_files = sorted(json_files)
    assert len(json_files) == n_files
    segmentations = {}
    params = {}
    for j in json_files:
        with open(j) as f:
            data = json.load(f)
            assert j.stem not in segmentations
            assert j.stem not in params
            assert len(data['segmentation']) == 1
            assert len(data['crystals']['locations']) == 1
            assert len(data['crystals']['scales']) == 1
            assert len(data['crystals']['rotations']) == 1
            segmentations[j.stem] = data['segmentation'][0]
            params[j.stem] = {
                'location': data['crystals']['locations'][0],
                'scale': data['crystals']['scales'][0],
                'rotation': data['crystals']['rotations'][0],
                'material': {
                    'brightness': data['materials']['brightnesses'][0],
                    'ior': data['materials']['iors'][0],
                    'roughness': data['materials']['roughnesses'][0],
                },
                'light': data['light']
            }

    # Write the combined segmentations and parameters to json files
    append_json(tmp_dir.parent / f'segmentations_{batch_idx:010d}.json', segmentations)
    append_json(tmp_dir.parent / f'params_{batch_idx:010d}.json', params)

    # Move the blender files
    blend_files = list(tmp_dir.glob('*.blend'))
    if len(blend_files):
        blend_dir = images_dir.parent / 'blender'
        assert blend_dir.exists(), f'Blender files dir does not exist! ({blend_dir})'
        blend_files = sorted(blend_files)
        assert len(blend_files) == n_files
        for i, ble in enumerate(blend_files):
            ble.rename(blend_dir / f'{img_start_idx + i:010d}.blend')

    # Clean up
    shutil.rmtree(tmp_dir)


def _render_batch_wrapper(args):
    return _render_batch(*args)


class CrystalRendererMitsuba(CrystalRenderer):
    def __init__(
            self,
            param_path: Path,
            dataset_args: DatasetSyntheticArgs,
            renderer_args: RendererArgs,
            quiet_render: bool = False
    ):
        super().__init__(dataset_args, renderer_args, quiet_render)
        self.param_path = param_path
        self.rendering_params_path = self.param_path.parent / 'rendering_parameters.json'
        self._init_crystal_settings()
        self._load_parameters()

    @property
    def images_dir(self) -> Path:
        return self.param_path.parent / 'images'

    def _init_crystal_settings(self):
        """
        Initialise the crystal settings.
        """
        reader = EntryReader()
        crystal = reader.crystal(self.dataset_args.crystal_id)
        self.lattice_unit_cell = [crystal.cell_lengths[0], crystal.cell_lengths[1], crystal.cell_lengths[2]]
        self.lattice_angles = [crystal.cell_angles[0], crystal.cell_angles[1], crystal.cell_angles[2]]
        self.point_group_symbol = '222'  # crystal.spacegroup_symbol

        # Parse the constraint string to get the miller indices
        constraints_parts = self.dataset_args.distance_constraints.split('>')
        hkls = []
        for i, k in enumerate(constraints_parts):
            if len(k) == 3:
                hkls.append(tuple(int(idx) for idx in k))
        self.miller_indices: List[Tuple[int, int, int]] = hkls
        # miller_indices = [(1, 0, 1), (0, 2, 1), (0, 1, 0)]

    def _load_parameters(self):
        """
        Load the crystal parameters from the parameter file.
        """
        assert self.param_path.exists(), f'Parameter file "{self.param_path}" does not exist.'
        logger.info(f'Loading crystal parameters from {self.param_path} and {self.rendering_params_path}.')

        # Load rendering parameters if they exist
        if self.rendering_params_path.exists():
            with open(self.rendering_params_path, 'r') as f:
                self.rendering_params = json.load(f)
        else:
            self.rendering_params = {}

        # Load the crystal parameters
        with open(self.param_path, 'r') as f:
            reader = csv.DictReader(f)
            headers_distances = {tuple(int(hkl) for hkl in h.split('_')[1]): h for h in reader.fieldnames if
                                 h[0] == 'd'}
            assert all(hkl in headers_distances for hkl in self.miller_indices), 'Missing distance headers!'
            self.data = {}
            for i, row in enumerate(reader):
                idx = int(row['idx'])
                assert i == idx, f'Missing row {i}!'
                self.data[idx] = {
                    'image': row['image'],
                    'distances': {hkl: float(row[headers_distances[hkl]]) for hkl in self.miller_indices},
                    'rendered': idx in self.rendering_params
                }

    def _load_crystal(self, idx: int, scale_init: float = 1) -> Crystal:
        """
        Load the crystal object from the parameters.
        """
        ra = self.renderer_args
        params = self.data[idx]

        # Sample material parameters
        ior = np.random.uniform(ra.min_ior, ra.max_ior)
        roughness = np.random.uniform(ra.min_roughness, ra.max_roughness)

        return self._init_crystal(
            distances=list(params['distances'].values()),
            scale=scale_init,
            material_roughness=roughness,
            material_ior=ior,
            use_bumpmap=ra.crystal_bumpmap_dim > 0,
        )

    def _init_crystal(
            self,
            distances: List[float],
            scale: float = 1.,
            origin: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            material_roughness: float = 0.05,
            material_ior: float = 1.5,
            use_bumpmap: bool = False
    ):
        """
        Instantiate a crystal from the parameters.
        """
        crystal = Crystal(
            lattice_unit_cell=self.lattice_unit_cell,
            lattice_angles=self.lattice_angles,
            miller_indices=self.miller_indices,
            point_group_symbol=self.point_group_symbol,
            scale=scale,
            distances=distances,
            origin=origin,
            rotation=rotation,
            rotation_mode=self.dataset_args.rotation_mode,
            material_roughness=material_roughness,
            material_ior=material_ior,
            use_bumpmap=use_bumpmap,
            # bumpmap_dim=ra.crystal_bumpmap_dim
        )
        crystal.to(device)
        return crystal

    def _make_bubbles(self) -> List[Bubble]:
        """
        Generate some bubbles to go in the growth cell.
        """
        bubbles = []
        ra = self.renderer_args
        n_bubbles = np.random.randint(ra.min_bubbles, ra.max_bubbles + 1)
        for i in range(n_bubbles):
            bubble = Bubble(
                shape_name=f'bubble_{i}',
                roughness=np.random.uniform(ra.bubbles_min_roughness, ra.bubbles_max_roughness),
                ior=np.random.uniform(ra.bubbles_min_ior, ra.bubbles_max_ior),
            )
            bubble.to(device)
            bubbles.append(bubble)
        return bubbles

    def render(self):
        """
        Render all crystal objects to images.
        """
        # if self.n_workers > 1:
        #     self._render_parallel()
        #     return
        bs = self.dataset_args.batch_size

        # Make batches of entries that need rendering
        idxs = [idx for idx in self.data.keys() if not self.data[idx]['rendered']]
        batches = [idxs[i:i + bs] for i in range(0, len(idxs), bs)]
        logger.info(f'Rendering {len(idxs)} crystals in {len(batches)} batches of size {bs}.')

        # Create a temporary output directory
        output_dir = self.images_dir.parent / 'tmp_output'
        output_dir.mkdir(exist_ok=True)

        # Loop over batches
        for i, batch_idxs in enumerate(batches):
            logger.info(f'Rendering batch {i + 1}/{len(batches)}')
            param_batch = {idx: self.data[idx] for idx in batch_idxs}
            self._render_batch(output_dir, param_batch, seed=SEED + i)

        # Remove the batch output directory
        output_dir.rmdir()

    def render_from_parameters(self, params: dict) -> np.ndarray:
        """
        Render a single crystal image from parameters.
        """
        crystal = self._init_crystal(
            distances=params['crystal']['distances'],
            scale=params['crystal']['scale'],
            origin=params['crystal']['origin'],
            rotation=params['crystal']['rotation'],
            material_roughness=params['crystal']['material_roughness'],
            material_ior=params['crystal']['material_ior'],
            use_bumpmap=params['crystal']['use_bumpmap']
        )

        bubbles = []
        if 'bubbles' in params:
            for i, bubble_params in enumerate(params['bubbles']):
                bubble = Bubble(
                    shape_name=f'bubble_{i}',
                    origin=bubble_params['origin'],
                    scale=bubble_params['scale'],
                    roughness=bubble_params['roughness'],
                    ior=bubble_params['ior'],
                )
                bubble.to(device)
                bubbles.append(bubble)

        # Create and render the scene
        scene = Scene(
            crystal=crystal,
            bubbles=bubbles,
            res=self.dataset_args.image_size,
            light_radiance=params['light_radiance'],
            **self.renderer_args.to_dict(),
        )
        img = scene.render(seed=params['seed'])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # todo: do we need this?

        return img

    def _render_parallel(self, obj_paths: Optional[list] = None):
        """
        Render all crystal objects to images using parallel processing.
        """
        raise NotImplementedError('To do!')

    def _render_batch(
            self,
            output_dir: Path,
            param_batch: dict,
            seed: Optional[int] = None
    ):
        """
        Render a batch of crystals to images.
        """
        assert not any(output_dir.iterdir()), f'Output dir not empty before batch render! ({output_dir})'
        da = self.dataset_args
        ra = self.renderer_args
        rendering_params = {}
        segmentations = {}
        scale_init = 1

        for idx, params in param_batch.items():
            try:
                # Sample the light radiance
                light_radiance = np.random.uniform(ra.light_radiance_min,
                                                   ra.light_radiance_max)

                # Create and render the scene
                scene = Scene(
                    crystal=self._load_crystal(idx, scale_init),
                    bubbles=self._make_bubbles(),
                    res=da.image_size,
                    light_radiance=light_radiance,
                    **ra.to_dict(),
                )
                scene.place_crystal(
                    min_area=da.min_area,
                    max_area=da.max_area,
                    centre_crystal=da.centre_crystals,
                    min_x=ra.crystal_min_x,
                    max_x=ra.crystal_max_x,
                    min_y=ra.crystal_min_y,
                    max_y=ra.crystal_max_y,
                )
                scene.place_bubbles(
                    min_x=ra.bubbles_min_x,
                    max_x=ra.bubbles_max_x,
                    min_y=ra.bubbles_min_y,
                    max_y=ra.bubbles_max_y,
                    min_scale=ra.bubbles_min_scale,
                    max_scale=ra.bubbles_max_scale,
                )
                scale_init = scene.crystal.scale.item()
                img = scene.render(seed=seed + idx)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # todo: do we need this?
                cv2.imwrite(str(output_dir / params['image']), img)
                scene_params = scene.to_dict()
                rendering_params[params['image']] = {
                    'seed': seed + idx,
                    'light_radiance': scene_params['light_radiance'].tolist(),
                    'crystal': scene_params['crystal'],
                }
                if 'bubbles' in scene_params:
                    rendering_params[params['image']]['bubbles'] = scene_params['bubbles']
                segmentations[params['image']] = scene.get_crystal_image_coords().tolist()

            except RenderError as e:
                append_json(output_dir.parent / 'errored.json', {params['image']: str(e)})
                raise e

        # Move images into the images directory
        image_files = list(output_dir.glob('*.png'))
        for img in image_files:
            img.rename(self.images_dir / img.name)

        # Write the combined segmentations and parameters to json files
        append_json(self.images_dir.parent / 'segmentations.json', segmentations)
        append_json(self.images_dir.parent / 'rendering_parameters.json', rendering_params)

        assert not any(output_dir.iterdir()), f'Output dir not properly emptied after batch render! ({output_dir})'

    def annotate_image(self, image_idx: int = 0):
        """
        Annotate the first image with the projected vertices and save to disk
        """
        imgs = list(self.images_dir.glob('*.png'))
        imgs = sorted(imgs)
        img0_path = imgs[image_idx]
        img0 = np.array(Image.open(img0_path))

        with open(self.images_dir.parent / 'segmentations.json') as f:
            segmentations = json.load(f)
        seg = np.array(segmentations[img0_path.name])

        # Plot the image with segmentation overlay
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(np.flipud(img0))
        ax.scatter(seg[:, 0], seg[:, 1], marker='x', c='r', s=50)
        fig.tight_layout()
        plt.savefig(self.images_dir.parent / f'segmentation_example_{img0_path.stem}.png')
        plt.close(fig)
