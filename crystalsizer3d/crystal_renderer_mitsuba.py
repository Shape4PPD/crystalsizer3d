import json
import os
import shutil
from multiprocessing import Lock
from pathlib import Path
from typing import Optional

import drjit as dr
import mitsuba as mi
import numpy as np
import torch

from crystalsizer3d import N_WORKERS, USE_CUDA, logger
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs
from crystalsizer3d.args.renderer_args import RendererArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.util.utils import SEED

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('llvm_ad_rgb')
    device = torch.device('cpu')

from mitsuba import ScalarTransform4f as T

SHAPE_NAME = 'crystal'
VERTEX_KEY = SHAPE_NAME + '.vertex_positions'
FACES_KEY = SHAPE_NAME + '.faces'
BSDF_KEY = SHAPE_NAME + '.bsdf'
COLOUR_KEY = BSDF_KEY + '.reflectance.value'


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
    bsdf = {
        'type': 'roughdielectric',
        'distribution': 'beckmann',
        'alpha': 0.02,
        'int_ior': 1.78,
    }
    props = mi.Properties()
    props[BSDF_KEY] = mi.load_dict(bsdf)

    mesh_transform = mi.Transform4f(np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ]))

    # Construct the mitsuba mesh and set the vertex positions and faces
    mesh = mi.Mesh(
        SHAPE_NAME,
        vertex_count=nv,
        face_count=nf,
        has_vertex_normals=False,
        has_vertex_texcoords=False,
        props=props
    )
    mesh_params = mi.traverse(mesh)
    # mesh_params['vertex_positions'] = dr.ravel(mesh_transform @ vertices)
    vertices = dr.unravel(mi.Point3f, dr.ravel(vertices))
    mesh_params['vertex_positions'] = dr.ravel(mesh_transform @ vertices)
    # mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['faces'] = dr.ravel(faces)

    # current_vertex_positions = dr.unravel(mi.Point3f, mesh_params['vertex_positions'])
    # mesh_params['vertex_positions'] = dr.ravel(trafo @ current_vertex_positions)

    # current_vertex_positions = dr.unravel(mi.Point3f, params[vertices_key])
    # params[vertices_key] = dr.ravel(trafo @ current_vertex_positions)

    return mesh


def create_scene(crystal: Crystal, spp: int = 256, res: int = 400) -> mi.Scene:
    """
    Create a Mitsuba scene containing the given crystal.
    """
    sensor_transform = T(np.array([
        [-1, 0, 0, 0],
        [0, 0, -1, 100],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ]))

    light_transform = T(np.array([
        [50, 0, 0, 0],
        [0, 0, 50, -18.75],
        [0, 50, 0, 0],
        [0, 0, 0, 1]
    ]))

    surface_transform = T(np.array([
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ]))

    scene = mi.load_dict({
        'type': 'scene',

        # Camera and rendering parameters
        'integrator': {
            'type': 'prb_projective',
            'max_depth': 64,
            'rr_depth': 5,
            'sppi': 0,
            'guiding': 'grid',
            'guiding_rounds': 10
        },
        'sensor': {
            'type': 'perspective',
            'near_clip': 0.001,
            'far_clip': 1000,
            'fov': 27,
            # 'to_world': T.look_at(
            #     origin=[0, 0, 100],
            #     target=[0, 0, 0],
            #     up=[0, 1, 0]
            # ),
            'to_world': sensor_transform,
            'sampler': {
                # 'type': 'independent',
                'type': 'stratified',  # seems better than independent
                # 'type': 'multijitter',  # better than indep, but maybe worse than strat
                # 'type': 'orthogonal',  # diverges a bit like indep
                # 'type': 'ldsampler',  # seems decent
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': res,
                'height': res,
                'filter': {'type': 'gaussian'},
                'sample_border': True,
            },
        },

        # Emitters
        'light': {
            'type': 'rectangle',
            'to_world': light_transform,
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': 0.75
                }
            },
        },

        # Shapes
        # 'surface': {
        #     'type': 'rectangle',
        #     'to_world': surface_transform,
        #     'surface_material': {
        #         'type': 'dielectric',
        #         'int_ior': 1.,
        #     },
        # },
        SHAPE_NAME: build_mitsuba_mesh(crystal)
    })

    return scene


def render_crystal_scene(
        crystal: Crystal,
        spp: int = 256,
        res: int = 400,
) -> np.ndarray:
    """
    Render a crystal scene.
    """
    scene = create_scene(crystal=crystal, spp=spp, res=res)
    img = mi.render(scene)
    img = np.array(mi.util.convert_to_bitmap(img))
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


class CrystalRenderer:
    def __init__(
            self,
            obj_dir: Path,
            dataset_args: DatasetSyntheticArgs,
            renderer_args: RendererArgs,
            quiet_render: bool = False
    ):
        self.obj_dir = obj_dir
        self.dataset_args = dataset_args
        self.renderer_args = renderer_args
        self.images_dir = self.obj_dir.parent / 'images'
        self.blend_dir = self.obj_dir.parent / 'blender'
        self.quiet_render = quiet_render
        # self._init_settings()
        if N_WORKERS > 0:
            self.n_workers = N_WORKERS
        else:
            self.n_workers = len(os.sched_getaffinity(0))

    def render(self, obj_paths: Optional[list] = None):
        """
        Render all crystal objects to images.
        """
        raise NotImplementedError('To do!')

    def _render_parallel(self, obj_paths: Optional[list] = None):
        """
        Render all crystal objects to images using parallel processing.
        """
        raise NotImplementedError('To do!')

    def _render_batch(self, output_dir: Path, start_idx: int = 0, batch_idx: int = 0, seed: Optional[int] = None):
        """
        Render a batch of crystals to images.
        """
        raise NotImplementedError('To do!')
