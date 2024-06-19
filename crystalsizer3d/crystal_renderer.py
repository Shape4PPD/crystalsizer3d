import csv
import json
import multiprocessing as mp
import os
import re
import shutil
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
from PIL import Image
from filelock import SoftFileLock as FileLock, Timeout

from crystalsizer3d import MI_CPU_VARIANT, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.nn.dataset import PARAMETER_HEADERS
from crystalsizer3d.scene_components.bubble import Bubble, make_bubbles
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.textures import NoiseTexture, NormalMapNoiseTexture, generate_crystal_bumpmap
from crystalsizer3d.scene_components.utils import RenderError
from crystalsizer3d.util.geometry import merge_vertices
from crystalsizer3d.util.utils import get_seed, hash_data, to_numpy

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant(MI_CPU_VARIANT)
    device = torch.device('cpu')


def append_to_shared_json(file_path: Path, new_data: dict, timeout: int = 60):
    """
    Append new data to a shared JSON file.
    """
    if not file_path.exists():
        with open(file_path, 'w') as f:
            json.dump({}, f)
    lock = FileLock(file_path.with_suffix('.lock'), timeout=timeout)
    lock.acquire()
    with open(file_path, 'r') as f:
        data = json.load(f)
    if len(data) > 0:
        for k in new_data.keys():
            if k in data and hash_data(data[k]) != hash_data(new_data[k]):
                raise ValueError(f'Key "{k}" already exists in {file_path} and is not the same!')
    data.update(new_data)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    lock.release()


def combine_json_files(master_path: Path, batches_dir: Path):
    """
    Combine JSON files in the batches directory into the master file.
    """
    if not batches_dir.exists():
        logger.warning(f'No batches directory found at {batches_dir}. Skipping collation.')
        return
    batch_files = list(batches_dir.glob('*.json'))
    if len(batch_files) == 0:
        logger.warning(f'No batch files found in {batches_dir}. Skipping collation.')
        return

    # Lock the master file
    lock = FileLock(master_path.with_suffix('.lock'), timeout=0)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        logger.warning(f'Could not acquire lock for {master_path}. Skipping collation.')
        return

    # Load any existing data
    if master_path.exists():
        with open(master_path, 'r') as f:
            data = json.load(f)
        data = {int(k): v for k, v in data.items()}
    else:
        data = {}

    # Collate the batch data
    logger.info(f'Collating results from {len(batch_files)} batches to {master_path}.')
    for i, batch_file in enumerate(batch_files):
        with open(batch_file, 'r') as f:
            data_batch = json.load(f)
        data_batch = {int(k): v for k, v in data_batch.items()}
        for idx, params in data_batch.items():
            if idx in data and hash_data(params) != hash_data(data[idx]):
                raise ValueError(f'Image "{idx}" from {batch_file} already exists '
                                 f'in {master_path} and is not the same!')
        data.update(data_batch)

    # Rebuild the data with sorted keys and write to disk
    data = {k: data[k] for k in sorted(list(data.keys()))}
    with open(master_path, 'w') as f:
        json.dump(data, f, indent=4)

    # Remove the merged batch files and directory
    for batch_file in batch_files:
        batch_file.unlink()
    batch_files = list(batches_dir.glob('*.json'))
    if len(batch_files) == 0:
        batches_dir.rmdir()
    else:
        logger.warning(f'Found more batch files in {batches_dir} after collating - will need re-running!')
    lock.release()


def _initialise_crystal(
        params: dict,
        dataset_args: DatasetSyntheticArgs,
        scale_init: float = 1,
        device: torch.device = torch.device('cpu'),
) -> Crystal:
    """
    Initialise a crystal object from the parameters.
    """
    da = dataset_args

    # Sample material parameters
    ior = np.random.uniform(da.min_ior, da.max_ior)
    roughness = np.random.uniform(da.min_roughness, da.max_roughness)

    # Create the crystal
    crystal = Crystal(
        **params,
        scale=scale_init,
        rotation_mode=da.rotation_mode,
        material_roughness=roughness,
        material_ior=ior,
        use_bumpmap=da.crystal_bumpmap_dim > 0,
        bumpmap_dim=da.crystal_bumpmap_dim
    )
    crystal.to(device)

    # Create the crystal bumpmap
    if da.crystal_bumpmap_dim > 0:
        n_defects = np.random.randint(da.min_defects, da.max_defects + 1)
        crystal.bumpmap.data = generate_crystal_bumpmap(
            crystal=crystal,
            n_defects=n_defects,
            defect_min_width=da.defect_min_width,
            defect_max_width=da.defect_max_width,
            defect_max_z=da.defect_max_z,
        )

    return crystal


def _render_batch(
        crystal_params: dict,
        param_batch: dict,
        batch_idx: int,
        n_batches: int,
        dataset_args: DatasetSyntheticArgs,
        root_dir: Path,
        output_dir: Optional[Path] = None,
        worker_id: Optional[str] = None,
        stale_worker_timeout: int = 1200,
):
    """
    Render a batch of crystals to images.
    """
    da = dataset_args
    seed = get_seed() + batch_idx
    worker_key = f'{worker_id}_{batch_idx:06d}'
    assert root_dir.exists(), f'Root dir does not exist! ({root_dir})'

    # Check that no other script is processing the same idxs
    timestamp = time.time()
    comlog_path = root_dir / 'comlog.json'
    comlog_lock = FileLock(comlog_path.with_suffix('.lock'), timeout=60)
    comlog_lock.acquire()
    if not comlog_path.exists():
        logger.info(f'Making comlog file at {comlog_path}.')
        with open(comlog_path, 'w') as f:
            json.dump({'workers': {}, 'completed_idxs': []}, f)
    with open(comlog_path, 'r') as f:
        comlog = json.load(f)

    # Prune any workers that have not been active recently
    if len(comlog['workers']) > 0:
        for k, worker in list(comlog['workers'].items()):
            if timestamp - worker['last_active'] > stale_worker_timeout:
                logger.warning(f'Worker {k} has timed out. Removing from log.')
                del comlog['workers'][k]

    # Filter out idxs that have already been processed
    param_batch = {idx: params for idx, params in param_batch.items() if idx not in comlog['completed_idxs']}

    # Filter out idxs that are currently being processed
    if len(comlog['workers']) > 0:
        running_idxs = np.concatenate([worker['idxs'] for worker in comlog['workers'].values()])
        param_batch = {idx: params for idx, params in param_batch.items() if idx not in running_idxs}

    # Add the worker to the comlog
    if len(param_batch) > 0:
        logger.info(f'Adding worker details to comlog with key {worker_key}.')
        comlog['workers'].update({worker_key: {'last_active': timestamp, 'idxs': list(param_batch.keys())}})
        with open(comlog_path, 'w') as f:
            json.dump(comlog, f, indent=4)
    comlog_lock.release()
    if len(param_batch) == 0:
        logger.info(f'Batch {batch_idx + 1}/{n_batches}: No crystals to render.')
        return

    # Sort directories
    images_dir = root_dir / 'images'
    params_dir = root_dir / 'rendering_parameters'
    segmentations_dir = root_dir / 'segmentations'
    vertices_dir = root_dir / 'vertices'
    crystal_bumpmaps_dir = root_dir / 'crystal_bumpmaps'
    clean_images_dir = root_dir / 'images_clean'
    images_dir.mkdir(exist_ok=True)
    params_dir.mkdir(exist_ok=True)
    segmentations_dir.mkdir(exist_ok=True)
    vertices_dir.mkdir(exist_ok=True)
    if da.crystal_bumpmap_dim > 0:
        crystal_bumpmaps_dir.mkdir(exist_ok=True)
    if da.generate_clean:
        clean_images_dir.mkdir(exist_ok=True)
    if output_dir is None:
        dirname = f'tmp_output_{worker_id}' if worker_id is not None else 'tmp_output'
        output_dir = root_dir / dirname
    else:
        dirname = f'tmp_{worker_id}_{batch_idx:010d}' if worker_id is not None else f'tmp_{batch_idx:010d}'
        output_dir = output_dir / dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the scene
    scene = Scene(res=da.image_size, **da.to_dict())

    # Render
    logger.info(f'Batch {batch_idx + 1}/{n_batches}: Rendering {len(param_batch)} crystals to {output_dir}.')
    rendering_params = {}
    segmentations = {}
    vertices = {}
    scale_init = 1
    for idx, params in param_batch.items():
        try:
            # Initialise the crystal
            crystal_params_i = crystal_params.copy()
            crystal_params_i['distances'] = list(params['distances'].values())
            scene.crystal = _initialise_crystal(crystal_params_i, da, scale_init, device=torch.device('cpu'))

            # Create the bubbles
            if da.max_bubbles > 0:
                scene.bubbles = make_bubbles(
                    n_bubbles=np.random.randint(da.min_bubbles, da.max_bubbles + 1),
                    min_roughness=da.bubbles_min_roughness,
                    max_roughness=da.bubbles_max_roughness,
                    min_ior=da.bubbles_min_ior,
                    max_ior=da.bubbles_max_ior,
                    device=torch.device('cpu'),
                )
            else:
                scene.bubbles = []

            # Sample the light radiance
            scene.light_radiance = np.random.uniform(da.light_radiance_min, da.light_radiance_max)

            # Randomise the light texture
            if da.light_texture_dim > -1:
                scene.light_st_texture = NoiseTexture(
                    dim=da.light_texture_dim,
                    channels=3,
                    perlin_freq=np.random.uniform(da.light_perlin_freq_min, da.light_perlin_freq_max),
                    perlin_octaves=np.random.randint(da.light_perlin_octaves_min, da.light_perlin_octaves_max + 1),
                    white_noise_scale=np.random.uniform(da.light_white_noise_scale_min, da.light_white_noise_scale_max),
                    max_amplitude=np.random.uniform(da.light_noise_amplitude_min, da.light_noise_amplitude_max),
                    zero_centred=True,
                    shift=1.,
                    seed=seed
                )

            # Randomise the cell bumpmap
            if da.cell_bumpmap_dim > -1:
                scene.cell_bumpmap = NormalMapNoiseTexture(
                    dim=da.cell_bumpmap_dim,
                    perlin_freq=np.random.uniform(da.cell_perlin_freq_min, da.cell_perlin_freq_max),
                    perlin_octaves=np.random.randint(da.cell_perlin_octaves_min, da.cell_perlin_octaves_max + 1),
                    white_noise_scale=np.random.uniform(da.cell_white_noise_scale_min, da.cell_white_noise_scale_max),
                    max_amplitude=np.random.uniform(da.cell_noise_amplitude_min, da.cell_noise_amplitude_max),
                    seed=seed
                )
                scene.cell_bumpmap_idx = 1  # Only one bumpmap supported for now

            # Create and render the scene
            with torch.no_grad():
                scene.place_crystal(
                    min_area=da.crystal_area_min,
                    max_area=da.crystal_area_max,
                    centre_crystal=da.centre_crystals,
                    rebuild_scene=False,
                )
                scene.place_bubbles(
                    min_scale=da.bubbles_min_scale,
                    max_scale=da.bubbles_max_scale,
                    rebuild_scene=False,
                )
            scale_init = scene.crystal.scale.item()
            scene.crystal.to(device)
            for bubble in scene.bubbles:
                bubble.to(device)
            scene.build_mi_scene()
            img = scene.render(seed=seed + idx)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # todo: do we need this?
            cv2.imwrite(str(output_dir / params['image']), img)
            scene_params = scene.to_dict()
            rendering_params[idx] = {
                'seed': seed + idx,
                'light_radiance': scene_params['light_radiance'].tolist(),
                'light_st_texture': scene_params['light_st_texture'],
                'cell_bumpmap': scene_params['cell_bumpmap'],
                'cell_bumpmap_idx': scene_params['cell_bumpmap_idx'],
                'crystal': scene_params['crystal'],
            }
            if 'bubbles' in scene_params:
                rendering_params[idx]['bubbles'] = scene_params['bubbles']
            segmentations[idx] = scene.get_crystal_image_coords().tolist()
            vertices[idx] = scene.crystal.vertices.tolist()

            # Save crystal bumpmap
            if da.crystal_bumpmap_dim > 0:
                assert scene.crystal.bumpmap is not None, f'No bumpmap found for crystal {idx}!'
                tex_path = output_dir / f'{params["image"][:-4]}_bumpmap.npz'
                np.savez_compressed(tex_path, data=to_numpy(scene.crystal.bumpmap.data))

            # Save a clean version of the image without bubbles and no bumpmap if required
            if da.generate_clean:
                scene.clear_interference()
                img_clean = scene.render(seed=seed + idx)
                img_clean = cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_dir / params['image']) + '_clean', img_clean)

        except RenderError as e:
            logger.warning(f'Rendering failed! {e}')
            logger.info(f'Adding idx={params["image"]} details to {str(output_dir.parent / "errored.json")}')
            append_to_shared_json(root_dir / 'errored.json', {params['image']: str(e)})
            raise e

        # Update the comlog to show that the worker is still active
        timestamp = time.time()
        if timestamp - comlog['workers'][worker_key]['last_active'] > stale_worker_timeout / 2:
            comlog_lock.acquire()
            with open(comlog_path, 'r') as f:
                comlog = json.load(f)
            comlog['workers'][worker_key]['last_active'] = timestamp
            with open(comlog_path, 'w') as f:
                json.dump(comlog, f, indent=4)
            comlog_lock.release()

    # Collate results
    logger.info(f'Batch {batch_idx + 1}/{n_batches}: Collating results.')

    # Move images into the images directory
    image_files = list(output_dir.glob('*.png'))
    for img in image_files:
        img.rename(images_dir / img.name)

    # Move crystal bumpmap textures
    tex_files = list(output_dir.glob(f'*_bumpmap.npz'))
    for tex in tex_files:
        tex.rename(crystal_bumpmaps_dir / (tex.name.split('_')[0] + '.npz'))

    # Move the clean images into the clean images directory
    clean_image_files = list(output_dir.glob('*_clean'))
    for img in clean_image_files:
        img.rename(clean_images_dir / (img.stem + '.png'))

    # Write the parameters, segmentations and vertices to json files
    with open(params_dir / f'{worker_key}.json', 'w') as f:
        json.dump(rendering_params, f, indent=4)
    with open(segmentations_dir / f'{worker_key}.json', 'w') as f:
        json.dump(segmentations, f, indent=4)
    with open(vertices_dir / f'{worker_key}.json', 'w') as f:
        json.dump(vertices, f, indent=4)

    # Update the comlog
    comlog_lock.acquire()
    with open(comlog_path, 'r') as f:
        comlog = json.load(f)
    comlog['completed_idxs'].extend(list(param_batch.keys()))
    comlog['completed_idxs'] = sorted(comlog['completed_idxs'])
    del comlog['workers'][worker_key]
    with open(comlog_path, 'w') as f:
        json.dump(comlog, f, indent=4)
    comlog_lock.release()

    # Clean up
    shutil.rmtree(output_dir)


def _render_batch_wrapper(args):
    return _render_batch(**args)


class CrystalRenderer:
    def __init__(
            self,
            param_path: Path,
            dataset_args: DatasetSyntheticArgs,
            quiet_render: bool = False,
            n_workers: int = 1,
            remove_mismatched: bool = False,
            migrate_distances: bool = True,
    ):
        self.dataset_args = dataset_args
        self.quiet_render = quiet_render
        self.n_workers = n_workers
        self.param_path = param_path
        self.root_dir = self.param_path.parent
        self.comlog_path = self.root_dir / 'comlog.json'
        self.comlog_lock = FileLock(self.comlog_path.with_suffix('.lock'))
        self.images_dir = self.root_dir / 'images'
        self.rendering_params_dir = self.root_dir / 'rendering_parameters'
        self.rendering_params_path = self.root_dir / 'rendering_parameters.json'
        self.segmentations_dir = self.root_dir / 'segmentations'
        self.segmentations_path = self.root_dir / 'segmentations.json'
        self.vertices_dir = self.root_dir / 'vertices'
        self.vertices_path = self.root_dir / 'vertices.json'
        self.remove_mismatched = remove_mismatched
        self.migrate_distances = migrate_distances
        self._init_crystal_settings()

    def _init_crystal_settings(self):
        """
        Initialise the crystal settings.
        """
        csd = CSDProxy()
        cs = csd.load(self.dataset_args.crystal_id)
        self.lattice_unit_cell = cs.lattice_unit_cell
        self.lattice_angles = cs.lattice_angles
        self.point_group_symbol = cs.point_group_symbol
        self.miller_indices = self.dataset_args.miller_indices

        # Expand the miller indices to include all faces if the distances are asymmetric
        if self.dataset_args.asymmetry is not None:
            crystal = self._init_crystal([1. for _ in self.miller_indices])
            self.miller_indices = [tuple(hkl) for hkl in crystal.all_miller_indices.tolist()]

    def __getattr__(self, name: str):
        """
        Lazy loading of the data.
        """
        if name in ['rendering_params', 'segmentations', 'vertices', 'data']:
            self._load_parameters(remove_mismatched=self.remove_mismatched)
            return getattr(self, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _load_parameters(self, remove_mismatched: bool = False):
        """
        Load the crystal parameters from the parameter file.
        """
        global device
        assert self.param_path.exists(), f'Parameter file "{self.param_path}" does not exist.'
        logger.info(f'Loading crystal parameters from {self.param_path} and {self.rendering_params_path}.')

        def _load_json(file_path: Path, int_keys: bool = False):
            if file_path.exists():
                lock = FileLock(file_path.with_suffix('.lock'), timeout=60)
                lock.acquire()
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if int_keys:
                    data = {int(k): v for k, v in data.items()}
                lock.release()
            else:
                data = {}
            return data

        def _write_json(file_path: Path, data: dict):
            lock = FileLock(file_path.with_suffix('.lock'), timeout=60)
            lock.acquire()
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            lock.release()

        def _update_comlog_completed_idxs(new_completed_idxs: List[int]):
            self.comlog_lock.acquire()
            with open(self.comlog_path, 'r') as f:
                comlog = json.load(f)
            comlog['completed_idxs'] = new_completed_idxs
            with open(self.comlog_path, 'w') as f:
                json.dump(comlog, f, indent=4)
            self.comlog_lock.release()

        # Load rendering parameters, segmentations and vertices if they exist
        self.rendering_params = _load_json(self.rendering_params_path, int_keys=True)
        self.segmentations = _load_json(self.segmentations_path, int_keys=True)
        self.vertices = _load_json(self.vertices_path, int_keys=True)

        # Load the comlog file if it exists to check for completed idxs
        comlog = _load_json(self.comlog_path)
        if 'completed_idxs' in comlog:
            completed_idxs = comlog['completed_idxs']
        else:
            completed_idxs = []

        # Check that the rendering parameters and segmentations match
        logger.info('Checking rendering parameters idxs match with the segmentations.')
        rp_idxs = set(self.rendering_params.keys())
        seg_idxs = set(self.segmentations.keys())
        broken_keys = rp_idxs.union(seg_idxs) - rp_idxs.intersection(seg_idxs)
        if len(broken_keys) > 0:
            if remove_mismatched and not self.is_active():
                logger.warning(f'Found {len(broken_keys)} mis-matched keys. Removing these and reloading...')
                rp_hash_pre = hash_data(self.rendering_params)
                seg_hash_pre = hash_data(self.segmentations)
                completed_hash_pre = hash_data(completed_idxs)
                for k in broken_keys:
                    if k in self.segmentations:
                        del self.segmentations[k]
                    if k in self.rendering_params:
                        del self.rendering_params[k]
                    if k in completed_idxs:
                        completed_idxs.remove(k)
                rp_hash_post = hash_data(self.rendering_params)
                seg_hash_post = hash_data(self.segmentations)
                completed_hash_post = hash_data(completed_idxs)
                if rp_hash_pre != rp_hash_post:
                    _write_json(self.rendering_params_path, self.rendering_params)
                if seg_hash_pre != seg_hash_post:
                    _write_json(self.segmentations_path, self.segmentations)
                if completed_hash_pre != completed_hash_post:
                    comlog['completed_idxs'] = completed_idxs
                    _update_comlog_completed_idxs(completed_idxs)
                return self._load_parameters()
            else:
                raise ValueError(f'Found {len(broken_keys)} mis-matched keys '
                                 f'between the rendering parameters and segmentations!')

        # Load the crystal parameters
        with open(self.param_path, 'r') as f:
            reader = csv.DictReader(f)
            headers_distances = {tuple(map(int, re.findall(r'-?\d', h.split('_')[1]))): h
                                 for h in reader.fieldnames if h[0] == 'd'}
            assert all(hkl in headers_distances for hkl in self.miller_indices), 'Missing distance headers!'

            include_areas = True
            needs_migrating = False
            headers_areas = {tuple(map(int, re.findall(r'-?\d', h.split('_')[1]))): h
                             for h in reader.fieldnames if h[0] == 'a'}
            if not all(hkl in headers_areas for hkl in self.miller_indices):
                if self.migrate_distances:
                    needs_migrating = True
                else:
                    logger.warning('Missing area parameters (not migrating).')
                include_areas = False

            self.data = {}
            for i, row in enumerate(reader):
                idx = int(row['idx'])
                assert i == idx, f'Missing row {i}!'
                self.data[idx] = {
                    'image': row['image'],
                    'distances': {hkl: float(row[headers_distances[hkl]]) for hkl in self.miller_indices},
                    'si': float(row['si']),
                    'il': float(row['il']),
                    'rendered': idx in self.rendering_params or idx in completed_idxs,
                }
                if include_areas:
                    self.data[idx]['areas'] = {hkl: float(row[headers_areas[hkl]]) for hkl in self.miller_indices}

        # Check the vertices match the rendering parameters
        vert_idxs = set(self.vertices.keys())
        if vert_idxs != rp_idxs:
            logger.warning(f'Vertices and rendering parameters do not match! '
                           f'(#vertices) {len(vert_idxs)} != {len(rp_idxs)} (#rendering parameters). Rebuilding the vertices.')
            device_og = device
            device = torch.device('cpu')
            vertices = {}
            for i, idx in enumerate(rp_idxs):
                if (i + 1) % 100 == 0:
                    logger.info(f'Building vertices for {i + 1}/{len(rp_idxs)} crystals.')
                if idx in self.vertices:
                    vertices[idx] = self.vertices[idx]
                else:
                    r_params = self.rendering_params[idx]
                    crystal = self._init_crystal(
                        distances=r_params['crystal']['distances'],
                        scale=r_params['crystal']['scale'],
                        origin=r_params['crystal']['origin'],
                        rotation=r_params['crystal']['rotation']
                    )
                    vertices[idx] = crystal.vertices.tolist()
            _write_json(self.vertices_path, vertices)
            device = device_og
            return self._load_parameters()

        # Migrate if required
        if needs_migrating:
            logger.info('Migrating distances to the new format.')
            device_og = device
            device = torch.device('cpu')
            n_changed = 0
            for i, (idx, params) in enumerate(self.data.items()):
                if (i + 1) % 100 == 0:
                    logger.info(f'Migrating crystal {i + 1}/{len(self.data)}.')
                distances_og = np.array(list(params['distances'].values()))
                distances_og[distances_og < 0] = 1e8
                crystal_og = self._init_crystal(distances_og)
                r_params = self.rendering_params[idx]

                # Check that all planes are touching the polyhedron
                distances_min = (crystal_og.N @ crystal_og.vertices.T).amax(dim=1)[:len(self.miller_indices)]
                distances_min = to_numpy(distances_min)

                # Renormalise the minimum distances and update scale
                if distances_min.max() < 1:
                    sf = 1 / distances_min.max()
                    distances_min *= sf
                    r_params['crystal']['scale'] /= sf

                # Rebuild crystal to get the correct areas
                if (not np.allclose(distances_min, to_numpy(crystal_og.distances), atol=1e-3)
                        or 'areas' not in params
                        or not np.allclose(params['areas'], to_numpy(crystal_og.areas), atol=1e-3)):
                    # Update distances and areas
                    crystal_min = self._init_crystal(distances_min)
                    self.data[idx]['distances'] = {hkl: crystal_min.distances[j].item() for j, hkl in
                                                   enumerate(self.miller_indices)}
                    self.data[idx]['areas'] = {hkl: crystal_min.areas[hkl] for hkl in self.miller_indices}
                    r_params['crystal']['distances'] = distances_min.tolist()

                    # Check that the vertices match (or thereabouts)
                    crystal_new = self._init_crystal(
                        distances=r_params['crystal']['distances'],
                        scale=r_params['crystal']['scale'],
                        origin=r_params['crystal']['origin'],
                        rotation=r_params['crystal']['rotation']
                    )
                    vertices_og = merge_vertices(torch.tensor(self.vertices[idx]))[0]
                    vertices_new = merge_vertices(crystal_new.vertices)[0]
                    v_dists = torch.cdist(vertices_og, vertices_new)
                    tol = vertices_og.abs().mean() * 1e-2
                    assert v_dists.amin(dim=0).max() < tol and v_dists.amin(dim=1).max() < tol, \
                        f'Vertices do not match for crystal {idx}!'
                    n_changed += 1
            device = device_og
            logger.info('Backing up the old parameters.')
            suffix = f'.backup_{START_TIMESTAMP}'
            self.param_path.rename(self.param_path.with_suffix(suffix))
            self.rendering_params_path.rename(self.rendering_params_path.with_suffix(suffix))
            logger.info('Updating the parameters.')
            with open(self.param_path, 'w') as f:
                headers = PARAMETER_HEADERS.copy()
                ref_idxs = [''.join(str(i) for i in k) for k in self.miller_indices]
                for i, hkl in enumerate(ref_idxs):
                    headers.append(f'd{i}_{hkl}')
                for i, hkl in enumerate(ref_idxs):
                    headers.append(f'a{i}_{hkl}')
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for i, (idx, params) in enumerate(self.data.items()):
                    entry = {
                        'crystal_id': self.dataset_args.crystal_id,
                        'idx': idx,
                        'image': f'{i:010d}.png',
                        'si': params['si'],
                        'il': params['il'],
                    }
                    for j, (hkl_str, hkl) in enumerate(zip(ref_idxs, self.miller_indices)):
                        entry[f'd{j}_{hkl_str}'] = float(params['distances'][hkl])
                        entry[f'a{j}_{hkl_str}'] = float(params['areas'][hkl])
                    writer.writerow(entry)
            _write_json(self.rendering_params_path, self.rendering_params)

            logger.info(f'Migration complete. Changed {n_changed} distances.')
            return self._load_parameters()

    def _init_crystal(
            self,
            distances: List[float],
            scale: float = 1.,
            origin: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            material_roughness: float = 0.05,
            material_ior: float = 1.5,
            bumpmap_dim: int = -1
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
            use_bumpmap=bumpmap_dim > 0,
            bumpmap_dim=bumpmap_dim
        )
        crystal.to(device)
        return crystal

    def render(self):
        """
        Render all crystal objects to images.
        """
        bs = self.dataset_args.batch_size

        # Make batches of entries that need rendering
        idxs = [idx for idx in self.data.keys() if not self.data[idx]['rendered']]
        np.random.shuffle(idxs)
        batches = []
        for i in range(0, len(idxs), bs):
            batches.append({idx: self.data[idx] for idx in idxs[i:i + bs]})
        if len(batches) == 0:
            logger.info('All crystals have been rendered.')
            self.collate_results()
            return
        logger.info(f'Rendering {len(idxs)} crystals in {len(batches)} batches of size {bs}.')
        shared_args = {
            'crystal_params': {
                'lattice_unit_cell': self.lattice_unit_cell,
                'lattice_angles': self.lattice_angles,
                'miller_indices': self.miller_indices,
                'point_group_symbol': self.point_group_symbol,
            },
            'dataset_args': self.dataset_args,
            'root_dir': self.root_dir,
            'n_batches': len(batches),
            'worker_id': str(os.getpid()),
        }

        if self.n_workers > 1:
            logger.info(f'Rendering crystals in parallel, worker pool size: {self.n_workers}')

            # Ensure that CUDA will work in subprocesses
            mp.set_start_method('spawn', force=True)

            # Create a temporary output directory
            output_dir = self.root_dir / 'tmp_output' / f'worker_{os.getpid()}'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create process arguments
            shared_args = {**shared_args, 'output_dir': output_dir}
            args = []
            for i, param_batch in enumerate(batches):
                args.append({'param_batch': param_batch, 'batch_idx': i, **shared_args})

            # Render in parallel
            with Pool(processes=self.n_workers) as pool:
                pool.map(_render_batch_wrapper, args)

            # Remove the temporary output directory
            output_dir.rmdir()

        else:
            # Loop over batches
            for i, param_batch in enumerate(batches):
                _render_batch(param_batch=param_batch, batch_idx=i, **shared_args)

        # If there are other active workers then just return here
        if self.is_active():
            logger.info('There are still active workers processing this dataset. Leaving the final collation to them.')
            return
        self.collate_results()

    def collate_results(self):
        """
        Combine the rendering parameters and segmentations.
        """
        try:
            self.comlog_lock.acquire(timeout=0)
        except Timeout:
            logger.warning(f'Could not acquire comlog lock. Skipping collation.')
            return
        combine_json_files(self.rendering_params_path, self.rendering_params_dir)
        combine_json_files(self.segmentations_path, self.segmentations_dir)
        combine_json_files(self.vertices_path, self.vertices_dir)
        self.comlog_lock.release()
        self._load_parameters()

    def render_from_parameters(self, params: dict, return_scene: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, Scene]]:
        """
        Render a single crystal image from parameters.
        """
        # Load light specular transmittance texture
        light_st_texture = None
        if 'light_st_texture' in params and isinstance(params['light_st_texture'], dict):
            light_st_texture = NoiseTexture(**params['light_st_texture'])

        # Load cell bumpmap
        cell_bumpmap = None
        if 'cell_bumpmap' in params and isinstance(params['cell_bumpmap'], dict):
            cell_bumpmap = NormalMapNoiseTexture(**params['cell_bumpmap'])

        # Load crystal bumpmap
        crystal_bumpmap = None
        if 'use_bumpmap' in params['crystal'] and params['crystal']['use_bumpmap']:
            assert 'bumpmap' in params['crystal'], 'No bumpmap found for crystal!'
            tex = params['crystal']['bumpmap']
            if isinstance(tex, Path):
                assert tex.exists(), f'Crystal bumpmap texture does not exist! ({tex})'
                crystal_bumpmap = torch.from_numpy(np.load(tex)['data']).to(device)
            elif isinstance(tex, np.ndarray):
                crystal_bumpmap = torch.from_numpy(tex).to(device)
            elif isinstance(tex, torch.Tensor):
                crystal_bumpmap = tex.clone().detach().to(device)

        # Create the crystal
        crystal = self._init_crystal(
            distances=params['crystal']['distances'],
            scale=params['crystal']['scale'],
            origin=params['crystal']['origin'],
            rotation=params['crystal']['rotation'],
            material_roughness=params['crystal']['material_roughness'],
            material_ior=params['crystal']['material_ior'],
            bumpmap_dim=crystal_bumpmap.shape[0] if crystal_bumpmap is not None else -1
        )
        if crystal_bumpmap is not None:
            crystal.bumpmap.data = crystal_bumpmap
        else:
            crystal.bumpmap.data.zero_()

        # Create the bubbles
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
            light_st_texture=light_st_texture,
            cell_bumpmap=cell_bumpmap,
            **self.dataset_args.to_dict(),
        )
        img = scene.render(seed=params['seed'] if 'seed' in params else get_seed())
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # todo: do we need this?

        if return_scene:
            return img, scene
        return img

    def annotate_image(self, image_idx: int = 0):
        """
        Annotate the first image with the projected vertices and save to disk
        """
        imgs = list(self.images_dir.glob('*.png'))
        imgs = sorted(imgs)
        img0_path = imgs[image_idx]
        img0 = np.array(Image.open(img0_path))
        seg = np.array(self.segmentations[image_idx])

        # Plot the image with segmentation overlay
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img0)
        ax.scatter(seg[:, 0], seg[:, 1], marker='x', c='r', s=50)
        fig.tight_layout()
        plt.savefig(self.images_dir.parent / f'segmentation_example_{img0_path.stem}.png')
        plt.close(fig)

    def is_active(self) -> bool:
        """
        Check if there are any workers still actively processing this dataset.
        """
        if not self.comlog_path.exists():
            return False
        try:
            self.comlog_lock.acquire(timeout=0)
        except Timeout:
            return True
        with open(self.comlog_path, 'r') as f:
            comlog = json.load(f)
        self.comlog_lock.release()
        if len(comlog['workers']) > 0:
            return True
        return False
