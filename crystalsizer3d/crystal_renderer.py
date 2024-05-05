import csv
import json
import multiprocessing as mp
import os
import shutil
from multiprocessing import Lock, Manager, Pool
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
from PIL import Image
from ccdc.io import EntryReader

from crystalsizer3d import N_WORKERS, USE_CUDA, logger
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.scene_components.bubble import Bubble, make_bubbles
from crystalsizer3d.scene_components.bumpmap import generate_bumpmap
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import RenderError
from crystalsizer3d.util.utils import SEED, append_json, to_numpy

# Ensure that CUDA will work in subprocesses
mp.set_start_method('spawn', force=True)

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('llvm_ad_rgb')
    device = torch.device('cpu')


def _initialise_crystal(
        params: dict,
        dataset_args: DatasetSyntheticArgs,
        scale_init: float = 1
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
        crystal.bumpmap.data = generate_bumpmap(
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
        lock: Optional[Lock] = None,
):
    """
    Render a batch of crystals to images.
    """
    da = dataset_args
    seed = SEED + batch_idx

    # Sort directories
    assert root_dir.exists(), f'Root dir does not exist! ({root_dir})'
    images_dir = root_dir / 'images'
    bumpmaps_dir = root_dir / 'bumpmaps'
    clean_images_dir = root_dir / 'images_clean'
    images_dir.mkdir(exist_ok=True)
    if da.crystal_bumpmap_dim > 0:
        bumpmaps_dir.mkdir(exist_ok=True)
    if da.generate_clean:
        clean_images_dir.mkdir(exist_ok=True)
    if output_dir is None:
        output_dir = root_dir / 'tmp_output'
    else:
        output_dir = output_dir / f'tmp_{batch_idx:010d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render
    logger.info(f'Batch {batch_idx + 1}/{n_batches}: Rendering {len(param_batch)} crystals to {output_dir}.')
    rendering_params = {}
    segmentations = {}
    scale_init = 1
    for idx, params in param_batch.items():
        try:
            # Initialise the crystal
            crystal_params_i = crystal_params.copy()
            crystal_params_i['distances'] = list(params['distances'].values())
            crystal = _initialise_crystal(crystal_params_i, da, scale_init)

            # Create the bubbles
            if da.max_bubbles > 0:
                bubbles = make_bubbles(
                    n_bubbles=np.random.randint(da.min_bubbles, da.max_bubbles + 1),
                    min_roughness=da.bubbles_min_roughness,
                    max_roughness=da.bubbles_max_roughness,
                    min_ior=da.bubbles_min_ior,
                    max_ior=da.bubbles_max_ior,
                    device=device,
                )
            else:
                bubbles = []

            # Sample the light radiance
            light_radiance = np.random.uniform(da.light_radiance_min, da.light_radiance_max)

            # Create and render the scene
            scene = Scene(
                crystal=crystal,
                bubbles=bubbles,
                res=da.image_size,
                light_radiance=light_radiance,
                **da.to_dict(),
            )
            scene.place_crystal(
                min_area=da.min_area,
                max_area=da.max_area,
                centre_crystal=da.centre_crystals,
                min_x=da.crystal_min_x,
                max_x=da.crystal_max_x,
                min_y=da.crystal_min_y,
                max_y=da.crystal_max_y,
            )
            scene.place_bubbles(
                min_x=da.bubbles_min_x,
                max_x=da.bubbles_max_x,
                min_y=da.bubbles_min_y,
                max_y=da.bubbles_max_y,
                min_scale=da.bubbles_min_scale,
                max_scale=da.bubbles_max_scale,
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

            # Save the defect bumpmap
            if da.crystal_bumpmap_dim > -1:
                bumpmap_path = output_dir / f'{params["image"][:-4]}.npz'
                np.savez_compressed(bumpmap_path, data=to_numpy(crystal.bumpmap))

            # Save a clean version of the image without bubbles and no bumpmap if required
            if da.generate_clean:
                scene.clear_bubbles_and_bumpmap()
                img_clean = scene.render(seed=seed + idx)
                img_clean = cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_dir / params['image']) + '_clean', img_clean)

        except RenderError as e:
            logger.warning(f'Rendering failed! {e}')
            logger.info(f'Adding idx={params["image"]} details to {str(output_dir.parent / "errored.json")}')
            append_json(root_dir / 'errored.json', {params['image']: str(e)}, lock)
            raise e

    logger.info(f'Batch {batch_idx + 1}/{n_batches}: Collating results.')

    # Move images into the images directory
    image_files = list(output_dir.glob('*.png'))
    for img in image_files:
        img.rename(images_dir / img.name)

    # Move bumpmaps into the bumpmaps directory
    bumpmap_files = list(output_dir.glob('*.npz'))
    for bumpmap in bumpmap_files:
        bumpmap.rename(bumpmaps_dir / bumpmap.name)

    # Move the clean images into the clean images directory
    clean_image_files = list(output_dir.glob('*_clean'))
    for img in clean_image_files:
        img.rename(clean_images_dir / (img.stem + '.png'))

    # Write the combined segmentations and parameters to json files
    append_json(root_dir / 'segmentations.json', segmentations, lock)
    append_json(root_dir / 'rendering_parameters.json', rendering_params, lock)

    # Clean up
    shutil.rmtree(output_dir)


def _render_batch_wrapper(args):
    return _render_batch(**args)


class CrystalRenderer:
    def __init__(
            self,
            param_path: Path,
            dataset_args: DatasetSyntheticArgs,
            quiet_render: bool = False
    ):
        self.dataset_args = dataset_args
        self.quiet_render = quiet_render
        if N_WORKERS > 0:
            self.n_workers = N_WORKERS
        else:
            self.n_workers = len(os.sched_getaffinity(0))
        self.param_path = param_path
        self.root_dir = self.param_path.parent
        self.images_dir = self.root_dir / 'images'
        self.rendering_params_path = self.root_dir / 'rendering_parameters.json'
        self._init_crystal_settings()
        self._load_parameters()

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
            headers_distances = {tuple(int(hkl) for hkl in h.split('_')[1]): h
                                 for h in reader.fieldnames if h[0] == 'd'}
            assert all(hkl in headers_distances for hkl in self.miller_indices), 'Missing distance headers!'
            self.data = {}
            for i, row in enumerate(reader):
                idx = int(row['idx'])
                assert i == idx, f'Missing row {i}!'
                self.data[idx] = {
                    'image': row['image'],
                    'distances': {hkl: float(row[headers_distances[hkl]]) for hkl in self.miller_indices},
                    'rendered': row['image'] in self.rendering_params,
                }

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
        batches = []
        for i in range(0, len(idxs), bs):
            batches.append({idx: self.data[idx] for idx in idxs[i:i + bs]})
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
        }

        if self.n_workers > 1:
            logger.info(f'Rendering crystals in parallel, worker pool size: {self.n_workers}')

            # Create a temporary output directory
            output_dir = self.root_dir / 'tmp_output'
            output_dir.mkdir(exist_ok=True)

            # Create process arguments
            manager = Manager()
            lock = manager.Lock()
            shared_args = {**shared_args, 'output_dir': output_dir, 'lock': lock}
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
            bumpmap_dim=self.dataset_args.crystal_bumpmap_dim
        )

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

        # Add bumpmap to the crystal
        if crystal.use_bumpmap:
            assert 'bumpmap' in params, 'Bumpmap path not provided!'
            assert params['bumpmap'].exists(), f'Bumpmap file does not exist! ({params["bumpmap"]})'
            crystal.bumpmap.data = torch.from_numpy(np.load(params['bumpmap'])['data']).to(device)

        # Create and render the scene
        scene = Scene(
            crystal=crystal,
            bubbles=bubbles,
            res=self.dataset_args.image_size,
            light_radiance=params['light_radiance'],
            **self.dataset_args.to_dict(),
        )
        img = scene.render(seed=params['seed'])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # todo: do we need this?

        return img

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
        ax.imshow(img0)
        ax.scatter(seg[:, 0], seg[:, 1], marker='x', c='r', s=50)
        fig.tight_layout()
        plt.savefig(self.images_dir.parent / f'segmentation_example_{img0_path.stem}.png')
        plt.close(fig)
