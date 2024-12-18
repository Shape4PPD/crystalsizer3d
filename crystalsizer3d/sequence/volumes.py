import json
from pathlib import Path

import numpy as np
import torch
import yaml
from trimesh import Trimesh

from crystalsizer3d import logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.args.sequence_fitter_args import SequenceFitterArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.sequence.utils import get_image_paths, load_manual_measurements
from crystalsizer3d.util.image_scale import get_pixels_to_mm_scale_factor
from crystalsizer3d.util.utils import init_tensor, json_to_numpy, to_numpy


def _calculate_volumes(
        sf_path: Path,
        pixel_to_mm: float | None = None
) -> np.ndarray:
    """
    Calculate the crystal volumes for the fitted sequence.
    """
    # Load the args
    with open(sf_path / 'args_sequence_fitter.yml', 'r') as f:
        sf_args = SequenceFitterArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    with open(sf_path / 'args_refiner.yml', 'r') as f:
        ref_args = RefinerArgs.from_args(yaml.load(f, Loader=yaml.FullLoader))
    image_paths = get_image_paths(sf_args, load_all=True)
    dist_to_mm = get_pixels_to_mm_scale_factor(sf_args.initial_scene, image_paths[0][1], pixel_to_mm)

    # Instantiate a manager
    manager = Manager.load(
        model_path=ref_args.predictor_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': False,
                'batch_size': 1
            },
        },
        save_dir=Path('/tmp')
    )

    # Get a crystal object
    ds = manager.ds
    cs = CSDProxy().load(ds.dataset_args.crystal_id)
    crystal = Crystal(
        lattice_unit_cell=cs.lattice_unit_cell,
        lattice_angles=cs.lattice_angles,
        miller_indices=ds.miller_indices,
        point_group_symbol=cs.point_group_symbol,
        dtype=torch.float64
    )

    # Load the parameters
    param_path = sf_path / 'parameters.json'
    logger.info(f'Loading parameters from {param_path}.')
    with open(param_path, 'r') as f:
        parameters = json_to_numpy(json.load(f))
    distances = parameters['eval']['distances']
    scales = parameters['eval']['scale'] * dist_to_mm
    assert len(distances) == len(scales) == len(image_paths), \
        f'Length mismatch: {len(distances)}, {len(scales)}, {len(image_paths)}'

    # Calculate the volumes
    logger.info('Calculating volumes.')
    volumes = np.zeros(len(image_paths))
    for i, (d, s) in enumerate(zip(distances, scales)):
        v, f = crystal.build_mesh(
            scale=init_tensor(s, dtype=torch.float64),
            distances=init_tensor(d, dtype=torch.float64)
        )
        v, f = to_numpy(v), to_numpy(f)
        m = Trimesh(vertices=v, faces=f)
        volumes[i] = m.volume

    return volumes


def generate_or_load_volumes(
        sf_path: Path,
        pixel_to_mm: float | None = None,
        cache_only: bool = False,
        regenerate: bool = False
) -> np.ndarray:
    """
    Generate or load the crystal volumes for the fitted sequence.
    """
    assert sf_path.exists(), f'Sequence fitter path {sf_path} does not exist.'
    vols_path = sf_path / 'volumes.json'

    vols = None
    if vols_path.exists():
        if regenerate:
            vols_path.unlink()
        else:
            try:
                with open(vols_path, 'r') as f:
                    vols = json_to_numpy(json.load(f))
            except Exception as e:
                logger.warning(f'Could not load volume data: {e}')

    if vols is None:
        if cache_only:
            raise RuntimeError(f'Cache could not be loaded!')
        logger.info('Processing crystal sequence.')
        vols = _calculate_volumes(sf_path, pixel_to_mm)
        with open(vols_path, 'w') as f:
            json.dump(vols.tolist(), f)

    return vols


def _calculate_manual_measurement_volumes(
        sf_args: SequenceFitterArgs,
        measurements_dir: Path,
        predictor_model_path: Path,
        pixel_to_mm: float | None = None
) -> np.ndarray:
    """
    Calculate the crystal volumes for the manual measurements.
    Predictor model path needed to determine the crystal id.
    """
    image_paths = get_image_paths(sf_args, load_all=True)
    dist_to_mm = get_pixels_to_mm_scale_factor(sf_args.initial_scene, image_paths[0][1], pixel_to_mm)

    # Instantiate a manager
    manager = Manager.load(
        model_path=predictor_model_path,
        args_changes={
            'runtime_args': {
                'use_gpu': False,
                'batch_size': 1
            },
        },
        save_dir=Path('/tmp')
    )

    # Get a crystal object
    ds = manager.ds
    cs = CSDProxy().load(ds.dataset_args.crystal_id)
    crystal = Crystal(
        lattice_unit_cell=cs.lattice_unit_cell,
        lattice_angles=cs.lattice_angles,
        miller_indices=ds.miller_indices,
        point_group_symbol=cs.point_group_symbol,
        dtype=torch.float64
    )

    # Load the measurements
    logger.info(f'Loading manual measurements from {measurements_dir}')
    measurements = load_manual_measurements(
        measurements_dir=measurements_dir,
        manager=manager
    )
    distances = measurements['distances']
    scales = measurements['scale'] * dist_to_mm
    assert len(distances) == len(scales), f'Length mismatch: {len(distances)}, {len(scales)}'

    # Calculate the volumes
    logger.info('Calculating volumes for the manual measurements.')
    volumes = np.zeros(len(distances))
    for i, (d, s) in enumerate(zip(distances, scales)):
        v, f = crystal.build_mesh(
            scale=init_tensor(s, dtype=torch.float64),
            distances=init_tensor(d, dtype=torch.float64)
        )
        v, f = to_numpy(v), to_numpy(f)
        m = Trimesh(vertices=v, faces=f)
        volumes[i] = m.volume

    return volumes


def generate_or_load_manual_measurement_volumes(
        sf_args: SequenceFitterArgs,
        measurements_dir: Path,
        predictor_model_path: Path,
        pixel_to_mm: float | None = None,
        cache_only: bool = False,
        regenerate: bool = False
) -> np.ndarray:
    """
    Generate or load the crystal volumes for the manual measurements.
    """
    assert measurements_dir.exists(), f'Manual measurements directory {measurements_dir} does not exist.'
    vols_path = measurements_dir / 'volumes.json'

    vols = None
    if vols_path.exists():
        if regenerate:
            vols_path.unlink()
        else:
            try:
                with open(vols_path, 'r') as f:
                    vols = json_to_numpy(json.load(f))
            except Exception as e:
                logger.warning(f'Could not load manual measurements volume data: {e}')

    if vols is None:
        if cache_only:
            raise RuntimeError(f'Cache could not be loaded!')
        logger.info('Processing manual measurements sequence.')
        vols = _calculate_manual_measurement_volumes(sf_args, measurements_dir, predictor_model_path, pixel_to_mm)
        with open(vols_path, 'w') as f:
            json.dump(vols.tolist(), f)

    return vols
