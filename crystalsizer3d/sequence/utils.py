import glob
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from crystalsizer3d.args.sequence_fitter_args import SequenceFitterArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.util.utils import init_tensor, to_numpy

measurements_cache = {}


def get_image_paths(args: Namespace | SequenceFitterArgs, load_all: bool = False) -> List[Tuple[int, Path]]:
    """
    Load the images defined in the args.
    """
    pathspec = str(args.images_dir.absolute()) + '/*.' + args.image_ext
    all_image_paths = sorted(glob.glob(pathspec))
    image_paths = [(idx, Path(all_image_paths[idx])) for idx in range(
        args.start_image,
        len(all_image_paths) if args.end_image == -1 else args.end_image,
        args.every_n_images if hasattr(args, 'every_n_images') and args.every_n_images > 1 and not load_all else 1
    )]
    return image_paths


def load_manual_measurements(
        measurements_dir: Path,
        manager: Manager
) -> Dict[str, np.ndarray | list]:
    """
    Load manual measurements from the measurements directory.
    """
    from crystalsizer3d.sequence.sequence_fitter import PARAMETER_KEYS
    assert measurements_dir.exists(), f'Measurements directory {measurements_dir} does not exist.'
    if measurements_dir in measurements_cache:
        return measurements_cache[measurements_dir]
    mi_ref = manager.crystal.all_miller_indices

    # Expect measurements dir to contain files like XXXX.json
    keys = ['idx', ] + [k for k in PARAMETER_KEYS if k != 'light_radiance']
    measurements = {k: [] for k in keys}
    for file_path in measurements_dir.iterdir():
        if file_path.suffix != '.json':
            continue
        if file_path.name == 'volumes.json':
            continue
        crystal = Crystal.from_json(file_path)
        idx = int(file_path.stem)
        for k in keys:
            if k == 'idx':
                measurements[k].append(idx)
            elif k == 'distances':
                mi = crystal.all_miller_indices
                mi_idxs = ((mi[None, ...] == mi_ref[:, None]).all(dim=2)).nonzero(as_tuple=True)[1]
                measurements[k].append(to_numpy(crystal.distances[mi_idxs]))
            elif k == 'origin':
                # Adjust the origin so that the crystal's smallest z-coordinate is at z=0
                z_offset = crystal.vertices.amin(dim=0)[2]
                measurements[k].append(to_numpy(crystal.origin - torch.tensor([0, 0, z_offset])))
            else:
                measurements[k].append(to_numpy(getattr(crystal, k)))
    for k in keys:
        measurements[k] = np.array(measurements[k])

    # Fix the origin for the automatic measurements to match the manual measurements
    ds = manager.ds
    cs = CSDProxy().load(ds.dataset_args.crystal_id)
    crystal = Crystal(
        lattice_unit_cell=cs.lattice_unit_cell,
        lattice_angles=cs.lattice_angles,
        miller_indices=ds.miller_indices,
        point_group_symbol=cs.point_group_symbol,
    )

    def adjust_distances(distances_old, origin_old):
        distances_new = np.zeros_like(distances_old)
        for i in range(len(origin_old)):
            crystal.distances.data = init_tensor(distances_old[i])
            crystal.origin.data = init_tensor(origin_old[i])
            crystal.adjust_origin(origin0, verify=False)
            distances_new[i] = to_numpy(crystal.distances)
        return distances_new

    # Use the origin from the first measured frame for all other frames
    origin0 = init_tensor(measurements['origin'][0])

    # Ensure the measurements all have the same origin
    if len(np.unique(measurements['origin'], axis=0)) != 1:
        measurements['distances'] = adjust_distances(measurements['distances'], measurements['origin'])
        measurements['origin'] = np.repeat(origin0[None, ...], len(measurements['origin']), axis=0)

    # Cache result for future use
    measurements_cache[measurements_dir] = measurements

    return measurements
