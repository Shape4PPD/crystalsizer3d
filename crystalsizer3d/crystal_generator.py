from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.random import default_rng
from trimesh import Trimesh

from crystalsizer3d import logger
from crystalsizer3d.args.dataset_synthetic_args import CRYSTAL_IDS
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.util.utils import SEED, to_numpy


def normalise_distances(distances: Dict[Tuple[int, ...], float]) -> Dict[Tuple[int, ...], float]:
    """
    Normalise the distances by the maximum.
    """
    dv = np.array(list(distances.values()))
    dv /= dv.max()
    d2 = {k: dv[i] for i, k in enumerate(distances.keys())}
    return d2


def _generate_crystal(
        crystal_id: str,
        miller_indices: List[Tuple[int, int, int]],
        ratio_means: List[float],
        ratio_stds: List[float],
        rng: Optional[np.random.Generator] = None,
        seed: int = 0,
        constraints: Optional[List[Tuple[int, ...]]] = None,
        zingg_bbox: Optional[List[float]] = None,
        idx: int = 0,
        max_attempts: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, Trimesh]:
    """
    Generate a crystal with randomised face distances.
    """
    if rng is None:
        rng = default_rng(seed)
    csd = CSDProxy()
    cs = csd.load(crystal_id)

    n_attempts = 0
    while n_attempts < max_attempts:
        n_attempts += 1

        # Generate randomised face distances
        rel_distances = rng.normal(loc=ratio_means, scale=ratio_stds)
        rel_distances = np.abs(rel_distances) / np.abs(rel_distances).max()

        # Check constraints
        if constraints is not None:
            vals = []
            for constraint in constraints:
                if constraint == 0:
                    vals.append(0)
                else:
                    vals.append(rel_distances[miller_indices.index(constraint)])
            vals = np.array(vals)
            if not np.all(vals[:-1] > vals[1:]):
                continue

        # Build the crystal
        crystal = Crystal(
            lattice_unit_cell=cs.lattice_unit_cell,
            lattice_angles=cs.lattice_angles,
            miller_indices=miller_indices,
            point_group_symbol=cs.point_group_symbol,
            distances=torch.from_numpy(rel_distances).to(torch.float32)
        )
        v, f = to_numpy(crystal.mesh_vertices), to_numpy(crystal.mesh_faces)
        mesh = Trimesh(vertices=v, faces=f, process=True, validate=True)
        mesh.fix_normals()
        if not mesh.is_watertight:
            continue

        # Calculate the position on the Zingg diagram
        bbox_lengths = sorted(mesh.bounding_box_oriented.primitive.extents)
        assert bbox_lengths[2] > bbox_lengths[1] > bbox_lengths[0], f'Bounding box is invalid: {bbox_lengths}'
        si = bbox_lengths[0] / bbox_lengths[1]  # Small/Intermediate
        il = bbox_lengths[1] / bbox_lengths[2]  # Intermediate/Large
        if (zingg_bbox is not None
                and not (zingg_bbox[0] <= si <= zingg_bbox[1] and zingg_bbox[2] <= il <= zingg_bbox[3])):
            continue

        # Centre the mesh at the origin
        mesh.vertices -= mesh.center_mass
        break

    if n_attempts >= max_attempts:
        raise RuntimeError(f'Failed to generate a valid crystal after {n_attempts} attempts.')
    mesh.metadata['name'] = f'{crystal_id}_{idx:06d}'

    return rel_distances, np.array([si, il]), mesh


def _generate_crystal_wrapper(args):
    return _generate_crystal(**args)


class CrystalGenerator:
    def __init__(
            self,
            crystal_id: str,
            miller_indices: List[Tuple[int, int, int]],
            ratio_means: List[float],
            ratio_stds: List[float],
            zingg_bbox: List[float],
            constraints: Optional[str] = None,
            n_workers: int = 1
    ):
        """
        Initialize the crystal generator.
        """
        assert crystal_id in CRYSTAL_IDS, f'Crystal ID must be one of {CRYSTAL_IDS}. {crystal_id} received.'
        self.crystal_id = crystal_id
        self.miller_indices = miller_indices
        self._init_crystal()

        # Validate that we have the right number of ratio means and standard deviations
        assert len(self.miller_indices) == len(ratio_means) == len(ratio_stds), \
            'The number of miller indices, ratio means and standard deviations must be the same.'
        self.ratio_means = ratio_means
        self.ratio_stds = ratio_stds

        # Validate the Zingg bbox
        assert len(zingg_bbox) == 4, f'Zingg bounding box must have 4 values. {len(zingg_bbox)} received.'
        for i, v in enumerate(zingg_bbox):
            assert 0 <= v <= 1, f'Zingg bounding box values must be between 0 and 1. {v} received.'
            if i % 2 == 0:
                assert v < zingg_bbox[i + 1], \
                    f'Zingg bounding box values must be in (min,max,min,max) form. {zingg_bbox} received.'
        self.zingg_bbox = zingg_bbox

        # Parse the constraint string
        if constraints == '':
            constraints = None
        if constraints is not None:
            constraints_parts = constraints.split('>')
            assert len(constraints_parts) > 1, f'Invalid constraint string: {constraints}.'
            constraints = []
            for i, k in enumerate(constraints_parts):
                if len(k) == 3:
                    hkl = tuple(int(idx) for idx in k)
                    assert hkl in self.miller_indices, f'Invalid constraint key: {hkl}'
                    constraints.append(hkl)
                else:
                    assert i == len(constraints_parts) - 1 and k == '0', \
                        f'Only a 0 is allowed at the end of the constraint string. {k} received.'
                    constraints.append(0)
        self.constraints = constraints

        # Number of workers
        self.n_workers = n_workers

        # Initialise the random number generator
        self.rng = default_rng(SEED)

    def _init_crystal(self):
        """
        Initialise a crystal.
        """
        # Load the crystal template from the CSD database
        csd = CSDProxy()
        cs = csd.load(self.crystal_id)
        self.lattice_unit_cell = cs.lattice_unit_cell
        self.lattice_angles = cs.lattice_angles
        self.point_group_symbol = cs.point_group_symbol

        # Instantiate the crystal
        self.crystal = Crystal(
            lattice_unit_cell=self.lattice_unit_cell,
            lattice_angles=self.lattice_angles,
            miller_indices=self.miller_indices,
            point_group_symbol=self.point_group_symbol
        )

    def generate_crystals(
            self,
            num: int = 1,
            max_attempts: int = 10000
    ) -> List[Tuple[np.ndarray, np.ndarray, Trimesh]]:
        """
        Generate a list of randomised crystals.
        """
        shared_args = dict(
            crystal_id=self.crystal_id,
            miller_indices=self.miller_indices,
            ratio_means=self.ratio_means,
            ratio_stds=self.ratio_stds,
            constraints=self.constraints,
            zingg_bbox=self.zingg_bbox,
            max_attempts=max_attempts
        )

        if self.n_workers > 1:
            logger.info(f'Generating crystals in parallel, worker pool size: {self.n_workers}')
            args = []
            for i in range(num):
                args.append({'idx': i, 'seed': SEED + i, **shared_args})
            with Pool(processes=self.n_workers) as pool:
                crystals = pool.map(_generate_crystal_wrapper, args)
        else:
            crystals = []
            for i in range(num):
                r, z, m = _generate_crystal(idx=i, rng=self.rng, **shared_args)
                crystals.append((r, z, m))

        return crystals

    def generate_crystal(
            self,
            distances: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, Trimesh]:
        """
        Generate a randomised crystal.
        """
        crystal = Crystal(
            lattice_unit_cell=self.lattice_unit_cell,
            lattice_angles=self.lattice_angles,
            miller_indices=self.miller_indices,
            point_group_symbol=self.point_group_symbol,
            distances=torch.from_numpy(distances).to(torch.float32)
        )
        v, f = to_numpy(crystal.mesh_vertices), to_numpy(crystal.mesh_faces)
        mesh = Trimesh(vertices=v, faces=f, process=True, validate=True)
        mesh.fix_normals()

        # Calculate the position on the Zingg diagram
        bbox_lengths = sorted(mesh.bounding_box_oriented.primitive.extents)
        si = bbox_lengths[0] / bbox_lengths[1]  # Small/Intermediate
        il = bbox_lengths[1] / bbox_lengths[2]  # Intermediate/Large

        # Centre the mesh at the origin
        mesh.vertices -= mesh.center_mass

        return distances, np.array([si, il]), mesh


if __name__ == '__main__':
    print('Generating...')
    rr_, z_, mesh_ = _generate_crystal(
        crystal_id='LGLUAC11',
        miller_indices=[(1, 0, 1), (0, 2, 1), (0, 1, 0)],
        ratio_means=[1, 1, 1],
        ratio_stds=[0, 0, 0],
        zingg_bbox=[0.01, 1.0, 0.01, 1.0],
    )
    mesh_.show()
