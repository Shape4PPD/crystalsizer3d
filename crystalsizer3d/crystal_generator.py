from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.random import default_rng
from trimesh import Trimesh

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


class CrystalGenerator:
    def __init__(
            self,
            crystal_id: str,
            miller_indices: List[Tuple[int, int, int]],
            ratio_means: List[float],
            ratio_stds: List[float],
            zingg_bbox: List[float],
            constraints: Optional[str] = None,
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
        self.ratio_means = {k: v for k, v in zip(self.miller_indices, ratio_means)}
        self.ratio_stds = {k: v for k, v in zip(self.miller_indices, ratio_stds)}

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
        crystals = []
        i = 0
        n_fails = 0
        while len(crystals) < num and n_fails < max_attempts:
            try:
                r, z, m = self.generate_crystal()
                n_fails = 0
            except AssertionError:
                n_fails += 1
                if n_fails >= max_attempts:
                    raise RuntimeError(f'Failed to generate a valid crystal after {n_fails} attempts.')
                continue
            m.metadata['name'] = f'{self.crystal_id}_{i:06d}'
            crystals.append((r, z, m))
            i += 1

        return crystals

    def generate_crystal(
            self,
            rel_distances: Optional[np.ndarray] = None,
            validate: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Trimesh]:
        """
        Generate a crystal from the provided rates or randomised.
        """
        if rel_distances is None:
            rel_distances = self.get_random_growth_rates()

        # Check constraints
        if validate and self.constraints is not None:
            vals = []
            for constraint in self.constraints:
                if constraint == 0:
                    vals.append(0)
                else:
                    vals.append(rel_distances[self.miller_indices.index(constraint)])
            vals = np.array(vals)
            assert np.all(vals[:-1] > vals[1:]), f'Constraints not satisfied: {vals} (should be decreasing)'

        # Build mesh
        v, f = self.crystal.build_mesh(distances=torch.from_numpy(rel_distances).to(torch.float32))
        v, f = to_numpy(v), to_numpy(f)
        mesh = Trimesh(vertices=v, faces=f, process=True, validate=True)
        mesh.fix_normals()
        assert mesh.is_watertight, 'Mesh is not watertight!'

        # Calculate the position on the Zingg diagram
        bbox_lengths = sorted(mesh.bounding_box_oriented.primitive.extents)
        assert bbox_lengths[2] > bbox_lengths[1] > bbox_lengths[0], f'Bounding box is invalid: {bbox_lengths}'
        si = bbox_lengths[0] / bbox_lengths[1]  # Small/Intermediate
        il = bbox_lengths[1] / bbox_lengths[2]  # Intermediate/Large
        if validate:
            assert (self.zingg_bbox[0] <= si <= self.zingg_bbox[1] and self.zingg_bbox[2] <= il <= self.zingg_bbox[3]), \
                f'Crystal shape outside of target Zingg bounding box: {si}, {il} ({self.zingg_bbox})'

        # Centre the mesh at the origin
        mesh.vertices -= mesh.center_mass

        return rel_distances, np.array([si, il]), mesh

    def get_random_growth_rates(self) -> np.ndarray:
        """
        Get randomised face growth rates.
        """
        rel_rates = self.rng.normal(
            loc=list(self.ratio_means.values()),
            scale=list(self.ratio_stds.values())
        )

        # Normalise the rates by the maximum
        rel_rates = np.abs(rel_rates) / np.abs(rel_rates).max()

        return rel_rates


if __name__ == '__main__':
    print('Generating...')
    generator = CrystalGenerator(
        crystal_id='LGLUAC02',
        ratio_means=[1, 0.21, 0.2, 0.2, 0.2, 0.2],
        ratio_stds=[0, 0, 0, 0, 0, 0],
        zingg_bbox=[0.01, 1.0, 0.01, 1.0],
    )
    rr_, z_, mesh_ = generator.generate_crystal()
    mesh_.show()
