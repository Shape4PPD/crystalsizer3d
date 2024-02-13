from typing import Dict, List, Optional, Tuple

import numpy as np
from ccdc.crystal import Crystal
from ccdc.io import EntryReader
from ccdc.morphology import VisualHabit, VisualHabitMorphology
from numpy.random import default_rng
from trimesh import Trimesh

from crystalsizer3d.args.dataset_synthetic_args import CRYSTAL_IDS
from crystalsizer3d.util.utils import SEED, normalise


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
        self._init_crystal()

        # Validate that we have the right number of ratio means and standard deviations
        assert len(ratio_means) == len(ratio_stds), \
            'The number of ratio means and standard deviations must be the same.'
        if len(ratio_means) < len(self.distances):
            ratio_means += [1] * (len(self.distances) - len(ratio_means))
            ratio_stds += [0] * (len(self.distances) - len(ratio_stds))
        elif len(ratio_means) > len(self.distances):
            ratio_means = ratio_means[:len(self.distances)]
            ratio_stds = ratio_stds[:len(self.distances)]
        self.ratio_means = {k: v for k, v in zip(self.distances.keys(), ratio_means)}
        self.ratio_stds = {k: v for k, v in zip(self.distances.keys(), ratio_stds)}

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
                    assert hkl in self.distances.keys(), f'Invalid constraint key: {hkl}'
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
        reader = EntryReader()
        self.crystal: Crystal = reader.crystal(self.crystal_id)

        # Initialise VisualHabit and calculate the surface energies
        vh_settings = VisualHabit.Settings()
        vh_settings.potential = 'dreidingII'
        self.results = VisualHabit(settings=vh_settings).calculate(self.crystal)
        self.og_morph = self.results.morphology

        # Get the face configurations
        faces = self.og_morph.facets
        face_list = [[face.miller_indices.hkl, face.perpendicular_distance] for face in faces]
        face_list = sorted(face_list, key=lambda x: x[1], reverse=True)
        self.face_list = face_list

        # Get the distances of the symmetric faces
        self.distances = self.get_distances(self.og_morph)

    def generate_crystals(self, num: int = 1, max_attempts: int = 100) -> List[Tuple[np.ndarray, np.ndarray, Trimesh]]:
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
            m.metadata['name'] = f'{self.crystal.identifier}_{i:06d}'
            crystals.append((r, z, m))
            i += 1

        return crystals

    def generate_crystal(
            self,
            rel_rates: Optional[np.ndarray] = None,
            validate: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Trimesh]:
        """
        Generate a crystal from the provided rates or randomised.
        """
        if rel_rates is None:
            rel_rates = self.get_random_growth_rates()
        growth_rates = self.get_expanded_growth_rates(rel_rates)
        morph = VisualHabitMorphology.from_growth_rates(self.crystal, growth_rates)

        # The morphological distances should be the same as the randomised growth rates but some faces may be missing
        distances = self.get_distances(morph, insert_missing=True)
        dv = np.array(list(distances.values()))
        assert dv.max() > 0, f'No non-zero distances in the resolved morphology.'
        rel_rates2 = np.array(list(normalise_distances(distances).values()))
        nz = rel_rates2 != 0
        rel_rates /= rel_rates[nz].max()
        assert np.allclose(rel_rates[nz], rel_rates2[nz]), f'Distances changed! {rel_rates[nz]} vs {rel_rates2[nz]}'

        # Calculate the position on the Zingg diagram
        bbox_lengths = morph.oriented_bounding_box.lengths
        assert bbox_lengths[0] > bbox_lengths[1] > bbox_lengths[2], f'Bounding box is invalid: {bbox_lengths}'
        si = bbox_lengths[2] / bbox_lengths[1]  # Small/Intermediate
        il = bbox_lengths[1] / bbox_lengths[0]  # Intermediate/Large
        if validate:
            assert (self.zingg_bbox[0] <= si <= self.zingg_bbox[1] and self.zingg_bbox[2] <= il <= self.zingg_bbox[3]), \
                f'Crystal shape outside of target Zingg bounding box: {si}, {il} ({self.zingg_bbox})'

        # Check constraints
        if validate and self.constraints is not None:
            hkls = list(self.distances.keys())
            vals = []
            for constraint in self.constraints:
                if constraint == 0:
                    vals.append(0)
                else:
                    vals.append(rel_rates2[hkls.index(constraint)])
            vals = np.array(vals)
            assert np.all(vals[:-1] > vals[1:]), f'Constraints not satisfied: {vals} (should be decreasing)'

        # Convert the morphology into a mesh
        com = np.array(morph.centre_of_geometry)
        vertices = []
        faces = []
        idx = -1
        for i, f in enumerate(morph.facets):
            cof = np.array(f.centre_of_geometry)
            vertices.append(cof)
            idx += 1
            cof_idx = idx
            for j, edge in enumerate(f.edges):
                v0 = np.array(edge[0])
                v1 = np.array(edge[1])
                if np.allclose(v0, v1):
                    continue
                normal = normalise(np.cross(v0 - cof, v1 - cof))
                if np.dot(normal, com - cof) < 0:
                    vertices.append(v0)
                    vertices.append(v1)
                else:
                    vertices.append(v1)
                    vertices.append(v0)
                faces.append([cof_idx, idx + 1, idx + 2])
                idx += 2
        vertices = np.stack(vertices)
        faces = np.stack(faces)

        # Create the mesh
        mesh = Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
        mesh.fix_normals()
        assert mesh.is_watertight, 'Mesh is not watertight!'

        # Centre the mesh at the origin
        mesh.vertices -= mesh.center_mass

        return rel_rates2, np.array([si, il]), mesh

    def get_random_growth_rates(self) -> np.ndarray:
        """
        Get randomised face growth rates.
        """
        rel_rates = self.rng.normal(
            loc=list(self.ratio_means.values()),
            scale=list(self.ratio_stds.values())
        )

        # Normalise the rates by the maximum
        rel_rates = np.abs(rel_rates) / rel_rates.max()

        return rel_rates

    def get_expanded_growth_rates(self, rel_rates: np.ndarray) -> List[Tuple[Crystal.MillerIndices, float]]:
        """
        Expand the growth rates to include the symmetries.
        """
        rel_rates_indexed = {k: v for k, v in zip(self.distances.keys(), rel_rates)}
        growth_rates = []
        for facet in self.og_morph.facets:
            idx = facet.miller_indices.hkl
            idx_canonical = tuple(np.abs(i) for i in idx)
            d = rel_rates_indexed[idx_canonical]
            if d == 0:
                continue
            row = (facet.miller_indices, d)
            growth_rates.append(row)

        return growth_rates

    def get_distances(self, morph: VisualHabitMorphology, insert_missing: bool = False) -> Dict[Tuple[int, ...], float]:
        """
        Get the perpendicular distances of the symmetric faces.
        """
        distances = {}
        for facet in morph.facets:
            idx = facet.miller_indices.hkl
            idx_canonical = tuple(np.abs(i) for i in idx)
            pd = np.abs(facet.perpendicular_distance)
            if idx_canonical not in distances.keys():
                distances[idx_canonical] = pd
            else:
                assert np.allclose(distances[idx_canonical], pd, atol=1e-6), \
                    (f'Face {idx} has different distance to the symmetric face {idx_canonical} '
                     f'({pd} vs {distances[idx_canonical]}).')

        # Rebuild the dictionary with the canonical indices if available
        if hasattr(self, 'distances'):
            distances_sorted = {}
            for k in self.distances.keys():
                if k in distances.keys():
                    distances_sorted[k] = distances[k]
                elif insert_missing:
                    distances_sorted[k] = 0
            if len(distances) > len(distances_sorted):
                raise ValueError(f'The morphology has distances that did not appear in the canonical morphology.')
        else:
            distances_sorted = distances

        return distances_sorted


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
