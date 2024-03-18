import csv
import json
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from trimesh import Trimesh
from trimesh.exchange.obj import load_obj

from crystalsizer3d import logger
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs
from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs, PREANGLES_MODE_AXISANGLE, \
    PREANGLES_MODE_QUATERNION, PREANGLES_MODE_SINCOS
from crystalsizer3d.args.renderer_args import RendererArgs
from crystalsizer3d.crystal_renderer import CrystalWellSettings
from crystalsizer3d.util.utils import axisangle_to_euler, euler_to_axisangle, euler_to_quaternion, from_preangles, \
    quaternion_to_euler, to_numpy

DATASET_TYPE_SYNTHETIC = 'synthetic'
DATASET_TYPE_IMAGES = 'images'
DATASET_TYPES = {
    DATASET_TYPE_SYNTHETIC: 'Synthetic crystals',
    DATASET_TYPE_IMAGES: 'Microscope images',
}


def plot_symmetry_group(mesh, symm_group):
    """
    Plot the symmetry group of a mesh for validation.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(symm_group), 3, figsize=(8, 2 * len(symm_group)))
    for i, q in enumerate(symm_group):
        R = Rotation.from_quat(q)
        T = np.eye(4)
        T[:3, :3] = R.as_matrix()
        if i == 0:
            assert np.allclose(T, np.eye(4))
        mesh_rotated = mesh.copy().apply_transform(T)
        for j, view in enumerate(['xy', 'xz', 'yz']):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(view)
            elif j == 1:
                qstr = 'q = [' + ', '.join([f'{qi:.1f}' for qi in q]) + ']'
                ax.set_title(qstr)
            idxs = 'xyz'.index(view[0]), 'xyz'.index(view[1])
            ax.scatter(*mesh.vertices[:, idxs].T, s=40, c='green', marker='o')
            ax.scatter(*mesh_rotated.vertices[:, idxs].T, s=80, c='red', marker='x')
    fig.tight_layout()
    plt.show()


class Dataset:
    labels_zingg = ['si', 'il']
    labels_transformation = ['x', 'y', 'z', 's']
    labels_transformation_sincos = ['rxs', 'rxc', 'rys', 'ryc', 'rzs', 'rzc']
    labels_transformation_quaternion = ['rw', 'rx', 'ry', 'rz']
    labels_transformation_axisangle = ['rax', 'ray', 'raz']
    labels_material = ['b', 'ior', 'r']
    labels_light = ['e']
    labels_light_location = ['lx', 'ly', 'lz']
    labels_light_sincos = ['lrxs', 'lrxc', 'lrys', 'lryc']
    labels_light_quaternion = ['lrw', 'lrx', 'lry', 'lrz']
    labels_light_axisangle = ['lrax', 'lray', 'lraz']

    def __init__(
            self,
            ds_args: DatasetTrainingArgs,
    ):
        self.ds_args = ds_args
        path = ds_args.dataset_path
        assert path.exists(), f'Dataset path does not exist: {path}'
        self.path = path

        # Load the data
        self._load_data()

        # Load dataset statistics
        self._load_ds_stats()

        # Split dataset into train and test
        self.train_test_split_target = self.ds_args.train_test_split
        self._split_dataset()

        # Set stats
        self.size_all = len(self.data)
        self.size_train = len(self.train_idxs)
        self.size_test = len(self.test_idxs)
        self.image_size = self.dataset_args.image_size

        # Set labels (and output vector size)
        labels = []
        if self.ds_args.train_zingg:
            labels += self.labels_zingg
        if self.ds_args.train_distances:
            labels += self.labels_distances
            if self.ds_args.use_distance_switches:
                labels += self.labels_distance_switches
        if self.ds_args.train_transformation:
            labels += self.labels_transformation
            if self.ds_args.preangles_mode == PREANGLES_MODE_SINCOS:
                labels += self.labels_transformation_sincos
            elif self.ds_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                labels += self.labels_transformation_quaternion
            else:
                assert self.ds_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                labels += self.labels_transformation_axisangle

        if self.ds_args.train_material:
            labels += self.labels_material
        if self.ds_args.train_light:
            labels += self.labels_light
            if not self.renderer_args.transmission_mode:
                labels += self.labels_light_location
                if self.ds_args.preangles_mode == PREANGLES_MODE_SINCOS:
                    labels += self.labels_light_sincos
                elif self.ds_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                    labels += self.labels_light_quaternion
                else:
                    assert self.ds_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                    labels += self.labels_light_axisangle

        # If any of the labels have 0 variance in the dataset then remove them
        fixed_parameters = {}
        for k in labels:
            if k[:2] == 'ds':
                continue
            if self.ds_stats[k]['var'] == 0:
                logger.warning(f'Removing key {k} from parameters as it has 0 variance.')
                fixed_parameters[k] = self.ds_stats[k]['mean']
                if k in self.labels_distances and self.ds_args.use_distance_switches:
                    k_switch = f'ds{self.labels_distances.index(k)}'
                    fixed_parameters[k_switch] = 1 if self.ds_stats[k]['mean'] > 0 else 0
        for k in fixed_parameters.keys():
            labels.remove(k)

        self.labels = labels
        self.label_size = len(labels)
        self.fixed_parameters = fixed_parameters
        self.labels_distances_active = [k for k in self.labels_distances if k in self.labels]
        self.labels_distance_switches_active = [k for k in self.labels_distance_switches if k in self.labels]
        self.labels_transformation_active = [k for k in self.labels_transformation + self.labels_transformation_sincos +
                                             self.labels_transformation_quaternion + self.labels_transformation_axisangle
                                             if k in self.labels]
        self.labels_material_active = [k for k in self.labels_material if k in self.labels]
        self.labels_light_active = [k for k in
                                    self.labels_light + self.labels_light_location + self.labels_light_sincos +
                                    self.labels_light_quaternion + self.labels_light_axisangle if k in self.labels]

        # Cache the rotation matrices and symmetry groups
        self.rotation_matrices = self._calculate_rotation_matrices()
        self.symmetry_groups = {}

    def _load_data(self):
        """
        Load the dataset from disk.
        """
        logger.info(f'Loading dataset from {self.path}')

        # Load arguments
        with open(self.path / 'options.yml', 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
            self.created = args['created']
            self.dataset_args = DatasetSyntheticArgs.from_args(args['dataset_args'])
            self.renderer_args = RendererArgs.from_args(args['renderer_args'])

        # Load vcw settings
        self.vcw_settings = CrystalWellSettings()
        self.vcw_settings.from_json(self.path / 'vcw_settings.json')

        # Load rendering parameters (but don't fail if they don't exist)
        r_params_file = self.path / 'rendering_parameters.json'
        if r_params_file.exists():
            with open(r_params_file, 'r') as f:
                rendering_parameters = json.load(f)
        else:
            logger.warning(f'Rendering parameters file {r_params_file} does not exist.')
            rendering_parameters = None

        # Load data
        with open(self.path / 'parameters.csv', 'r') as f:
            reader = csv.DictReader(f)
            self.headers = reader.fieldnames
            self.labels_distances = [h for h in self.headers if h[0] == 'd']
            self.labels_distance_switches = [f'ds{i}' for i in range(len(self.labels_distances))]

            self.data = {}
            for i, row in enumerate(reader):
                idx = int(row['idx'])
                assert i == idx, f'Missing row {i}!'
                item = {}
                for header in self.headers:
                    if header == 'idx':
                        item[header] = idx
                    elif header == 'crystal_id':
                        item[header] = row[header]
                    elif header == 'image':
                        image_path = self.path / 'images' / row['image']
                        if self.ds_args.check_image_paths:
                            assert image_path.exists(), f'Image path does not exist: {image_path}'
                        item['image'] = image_path
                    elif header[0] == 'd':
                        item[header] = float(row[header])
                    elif header in ['si', 'il']:
                        item[header] = float(row[header])

                # Include the rendering parameters
                if rendering_parameters is not None:
                    item['rendering_parameters'] = rendering_parameters[row['image']]

                self.data[idx] = item

    def _load_ds_stats(self):
        """
        Generate or load dataset statistics for normalisation.
        """
        path = self.path / 'ds_stats.yml'
        keys = (self.labels_zingg + self.labels_distances +
                self.labels_transformation + self.labels_transformation_sincos +
                self.labels_transformation_quaternion + self.labels_transformation_axisangle +
                self.labels_material + self.labels_light)
        if not self.renderer_args.transmission_mode:
            keys += self.labels_light_location + self.labels_light_sincos + self.labels_light_quaternion + self.labels_light_axisangle

        if path.exists():
            # Load normalisation parameters
            try:
                with open(path, 'r') as f:
                    stats = yaml.load(f, Loader=yaml.FullLoader)
                    for k in keys:
                        assert k in stats, f'Key {k} not in normalisation parameters.'
                    logger.info(f'Loaded dataset statistics from {path}.')
                    self.ds_stats = stats
                    return
            except Exception as e:
                logger.error(f'Could not load normalisation parameters from {path}: {e}.')

        # Generate normalisation parameters
        logger.info('Calculating dataset statistics.')
        stats = {k: {'min': np.inf, 'max': -np.inf, 'mean': 0, 'M2': 0, 'var': 0} for k in keys}
        # var_test = {k: [] for k in keys}  # to test the variance calculation

        # Loop over the dataset
        for i, (idx, item) in enumerate(self.data.items()):
            if (i + 1) % 1000 == 0:
                logger.info(f'Processed {i + 1} / {len(self.data)} items.')
            r_params = item['rendering_parameters']
            for k in keys:
                if k in self.labels_transformation:
                    if k in ['x', 'y', 'z']:
                        val = r_params['location']['xyz'.index(k)]
                    else:
                        assert k == 's'
                        val = r_params['scale']
                elif k in self.labels_transformation_sincos:
                    angle = r_params['rotation']['xyz'.index(k[1])]
                    val = np.sin(angle) if k[2] == 's' else np.cos(angle)
                elif k in self.labels_transformation_quaternion:
                    quat = euler_to_quaternion(r_params['rotation'])
                    val = quat['wxyz'.index(k[1])]
                elif k in self.labels_transformation_axisangle:
                    v = euler_to_axisangle(r_params['rotation'])
                    val = v['xyz'.index(k[2])]
                elif k in self.labels_material:
                    m_params = r_params['material']
                    if k == 'b':
                        val = m_params['brightness']
                    elif k == 'ior':
                        val = m_params['ior']
                    else:
                        assert k == 'r'
                        if 'roughness' in m_params:
                            val = m_params['roughness']
                        else:
                            val = 0
                elif k in self.labels_light:
                    assert k == 'e'
                    val = r_params['light']['energy']
                elif k in self.labels_light_location:
                    val = r_params['light']['location']['xyz'.index(k[1])]
                elif k in self.labels_light_sincos:
                    angle = r_params['light']['rotation']['xyz'.index(k[2])]
                    val = np.sin(angle) if k[3] == 's' else np.cos(angle)
                elif k in self.labels_light_quaternion:
                    quat = euler_to_quaternion(r_params['light']['rotation'])
                    val = quat['wxyz'.index(k[2])]
                elif k in self.labels_light_axisangle:
                    v = euler_to_axisangle(r_params['light']['rotation'])
                    val = v['xyz'.index(k[3])]
                else:
                    val = item[k]
                val = float(val)
                stats[k]['min'] = min(stats[k]['min'], val)
                stats[k]['max'] = max(stats[k]['max'], val)
                delta = val - stats[k]['mean']
                stats[k]['mean'] += delta / (i + 1)
                delta2 = val - stats[k]['mean']
                stats[k]['M2'] += delta * delta2
                # var_test[k].append(val)

        # Calculate variance
        for k in keys:
            stats[k]['var'] = stats[k]['M2'] / len(self.data)
            # assert np.allclose(stats[k]['var'], np.var(var_test[k]))

        # Save stats
        with open(path, 'w') as f:
            logger.info(f'Saving dataset statistics to {path}.')
            yaml.dump(stats, f)
        self.ds_stats = stats

    def _split_dataset(self):
        """
        Split the dataset into train and test sets.
        """
        len_ds = len(self.data)

        # First check if the split is already available here
        tt_split_path = self.path / f'train_test_split_{self.train_test_split_target:.2f}.json'
        if tt_split_path.exists():
            logger.info(f'Loading train/test splits from {tt_split_path}')
            with open(tt_split_path, 'r') as f:
                tt_split = json.load(f)
                self.train_test_split_actual = tt_split['train_test_split_actual']
                self.train_idxs = np.array(tt_split['train_idxs'])
                self.test_idxs = np.array(tt_split['test_idxs'])
            return

        # Otherwise, split the dataset
        logger.info('Splitting dataset into train/test sets.')
        idxs = np.arange(len_ds)
        np.random.shuffle(idxs)
        split_idx = int(len_ds * self.train_test_split_target)
        self.train_idxs = idxs[:split_idx]
        self.test_idxs = idxs[split_idx:]
        self.train_test_split_actual = len(self.train_idxs) / len_ds

        # Save the split
        logger.info(f'Saving train/test splits to {tt_split_path}')
        tt_split = {
            'train_test_split_target': self.train_test_split_target,
            'train_test_split_actual': self.train_test_split_actual,
            'train_idxs': self.train_idxs.tolist(),
            'test_idxs': self.test_idxs.tolist(),
        }
        with open(tt_split_path, 'w') as f:
            json.dump(tt_split, f, indent=4)

    def get_size(self, train_or_test: str = None) -> int:
        """
        Get the size of the dataset.
        """
        if train_or_test == 'train':
            return self.size_train
        elif train_or_test == 'test':
            return self.size_test
        else:
            return self.size_all

    def load_item(self, idx: int) -> Tuple[dict, Image.Image, Dict[str, np.ndarray]]:
        """
        Get an item from the dataset.
        """
        assert idx in self.data, f'Index {idx} not in dataset!'
        item = self.data[idx]
        r_params = item['rendering_parameters']

        # Load the image
        img = Image.open(item['image']).convert('L')

        def z_transform(x, k):
            return (x - self.ds_stats[k]['mean']) / np.sqrt(self.ds_stats[k]['var'])

        # Format the output vectors
        params = {}

        # Zingg parameters are always in [0, 1]
        if self.ds_args.train_zingg:
            params['zingg'] = np.array([
                item['si'],
                item['il']
            ])

        # Distances are always in [0, 1], where 0 indicates a collapsed face / missing distance
        if self.ds_args.train_distances:
            params['distances'] = np.array([
                item[k]
                for k in self.labels_distances_active
            ])
            if self.ds_args.use_distance_switches:
                params['distance_switches'] = np.array([
                    0 if item[k] == 0 else 1
                    for k in self.labels_distances_active
                ])

        # Transformation parameters are normalised by the dataset statistics
        if self.ds_args.train_transformation:
            # Normalise the location coordinates to [-1, 1], but scale together
            location = np.array(r_params['location'])
            l_max = np.array([self.ds_stats[xyz]['max'] for xyz in 'xyz'])
            l_min = np.array([self.ds_stats[xyz]['min'] for xyz in 'xyz'])
            ranges = l_max - l_min
            range_max = ranges.max()
            location = 2 * (location - l_min) / range_max - 1

            # Standardise the scale
            scale = z_transform(r_params['scale'], 's')

            # Rotation representation
            R0 = Rotation.from_euler('xyz', r_params['rotation'])

            # Apply the rotation to the symmetry group
            sym_group = self._get_symmetry_group(idx)
            sym_R = np.zeros((len(sym_group), 3, 3))
            for i, R in enumerate(sym_group):
                sym_R[i] = (R0 * R).as_matrix()

            # # Check the symmetry group
            # mesh = self.load_mesh(idx)
            # v = mesh.vertices
            # v0 = R0.apply(v)
            # for r_mat in sym_R:
            #     R = Rotation.from_matrix(r_mat)
            #     v_rotated = R.apply(v)
            #     cd = cdist(v0, v_rotated)
            #     min_vertex_dists = cd.min(axis=1)
            #     assert np.all(min_vertex_dists < 0.1), 'Symmetry group is not correct!'

            # Get rotation representation
            if self.ds_args.preangles_mode == PREANGLES_MODE_SINCOS:
                rotation_preangles = np.column_stack([
                    np.sin(r_params['rotation']),
                    np.cos(r_params['rotation'])
                ]).ravel()

            elif self.ds_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                # Canonical rotations takes the quaternion with the smallest w value
                if self.ds_args.use_canonical_rotations:
                    sym_q = np.array([Rotation.from_matrix(R).as_quat(canonical=True)[[3, 0, 1, 2]] for R in sym_R])
                    q_canon = sym_q[np.argmin(sym_q[:, 0])]
                    rotation_preangles = q_canon
                else:
                    rotation_preangles = R0.as_quat(canonical=True)[[3, 0, 1, 2]]

            else:
                assert self.ds_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                rotation_preangles = R0.as_rotvec()

            params['transformation'] = np.array([
                *location,
                scale,
                *rotation_preangles
            ])
            params['sym_rotations'] = sym_R

        # Material parameters are z-score standardised
        if self.ds_args.train_material and len(self.labels_material_active) > 0:
            m = r_params['material']
            m_params = []
            if 'b' in self.labels_material_active:
                m_params.append(z_transform(m['brightness'], 'b'))
            if 'ior' in self.labels_material_active:
                m_params.append(z_transform(m['ior'], 'ior'))
            if 'r' in self.labels_material_active:
                m_params.append(z_transform(m['roughness'], 'r'))
            params['material'] = np.array(m_params)

        if self.ds_args.train_light:
            # Standardise the energy
            energy = z_transform(r_params['light']['energy'], 'e')

            if not self.renderer_args.transmission_mode:
                # Normalise the location coordinates to [-1, 1], but scale together
                location = np.array(r_params['light']['location'])
                l_max = np.array([self.ds_stats[xyz]['max'] for xyz in ['lx', 'ly', 'lz']])
                l_min = np.array([self.ds_stats[xyz]['min'] for xyz in ['lx', 'ly', 'lz']])
                ranges = l_max - l_min
                range_max = ranges.max()
                location = 2 * (location - l_min) / range_max - 1

                # Rotation pre-angles - [-1, 1]
                if self.ds_args.preangles_mode == PREANGLES_MODE_SINCOS:
                    rotation_preangles = np.column_stack([
                        np.sin(r_params['light']['rotation'][:2]),
                        np.cos(r_params['light']['rotation'][:2])
                    ]).ravel()
                elif self.ds_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                    rotation_preangles = euler_to_quaternion(r_params['light']['rotation'])
                else:
                    assert self.ds_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                    rotation_preangles = euler_to_axisangle(r_params['light']['rotation'])

                params['light'] = np.array([
                    *location,
                    energy,
                    *rotation_preangles,
                ])

            else:
                params['light'] = np.array([energy, ])

        return item, img, params

    def load_mesh(self, idx: int) -> Trimesh:
        """
        Load the mesh for an item.
        """
        obj_path = self.path / 'crystals' / f'crystals_{idx // self.dataset_args.n_objs_per_file:05d}.obj'
        with open(obj_path, 'r') as f:
            scene = load_obj(
                f,
                group_material=False,
                skip_materials=True,
                maintain_order=True,
            )
            mesh_params = scene['geometry'][f'{self.dataset_args.crystal_id}_{idx:06d}']
            mesh = Trimesh(vertices=mesh_params['vertices'], faces=mesh_params['faces'], process=True, validate=True)
            assert mesh.is_watertight, 'Mesh is not watertight!'
        return mesh

    def _get_symmetry_group(self, idx: int) -> List[Rotation]:
        """
        Get the rotational symmetry group of the crystal from cache or calculation.
        """
        item = self.data[idx]
        d_str = ''.join([f'{item[k]:.2f}' for k in self.labels_distances])
        if d_str in self.symmetry_groups:
            return self.symmetry_groups[d_str]
        sym_group = self._calculate_symmetry_group(idx)
        self.symmetry_groups[d_str] = sym_group
        return sym_group

    def _calculate_symmetry_group(self, idx: int) -> List[Rotation]:
        """
        Calculate the rotational symmetry group of the crystal.
        """
        mesh = self.load_mesh(idx)
        sym_group = []
        vertices = mesh.vertices
        for R in self.rotation_matrices:
            vertices_rotated = R.apply(vertices)
            cd = cdist(vertices, vertices_rotated)
            min_vertex_dists = cd.min(axis=1)
            if np.all(min_vertex_dists < 0.1):
                sym_group.append(R)
        return sym_group

    def _calculate_rotation_matrices(self):
        """
        Calculate a set of rotation matrices for a given number of angles.
        """
        if self.ds_args.check_symmetries == 0:
            angles = [0, ]
        else:
            angles = np.linspace(0, np.pi, 1 + self.ds_args.check_symmetries, endpoint=True)
        Rs_all = []
        for rx, ry, rz in product(angles, repeat=3):
            R = Rotation.from_euler('XYZ', [rx, ry, rz])
            Rs_all.append(R)

        # De-duplicate
        Rs = []
        for i, Ri in enumerate(Rs_all):
            for j, Rj in enumerate(Rs_all):
                if j < i:
                    if np.allclose(Ri.as_matrix(), Rj.as_matrix()):
                        break
                    continue
                elif i == j:
                    Rs.append(Ri)
                else:
                    break

        return Rs

    def denormalise_rendering_params(
            self,
            outputs: Dict[str, torch.Tensor],
            idx: int = 0,
            default_rendering_params: Optional[Dict[str, Any]] = None
    ) -> dict:
        """
        Denormalise the rendering parameters.
        """

        def inverse_z_transform(z, k):
            return float(z * np.sqrt(self.ds_stats[k]['var']) + self.ds_stats[k]['mean'])

        r_params = {}

        # Transformation parameters
        if 'transformation' in outputs:
            trans = outputs['transformation']
            if trans.ndim == 2:
                trans = trans[idx]
            location = to_numpy(trans[:3])
            l_max = np.array([self.ds_stats[xyz]['max'] for xyz in 'xyz'])
            l_min = np.array([self.ds_stats[xyz]['min'] for xyz in 'xyz'])
            ranges = l_max - l_min
            range_max = ranges.max()
            location = (location + 1) / 2 * range_max + l_min
            r_params['location'] = location.tolist()

            # Inverse z-transform the scale
            r_params['scale'] = inverse_z_transform(trans[3].item(), 's')

            # Rotation angles
            if self.ds_args.preangles_mode == PREANGLES_MODE_SINCOS:
                angles = from_preangles(trans[4:]).tolist()
            elif self.ds_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                angles = quaternion_to_euler(trans[4:]).tolist()
            else:
                assert self.ds_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                angles = axisangle_to_euler(trans[4:]).tolist()
            r_params['rotation'] = angles
        else:
            assert default_rendering_params is not None, 'Need to provide defaults for missing transformation parameters.'
            r_params['location'] = default_rendering_params['location']
            r_params['scale'] = default_rendering_params['scale']
            r_params['rotation'] = default_rendering_params['rotation']

        # Material parameters
        m_params = {}
        if 'material' in outputs:
            m = outputs['material']
            if m.ndim == 2:
                m = m[idx]
            m_idx = 0
            if 'b' in self.labels_material_active:
                m_params['brightness'] = inverse_z_transform(m[m_idx].item(), 'b')
                m_idx += 1
            if 'ior' in self.labels_material_active:
                m_params['ior'] = inverse_z_transform(m[m_idx].item(), 'ior')
                m_idx += 1
            if 'r' in self.labels_material_active:
                m_params['roughness'] = inverse_z_transform(m[m_idx].item(), 'r')
        for k in ['brightness', 'ior', 'roughness']:
            if k not in m_params:
                if k != 'roughness':
                    assert default_rendering_params is not None, 'Need to provide defaults for missing material parameters.'
                if default_rendering_params is not None and k in default_rendering_params['material']:
                    m_params[k] = default_rendering_params['material'][k]
                elif k == 'roughness':  # Handle roughness different as it was added later so might not have a default
                    m_params[k] = 0
        r_params['material'] = m_params

        # Light parameters
        if self.ds_args.train_light:
            light = outputs['light']
            if light.ndim == 2:
                light = light[idx]

            if self.renderer_args.transmission_mode:
                l_params = {
                    'location': [0, 0, 0],
                    'energy': inverse_z_transform(light[0].item(), 'e'),
                    'rotation': [0, 0, 0],
                    'angle': 0,
                }

            else:
                location = to_numpy(light[:3])
                l_max = np.array([self.ds_stats[xyz]['max'] for xyz in ['lx', 'ly', 'lz']])
                l_min = np.array([self.ds_stats[xyz]['min'] for xyz in ['lx', 'ly', 'lz']])
                ranges = l_max - l_min
                range_max = ranges.max()
                location = (location + 1) / 2 * range_max + l_min

                # Inverse z-transform the energy
                energy = inverse_z_transform(light[3].item(), 'e')

                # Rotation angles
                if self.ds_args.preangles_mode == PREANGLES_MODE_SINCOS:
                    angles = from_preangles(light[4:]).tolist() + [0., ]
                elif self.ds_args.preangles_mode == PREANGLES_MODE_QUATERNION:
                    angles = quaternion_to_euler(light[4:]).tolist()
                else:
                    assert self.ds_args.preangles_mode == PREANGLES_MODE_AXISANGLE
                    angles = axisangle_to_euler(light[4:]).tolist()

                l_params = {
                    'location': location.tolist(),
                    'energy': energy,
                    'rotation': angles,
                    'angle': angles[0],
                }
        elif default_rendering_params is not None:
            logger.warning('Missing light parameters and no defaults provided.')
            l_params = default_rendering_params['light']
            if self.renderer_args.transmission_mode:
                l_params['location'] = [0, 0, 0]
                l_params['rotation'] = [0, 0, 0]
                l_params['angle'] = 0
        else:
            l_params = {}
        r_params['light'] = l_params

        return r_params
