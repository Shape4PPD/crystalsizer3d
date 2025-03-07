import csv
import json
import re
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import orjson
import torch
import yaml
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from trimesh import Trimesh

from crystalsizer3d import DATASET_PROXY_PATH, logger
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs
from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs
from crystalsizer3d.crystal import Crystal, ROTATION_MODE_AXISANGLE, ROTATION_MODE_QUATERNION
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.util.utils import ArgsCompatibleJSONEncoder, hash_data, to_numpy

DATASET_TYPE_SYNTHETIC = 'synthetic'
DATASET_TYPE_IMAGES = 'images'
DATASET_TYPES = {
    DATASET_TYPE_SYNTHETIC: 'Synthetic crystals',
    DATASET_TYPE_IMAGES: 'Microscope images',
}

PARAMETER_HEADERS = [
    'crystal_id',
    'idx',
    'image',
    'si',
    'il'
]

MIN_NORMALISED_SCALE = 1e-3  # Minimum normalised scale value as using scales <= 0 can cause issues


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
    labels_origin = ['x', 'y', 'z']
    labels_scale = ['s', ]
    labels_rotation_quaternion = ['rw', 'rx', 'ry', 'rz']
    labels_rotation_axisangle = ['rax', 'ray', 'raz']
    labels_material = ['ior', 'r']
    labels_light = ['er', 'eg', 'eb']

    def __init__(
            self,
            dst_args: DatasetTrainingArgs,
    ):
        self.dst_args = dst_args
        path = dst_args.dataset_path
        assert path.exists(), f'Dataset path does not exist: {path}'
        self.path = path
        self.csd_proxy = CSDProxy()

        # Load the data
        self._load_data()

        # Load dataset statistics
        self._load_ds_stats()

        # Split dataset into train and test
        self.train_test_split_target = self.dst_args.train_test_split
        self._split_dataset()

        # Set stats
        self.size_all = len(self.data)
        self.size_train = len(self.train_idxs)
        self.size_test = len(self.test_idxs)
        self.image_size = self.dataset_args.image_size

        # Set labels (and output vector size)
        labels = []
        if self.dst_args.train_zingg:
            labels += self.labels_zingg
        if self.dst_args.train_distances:
            labels += self.labels_distances
            if self.dst_args.use_distance_switches:
                labels += self.labels_distance_switches
        if self.dst_args.train_transformation:
            labels += self.labels_origin
            if self.dst_args.train_scale:
                labels += self.labels_scale
            if self.dst_args.rotation_mode == ROTATION_MODE_QUATERNION:
                labels += self.labels_rotation_quaternion
            else:
                assert self.dst_args.rotation_mode == ROTATION_MODE_AXISANGLE
                labels += self.labels_rotation_axisangle
        if self.dst_args.train_material:
            labels += self.labels_material
        if self.dst_args.train_light:
            labels += self.labels_light

        # If any of the labels have 0 variance in the dataset then remove them
        fixed_parameters = {}
        for k in labels:
            if k[:2] == 'ds':
                continue
            if k not in self.labels_origin + self.labels_scale and self.ds_stats[k]['var'] == 0:
                logger.warning(f'Removing key {k} from parameters as it has 0 variance.')
                fixed_parameters[k] = self.ds_stats[k]['mean']
                if k in self.labels_distances and self.dst_args.use_distance_switches:
                    k_switch = f'ds{self.labels_distances.index(k)}'
                    fixed_parameters[k_switch] = 1 if self.ds_stats[k]['mean'] > 0 else 0
        for k in fixed_parameters.keys():
            labels.remove(k)

        self.labels = labels
        self.label_size = len(labels)
        self.fixed_parameters = fixed_parameters
        self.labels_distances_active = [k for k in self.labels_distances if k in self.labels]
        self.labels_distance_switches_active = [k for k in self.labels_distance_switches if k in self.labels]
        self.labels_transformation_active = [k for k in
                                             self.labels_origin + self.labels_scale
                                             + self.labels_rotation_quaternion + self.labels_rotation_axisangle
                                             if k in self.labels]
        self.labels_material_active = [k for k in self.labels_material if k in self.labels]
        self.labels_light_active = [k for k in self.labels_light if k in self.labels]
        if self.dst_args.train_light:
            assert len(self.labels_light_active) > 0, 'Light parameters not present in dataset so can\'t train them!'

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

        # Load rendering parameters (but don't fail if they don't exist)
        r_params_file = self.path / 'rendering_parameters.json'
        if r_params_file.exists():
            with open(r_params_file, 'rb') as f:
                rendering_parameters = orjson.loads(f.read())
            rendering_parameters = {int(k): v for k, v in rendering_parameters.items()}
        else:
            logger.warning(f'Rendering parameters file {r_params_file} does not exist.')
            rendering_parameters = None

        # Load crystal vertices
        vertices_file = self.path / 'vertices.json'
        if vertices_file.exists():
            with open(vertices_file, 'rb') as f:
                vertices = orjson.loads(f.read())
            vertices = {int(k): v for k, v in vertices.items()}
        else:
            logger.warning(f'Vertices file {vertices_file} does not exist.')
            vertices = None

        # Load data
        with open(self.path / 'parameters.csv', 'r') as f:
            reader = csv.DictReader(f)
            self.headers = reader.fieldnames
            self.labels_distances = [h for h in self.headers if h[0] == 'd']
            self.labels_distance_switches = [f'ds{i}' for i in range(len(self.labels_distances))]

            # Set miller indices
            self.miller_indices = self.dataset_args.miller_indices
            if self.dataset_args.asymmetry is not None:
                self.miller_indices = [tuple(map(int, re.findall(r'-?\d', h.split('_')[1])))
                                       for h in self.labels_distances]

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
                        if self.dst_args.check_image_paths:
                            assert image_path.exists(), f'Image path does not exist: {image_path}'
                        item['image'] = image_path
                        if self.dataset_args.generate_clean:
                            clean_image_path = self.path / 'images_clean' / row['image']
                            if self.dst_args.check_image_paths:
                                assert clean_image_path.exists(), f'Clean image path does not exist: {clean_image_path}'
                            item['image_clean'] = clean_image_path
                    elif header[0] == 'd':
                        item[header] = float(row[header])
                    elif header[0] == 'a':
                        item[header] = float(row[header])
                    elif header in ['si', 'il']:
                        item[header] = float(row[header])

                # Include the rendering parameters
                if rendering_parameters is not None:
                    item['rendering_parameters'] = rendering_parameters[idx]

                    # Add the bumpmap path if required
                    if self.dataset_args.crystal_bumpmap_dim > -1:
                        bumpmap_path = self.path / 'crystal_bumpmaps' / f'{row["image"][:-4]}.npz'
                        if self.dst_args.check_image_paths:
                            assert bumpmap_path.exists(), f'Bumpmap path does not exist: {bumpmap_path}'
                        item['rendering_parameters']['crystal']['bumpmap'] = bumpmap_path

                # Include the crystal vertices
                if vertices is not None:
                    item['vertices'] = vertices[idx]

                # Add the keypoints image path if required
                if self.dst_args.train_keypoint_detector or self.dst_args.train_edge_detector:
                    keypoints_dir = self.path / f'keypoints_wfv={self.dst_args.wireframe_blur_variance:.1f}_kpv={self.dst_args.heatmap_blob_variance:.1f}'
                    keypoints_path = keypoints_dir / f'{row["image"][:-4]}.png'
                    item['keypoints_image'] = keypoints_path

                self.data[idx] = item

    def _load_ds_stats(self):
        """
        Generate or load dataset statistics for normalisation.
        """
        path = self.path / 'ds_stats.yml'
        keys = (self.labels_zingg + self.labels_distances + self.labels_origin +
                self.labels_scale + self.labels_material + self.labels_light)
        if self.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE:
            keys += self.labels_rotation_axisangle
        else:
            keys += self.labels_rotation_quaternion

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
                if k in self.labels_origin:
                    val = r_params['crystal']['origin']['xyz'.index(k)]
                elif k in self.labels_scale:
                    val = r_params['crystal']['scale']
                elif k in self.labels_rotation_quaternion:
                    quat = r_params['crystal']['rotation']
                    val = quat['wxyz'.index(k[1])]
                elif k in self.labels_rotation_axisangle:
                    v = r_params['crystal']['rotation']
                    val = v['xyz'.index(k[2])]
                elif k in self.labels_material:
                    if k == 'ior':
                        val = r_params['crystal']['material_ior']
                    else:
                        assert k == 'r'
                        val = r_params['crystal']['material_roughness']
                elif k in self.labels_light:
                    val = r_params['light_radiance']['rgb'.index(k[1])]
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
            with open(tt_split_path, 'rb') as f:
                tt_split = orjson.loads(f.read())
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

    def load_item(self, idx: int) -> Tuple[dict, Image.Image, Optional[Image.Image], Dict[str, np.ndarray]]:
        """
        Get an item from the dataset.
        """
        assert idx in self.data, f'Index {idx} not in dataset!'
        item = self.data[idx]
        r_params = item['rendering_parameters']

        # Load the noisy and clean images
        img = Image.open(item['image'])
        if item['image_clean'].exists():
            img_clean = Image.open(item['image_clean'])
        else:
            img_clean = None

        def z_transform(x, k):
            return (x - self.ds_stats[k]['mean']) / np.sqrt(self.ds_stats[k]['var'])

        def eps_1_normalise(x, k, eps):
            # Normalise to [eps, 1]
            x_min = self.ds_stats[k]['min']
            x_max = self.ds_stats[k]['max']
            return eps + (1 - eps) * (x - x_min) / (x_max - x_min)

        # Format the output vectors
        params = {}

        # Zingg parameters are always in [0, 1]
        if self.dst_args.train_zingg:
            params['zingg'] = np.array([
                item['si'],
                item['il']
            ])

        # Distances are in (0, 1]
        if self.dst_args.train_distances:
            params['distances'] = np.array([
                item[k]
                for k in self.labels_distances_active
            ])
            if self.dst_args.use_distance_switches:
                params['distance_switches'] = np.array([
                    0 if item[k] == 0 else 1
                    for k in [k.replace('d', 'a') for k in self.labels_distances_active]
                ])

        # Transformation parameters are normalised by the dataset statistics
        if self.dst_args.train_transformation or self.dst_args.train_3d:
            # Normalise the origin to [-1, 1] x 3, but scale together
            origin = np.array(r_params['crystal']['origin'])
            l_max = np.array([self.ds_stats[xyz]['max'] for xyz in 'xyz'])
            l_min = np.array([self.ds_stats[xyz]['min'] for xyz in 'xyz'])
            ranges = l_max - l_min
            range_max = ranges.max()
            origin = 2 * (origin - l_min - ranges / 2) / range_max
            transformation = [*origin, ]

            if self.dst_args.train_scale:
                # Standardise the scale to [eps, 1]
                scale = eps_1_normalise(r_params['crystal']['scale'], 's', MIN_NORMALISED_SCALE)
                transformation.append(scale)
            else:
                # Multiply the distances by the scale
                params['distances'] *= r_params['crystal']['scale']

            # Rotation representation in the dataset
            if self.dataset_args.rotation_mode == ROTATION_MODE_QUATERNION:
                R0 = Rotation.from_quat(r_params['crystal']['rotation'])
            else:
                assert self.dataset_args.rotation_mode == ROTATION_MODE_AXISANGLE
                R0 = Rotation.from_rotvec(r_params['crystal']['rotation'])

            # Rotation representation to be returned
            if self.dst_args.rotation_mode == ROTATION_MODE_QUATERNION:
                rotation = R0.as_quat(canonical=True)[[3, 0, 1, 2]]
            else:
                assert self.dst_args.rotation_mode == ROTATION_MODE_AXISANGLE
                rotation = R0.as_rotvec()
            transformation.extend(list(rotation))
            params['transformation'] = np.array(transformation)

            # Apply the rotation to the symmetry group
            if self.dst_args.check_symmetries > 0:
                sym_group = self._get_symmetry_group(idx)
                sym_R = np.zeros((len(sym_group), 3, 3))
                for i, R in enumerate(sym_group):
                    sym_R[i] = (R0 * R).as_matrix()
                params['sym_rotations'] = sym_R

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

        # 3D vertices are transformed by the normalised transformation parameters
        if self.dst_args.train_3d:
            origin_og = np.array(r_params['crystal']['origin'])
            scale_og = r_params['crystal']['scale']
            origin_new = params['transformation'][:3]
            scale_new = params['transformation'][3]
            if 'vertices' in item:
                vertices_og = np.array(item['vertices'])
                vertices_new = (vertices_og - origin_og) / scale_og * scale_new + origin_new
                params['vertices'] = vertices_new

            # Include the face areas for the 3D loss
            params['face_areas'] = np.array([
                item[k]
                for k in [k.replace('d', 'a') for k in self.labels_distances_active]
            ])

        # Material parameters are z-score standardised
        if self.dst_args.train_material and len(self.labels_material_active) > 0:
            m_params = []
            if 'ior' in self.labels_material_active:
                m_params.append(z_transform(r_params['crystal']['material_ior'], 'ior'))
            if 'r' in self.labels_material_active:
                m_params.append(z_transform(r_params['crystal']['material_roughness'], 'r'))
            params['material'] = np.array(m_params)

        if self.dst_args.train_light:
            # Standardise the radiance
            params['light'] = np.array([
                z_transform(r_params['light_radiance'][i], f'e{rgb}')
                for i, rgb in enumerate('rgb')
            ])

        # Include the projected vertex positions for the keypoint detector
        if self.dst_args.train_keypoint_detector:
            path = item['keypoints_image']
            assert path.exists(), \
                f'Keypoints image path does not exist: {path}. Run `scripts/generate_synthetic_dataset_keypoints.py` to generate keypoints images.'
            kp_img = np.array(Image.open(item['keypoints_image'])).astype(np.float32) / 255
            kp_img = np.clip(kp_img, 0, 1)
            params['wireframe'] = kp_img[..., :2].transpose(2, 0, 1)
            params['kp_heatmap'] = kp_img[..., 2]

        return item, img, img_clean, params

    def load_crystal(
            self,
            idx: Optional[int] = None,
            r_params: Optional[Dict[str, Any]] = None,
            zero_origin: bool = False,
            zero_rotation: bool = False,
            use_bumpmap: bool = True,
            merge_vertices: bool = False
    ) -> Crystal:
        """
        Load the crystal for an item.
        """
        cs = self.csd_proxy.load(self.dataset_args.crystal_id)
        if idx is not None:
            r_params = self.data[idx]['rendering_parameters']
        else:
            assert r_params is not None, 'Need to provide either an index or parameters.'
        crystal = Crystal(
            lattice_unit_cell=cs.lattice_unit_cell,
            lattice_angles=cs.lattice_angles,
            miller_indices=self.miller_indices,
            point_group_symbol=cs.point_group_symbol,
            scale=r_params['crystal']['scale'],
            distances=r_params['crystal']['distances'],
            origin=None if zero_origin else r_params['crystal']['origin'],
            rotation=None if zero_rotation else r_params['crystal']['rotation'],
            rotation_mode=self.dataset_args.rotation_mode,
            material_roughness=r_params['crystal']['material_roughness'],
            material_ior=r_params['crystal']['material_ior'],
            use_bumpmap=use_bumpmap and self.dataset_args.crystal_bumpmap_dim > 0,
            bumpmap_dim=self.dataset_args.crystal_bumpmap_dim,
            merge_vertices=merge_vertices,
        )
        return crystal

    def load_mesh(self, idx: int) -> Trimesh:
        """
        Load the mesh for an item.
        """
        crystal = self.load_crystal(idx, zero_origin=True, zero_rotation=True)
        mesh = Trimesh(
            vertices=to_numpy(crystal.mesh_vertices),
            faces=to_numpy(crystal.mesh_faces),
            process=True,
            validate=True
        )
        assert mesh.is_watertight, 'Mesh is not watertight!'
        return mesh

    def _get_symmetry_group(self, idx: int) -> List[Rotation]:
        """
        Get the rotational symmetry group of the crystal from cache or calculation.
        """
        item = self.data[idx]
        d_str = ','.join([f'{item[k]:.2f}' for k in self.labels_distances])
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
        if self.dst_args.check_symmetries == 0:
            angles = [0, ]
        else:
            angles = np.linspace(0, np.pi, 1 + self.dst_args.check_symmetries, endpoint=True)
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
            default_rendering_params: Optional[Dict[str, Any]] = None,
            copy_bumpmap: bool = False,
            copy_bubbles: bool = False,
    ) -> dict:
        """
        Denormalise the rendering parameters.
        copy_bumpmap: If True, copy the bumpmap from the default rendering parameters if it is missing.
        copy_bubbles: If True, copy the bubbles from the default rendering parameters if they are missing.
        """
        return denormalise_rendering_params(
            dst_args=self.dst_args,
            dataset_args=self.dataset_args,
            ds_stats=self.ds_stats,
            labels_distances=self.labels_distances,
            labels_distances_active=self.labels_distances_active,
            labels_material_active=self.labels_material_active,
            outputs=outputs,
            idx=idx,
            default_rendering_params=default_rendering_params,
            copy_bumpmap=copy_bumpmap,
            copy_bubbles=copy_bubbles,
        )

    def prep_distances(
            self,
            distance_vals: torch.Tensor,
            switches: Optional[torch.Tensor] = None,
            missing_distance_val: float = 100.,
            normalise: bool = True
    ) -> torch.Tensor:
        """
        Prepare the distance values.
        """
        return prep_distances(
            dst_args=self.dst_args,
            dataset_args=self.dataset_args,
            ds_stats=self.ds_stats,
            labels_distances=self.labels_distances,
            labels_distances_active=self.labels_distances_active,
            distance_vals=distance_vals,
            switches=switches,
            missing_distance_val=missing_distance_val,
            normalise=normalise,
        )


class DatasetProxy:
    ds_cache: Dict[str, Any] = {}
    ds: Optional[Dataset]

    def __init__(
            self,
            dst_args: DatasetTrainingArgs,
    ):
        self.dst_args = dst_args
        self.path = dst_args.dataset_path
        self.cache_path = DATASET_PROXY_PATH / f'{dst_args.hash()}.json'
        if not self._load_cache():
            self._load_dataset()

    def __getattr__(self, name):
        # Return attributes from the cache
        if name in self.ds_cache:
            if name == 'dataset_args' and not isinstance(self.ds_cache[name], DatasetSyntheticArgs):
                self.ds_cache[name] = DatasetSyntheticArgs.from_args(self.ds_cache[name])
            return self.ds_cache[name]

        # Load dataset
        if name == 'ds':
            self._load_dataset()
            return self.ds

        # Return attributes from the dataset
        if hasattr(self.ds, name):
            val = getattr(self.ds, name)
            if isinstance(val, (int, float, str, list, dict, tuple, BaseArgs)):
                self._update_cache(name, val)
            return val

        raise AttributeError(f'Attribute {name} not found in DatasetProxy or Dataset.')

    def _load_dataset(self):
        self.ds = Dataset(dst_args=self.dst_args)

    def _load_cache(self):
        DATASET_PROXY_PATH.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            return False
        try:
            with open(self.cache_path, 'rb') as f:
                self.ds_cache = orjson.loads(f.read())
        except Exception as e:
            logger.error(f'Could not load cache from {self.cache_path}: {e}. Removing file.')
            self.cache_path.unlink()
            return False
        logger.info(f'Loaded dataset proxy cache from {self.cache_path}.')
        return True

    def _update_cache(self, name, val):
        self.ds_cache[name] = val
        with open(self.cache_path, 'w') as f:
            json.dump(self.ds_cache, f, sort_keys=True, cls=ArgsCompatibleJSONEncoder, indent=4)

    def denormalise_rendering_params(
            self,
            outputs: Dict[str, torch.Tensor],
            idx: int = 0,
            default_rendering_params: Optional[Dict[str, Any]] = None,
            copy_bumpmap: bool = False,
            copy_bubbles: bool = False,
    ) -> dict:
        """
        Denormalise the rendering parameters.
        copy_bumpmap: If True, copy the bumpmap from the default rendering parameters if it is missing.
        copy_bubbles: If True, copy the bubbles from the default rendering parameters if they are missing.
        """
        return denormalise_rendering_params(
            dst_args=self.dst_args,
            dataset_args=self.dataset_args,
            ds_stats=self.ds_stats,
            labels_distances=self.labels_distances,
            labels_distances_active=self.labels_distances_active,
            labels_material_active=self.labels_material_active,
            outputs=outputs,
            idx=idx,
            default_rendering_params=default_rendering_params,
            copy_bumpmap=copy_bumpmap,
            copy_bubbles=copy_bubbles,
        )

    def prep_distances(
            self,
            distance_vals: torch.Tensor,
            switches: Optional[torch.Tensor] = None,
            missing_distance_val: float = 100.,
            normalise: bool = True
    ) -> torch.Tensor:
        """
        Prepare the distance values.
        """
        return prep_distances(
            dst_args=self.dst_args,
            dataset_args=self.dataset_args,
            ds_stats=self.ds_stats,
            labels_distances=self.labels_distances,
            labels_distances_active=self.labels_distances_active,
            distance_vals=distance_vals,
            switches=switches,
            missing_distance_val=missing_distance_val,
            normalise=normalise,
        )


def denormalise_rendering_params(
        dst_args: DatasetTrainingArgs,
        dataset_args: DatasetSyntheticArgs,
        ds_stats: Dict[str, Dict[str, float]],
        labels_distances: List[str],
        labels_distances_active: List[str],
        labels_material_active: List[str],
        outputs: Dict[str, torch.Tensor],
        idx: int = 0,
        default_rendering_params: Optional[Dict[str, Any]] = None,
        copy_bumpmap: bool = False,
        copy_bubbles: bool = False,
        switches: Optional[torch.Tensor] = None,
        missing_distance_val: float = 100.
) -> dict:
    """
    Denormalise the rendering parameters.
    copy_bumpmap: If True, copy the bumpmap from the default rendering parameters if it is missing.
    copy_bubbles: If True, copy the bubbles from the default rendering parameters if they are missing.
    """

    def inverse_z_transform(z, k):
        return float(z * np.sqrt(ds_stats[k]['var']) + ds_stats[k]['mean'])

    def inverse_eps_1_normalise(z, k, eps):
        x_min = ds_stats[k]['min']
        x_max = ds_stats[k]['max']
        return x_min + (z - eps) * (x_max - x_min) / (1 - eps)

    r_params = {}
    c_params = {}

    # Distance parameters
    assert 'distances' in outputs, 'Distances are missing!'
    dist = outputs['distances']
    if dist.ndim == 2:
        dist = dist[idx]
    c_params['distances'] = prep_distances(
        dst_args=dst_args,
        dataset_args=dataset_args,
        ds_stats=ds_stats,
        labels_distances=labels_distances,
        labels_distances_active=labels_distances_active,
        distance_vals=dist,
        switches=switches,
        missing_distance_val=missing_distance_val,
        normalise=dst_args.train_scale
    ).tolist()

    # Transformation parameters
    if 'transformation' in outputs:
        trans = outputs['transformation']
        if trans.ndim == 2:
            trans = trans[idx]
        location = to_numpy(trans[:3])
        l_max = np.array([ds_stats[xyz]['max'] for xyz in 'xyz'])
        l_min = np.array([ds_stats[xyz]['min'] for xyz in 'xyz'])
        ranges = l_max - l_min
        range_max = ranges.max()
        location = location * range_max / 2 + l_min + ranges / 2
        c_params['origin'] = location.tolist()

        # Invert the scale back to the original scale
        if dst_args.train_scale:
            c_params['scale'] = inverse_eps_1_normalise(trans[3].item(), 's', MIN_NORMALISED_SCALE)
            r_idx = 4
        else:
            c_params['scale'] = 1
            r_idx = 3

        # Rotation angles
        c_params['rotation'] = trans[r_idx:].tolist()
    else:
        assert default_rendering_params is not None, 'Need to provide defaults for missing transformation parameters.'
        assert 'crystal' in default_rendering_params, 'Need to provide defaults for missing transformation parameters.'
        c_params['origin'] = default_rendering_params['crystal']['origin']
        c_params['scale'] = default_rendering_params['crystal']['scale']
        c_params['rotation'] = default_rendering_params['crystal']['rotation']

    # Material parameters
    if 'material' in outputs:
        m = outputs['material']
        if m.ndim == 2:
            m = m[idx]
        m_idx = 0
        if 'ior' in labels_material_active:
            c_params['material_ior'] = inverse_z_transform(m[m_idx].item(), 'ior')
            m_idx += 1
        if 'r' in labels_material_active:
            c_params['material_roughness'] = inverse_z_transform(m[m_idx].item(), 'r')
    for k in ['ior', 'roughness']:
        m_key = f'material_{k}'
        if (m_key not in c_params
                and default_rendering_params is not None
                and m_key in default_rendering_params['material']):
            c_params[m_key] = default_rendering_params[m_key]
    r_params['crystal'] = c_params

    # Light parameters
    if 'light' in outputs:
        light = outputs['light']
        if light.ndim == 2:
            light = light[idx]
        r_params['light_radiance'] = (
            inverse_z_transform(light[0].item(), 'er'),
            inverse_z_transform(light[1].item(), 'eg'),
            inverse_z_transform(light[2].item(), 'eb'),
        )

    elif default_rendering_params is not None and 'light_radiance' in default_rendering_params:
        r_params['light_radiance'] = default_rendering_params['light_radiance']
    else:
        r_params['light_radiance'] = (0.5, 0.5, 0.5)

    # Copy the surface bumpmap from defaults if required
    if 'bumpmap' in outputs:
        bumpmap = outputs['bumpmap']
        if bumpmap.ndim == 2:
            bumpmap = bumpmap[idx]
        r_params['bumpmap'] = bumpmap
    elif (default_rendering_params is not None
          and 'bumpmap' in default_rendering_params
          and copy_bumpmap):
        r_params['bumpmap'] = default_rendering_params['bumpmap']
    else:
        r_params['bumpmap'] = None

    # Bubbles
    if 'bubbles' in outputs:
        bubbles = outputs['bubbles']
        if bubbles.ndim == 2:
            bubbles = bubbles[idx]
        r_params['bubbles'] = bubbles
    elif (default_rendering_params is not None
          and 'bubbles' in default_rendering_params
          and copy_bubbles):
        r_params['bubbles'] = default_rendering_params['bubbles']
    else:
        r_params['bubbles'] = []

    return r_params


def prep_distances(
        dst_args: DatasetTrainingArgs,
        dataset_args: DatasetSyntheticArgs,
        ds_stats: Dict[str, Dict[str, float]],
        labels_distances: List[str],
        labels_distances_active: List[str],
        distance_vals: torch.Tensor,
        switches: Optional[torch.Tensor] = None,
        missing_distance_val: float = 100.,
        normalise: bool = True
) -> torch.Tensor:
    """
    Prepare the distance values.
    """
    strip_batch = False
    if distance_vals.ndim == 1:
        strip_batch = True
        distance_vals = distance_vals[None, :]
    bs = distance_vals.shape[0]

    # Set distances to something large if the switches are off or if the distance is negative
    if dst_args.use_distance_switches and switches is not None:
        distance_vals = torch.where(switches < .5, missing_distance_val, distance_vals)
    distance_vals[distance_vals < 0] = missing_distance_val

    # Put the distances into the correct order
    distances = torch.zeros(bs, len(labels_distances), device=distance_vals.device)
    pos_active = [labels_distances.index(k) for k in labels_distances_active]
    for i, pos in enumerate(pos_active):
        distances[:, pos] = distance_vals[:, i]

    # Add any distances that are not in the active set
    if dataset_args.asymmetry is None and labels_distances != labels_distances_active:
        for d_lbl in labels_distances:
            if d_lbl not in labels_distances_active:
                d_pos = labels_distances.index(d_lbl)
                distances[:, d_pos] = ds_stats[d_lbl]['mean']

    # Normalise the distances by the maximum (in each batch element)
    if normalise:
        d_max = distances.amax(dim=1, keepdim=True)
        distances = torch.where(
            d_max > 1e-8,
            distances / d_max,
            distances
        )

    # Ensure negative distances are set to the missing value
    distance_vals[distance_vals < 0] = missing_distance_val

    if strip_batch:
        distances = distances[0]

    return distances
