from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import Union

from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.crystal import ROTATION_MODES, ROTATION_MODE_AXISANGLE
from crystalsizer3d.util.utils import str2bool


class DatasetTrainingArgs(BaseArgs):
    def __init__(
            self,
            dataset_path: Union[str, Path],
            check_image_paths: bool = False,
            train_test_split: float = 0.8,
            augment: bool = False,
            train_zingg: bool = True,
            train_distances: bool = True,
            train_transformation: bool = True,
            train_material: bool = True,
            train_light: bool = True,
            train_predictor: bool = True,
            train_generator: bool = True,
            train_combined: bool = False,
            rotation_mode: str = 'axisangle',
            use_distance_switches: bool = True,
            add_coord_grid: bool = False,
            check_symmetries: int = 0,
            use_canonical_rotations: bool = False,
            **kwargs
    ):
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        assert dataset_path.exists(), f'Dataset path does not exist: {dataset_path}'
        self.dataset_path = dataset_path
        self.check_image_paths = check_image_paths
        self.train_test_split = train_test_split
        self.augment = augment
        self.train_zingg = train_zingg
        self.train_distances = train_distances
        self.train_transformation = train_transformation
        self.train_material = train_material
        self.train_light = train_light
        self.train_predictor = train_predictor
        self.train_generator = train_generator
        if train_combined:
            assert train_predictor and train_generator, \
                'train_predictor and train_generator must be True when train_combined is True.'
        self.train_combined = train_combined
        assert rotation_mode in ROTATION_MODES, f'Invalid rotation mode {rotation_mode}, must be one of {ROTATION_MODES}'
        self.rotation_mode = rotation_mode
        self.use_distance_switches = use_distance_switches
        self.add_coord_grid = add_coord_grid
        self.check_symmetries = check_symmetries
        self.use_canonical_rotations = use_canonical_rotations

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Dataset Args')
        group.add_argument('--dataset-path', type=Path,
                           help='Path to the dataset.')
        group.add_argument('--check-image-paths', type=str2bool, default=False,
                           help='Check that the image paths exist for all images in dataset (slow for large datasets).')
        group.add_argument('--train-test-split', type=float, default=0.8,
                           help='Train/test split.')
        group.add_argument('--augment', type=str2bool, default=False,
                           help='Apply data augmentation.')
        group.add_argument('--train-zingg', type=str2bool, default=True,
                           help='Train Zingg ratios.')
        group.add_argument('--train-distances', type=str2bool, default=True,
                           help='Train distance parameters.')
        group.add_argument('--train-transformation', type=str2bool, default=True,
                           help='Train transformation parameters.')
        group.add_argument('--train-material', type=str2bool, default=True,
                           help='Train material parameters.')
        group.add_argument('--train-light', type=str2bool, default=True,
                           help='Train light parameters.')
        group.add_argument('--train-predictor', type=str2bool, default=True,
                           help='Train predictor network.')
        group.add_argument('--train-generator', type=str2bool, default=True,
                           help='Train generator network.')
        group.add_argument('--train-combined', type=str2bool, default=False,
                           help='Train the full predictor and generator networks together as a visual autoencoder.')
        group.add_argument('--rotation-mode', type=str, default=ROTATION_MODE_AXISANGLE, choices=ROTATION_MODES,
                           help='Rotation mode (axisangle or quaternion).')
        group.add_argument('--use-distance-switches', type=str2bool, default=True,
                           help='Use distance switches.')
        group.add_argument('--add-coord-grid', type=str2bool, default=False,
                           help='Add coordinate grid to images as a separate channel.')
        group.add_argument('--check-symmetries', type=int, default=0,
                           help='Check symmetries in the meshes so we can use a more reasonable rotational error to the group.')
        group.add_argument('--use-canonical-rotations', type=str2bool, default=False,
                           help='Use canonical rotations for the symmetry group.')
        return group
