from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple

from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.crystal import ROTATION_MODES, ROTATION_MODE_AXISANGLE
from crystalsizer3d.util.utils import str2bool

CRYSTAL_IDS = ['LGLUAC01', 'LGLUAC02', 'LGLUAC11', 'AMBNAC01', 'AMBNAC04', 'IBPRAC', 'IBPRAC04', 'CEKBEZ']


class DatasetSyntheticArgs(BaseArgs):
    def __init__(
            self,
            crystal_id: str,
            miller_indices: List[Tuple[int, int, int]],
            ratio_means: List[float],
            ratio_stds: List[float],
            zingg_bbox: List[float],
            distance_constraints: Optional[str] = None,
            n_samples: int = 1000,
            obj_path: Optional[Path] = None,
            batch_size: int = 100,
            image_size: int = 200,
            centre_crystals: bool = False,
            optimise_rotation: bool = True,
            rotation_mode: str = ROTATION_MODE_AXISANGLE,

            min_area: float = 0.05,
            max_area: float = 0.5,
            validate_n_samples: int = 10,
            generate_blender: bool = False,
            **kwargs
    ):
        # Check arguments are valid
        assert crystal_id in CRYSTAL_IDS, f'Crystal ID must be one of {CRYSTAL_IDS}. {crystal_id} received.'
        self.crystal_id = crystal_id
        assert len(miller_indices) > 0, \
            f'Number of miller indices must be greater than 0. {len(miller_indices)} received.'
        for hkl in miller_indices:
            assert len(hkl) == 3, f'Miller indices must have 3 values. {hkl} received.'
        self.miller_indices = miller_indices
        assert len(ratio_means) == len(miller_indices), \
            f'Number of ratio means must equal number of miller indices. {len(ratio_means)} != {len(miller_indices)}'
        self.ratio_means = ratio_means
        assert len(ratio_stds) == len(miller_indices), \
            f'Number of ratio stds must equal number of miller indices. {len(ratio_stds)} != {len(miller_indices)}'
        self.ratio_stds = ratio_stds
        assert len(zingg_bbox) == 4, f'Zingg bounding box must have 4 values. {len(zingg_bbox)} received.'
        for i, v in enumerate(zingg_bbox):
            assert 0.01 <= v <= 1, f'Zingg bounding box values must be between 0.01 and 1. {v} received.'
            if i % 2 == 0:
                assert v < zingg_bbox[i + 1], \
                    f'Zingg bounding box values must be in (min,max,min,max) form. {zingg_bbox} received.'
        self.zingg_bbox = zingg_bbox
        self.distance_constraints = distance_constraints
        assert n_samples > 0, f'Number of samples must be greater than 0. {n_samples} received.'
        self.n_samples = n_samples
        param_path = None
        if obj_path is not None:
            if type(obj_path) == str:
                obj_path = Path(obj_path)
            assert obj_path.exists(), f'Object file "{obj_path}" does not exist.'
            param_path = obj_path.parent / 'parameters.csv'
            if not param_path.exists() and obj_path.is_dir():
                param_path = obj_path / 'parameters.csv'
            if not param_path.exists():
                param_path = obj_path.parent.parent / 'parameters.csv'
            assert param_path.exists(), f'Parameters file "{param_path}" does not exist.'
        self.obj_path = obj_path
        self.batch_size = batch_size
        self.param_path = param_path
        assert image_size > 0, f'Image size must be greater than 0. {image_size} received.'
        self.image_size = image_size
        self.centre_crystals = centre_crystals
        self.optimise_rotation = optimise_rotation
        assert rotation_mode in ROTATION_MODES, f'Rotation mode must be one of {ROTATION_MODES}. {rotation_mode} received.'
        self.rotation_mode = rotation_mode
        assert min_area > 0.01, f'Minimum area must be greater than 0.01. {min_area} received.'
        self.min_area = min_area
        assert max_area > min_area, f'Maximum area must be greater than minimum area. {max_area} received.'
        assert max_area < 1, f'Maximum area must be less than 1. {max_area} received.'
        self.max_area = max_area
        self.validate_n_samples = validate_n_samples
        if generate_blender and n_samples > 100:
            raise ValueError('Generating more than 100 blender files at a time is '
                             'disabled for sanity reasons (disk space).')
        self.generate_blender = generate_blender

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        parser.add_argument('--crystal-id', type=str, default='LGLUAC01',
                            help='Crystal ID to generate images for.')
        parser.add_argument('--miller-indices', type=lambda s: [tuple(int(i) for i in item) for item in s.split(',')],
                            default='101,021,010', help='Miller indices of the canonical distances.')
        parser.add_argument('--ratio-means', type=lambda s: [float(item) for item in s.split(',')],
                            default='1,1,1,1,1,1', help='Means of the ratios of growth rates.')
        parser.add_argument('--ratio-stds', type=lambda s: [float(item) for item in s.split(',')],
                            default='0.5,0.5,0.5,0.5,0.5,0.5', help='Standard deviations of the growth rates.')
        parser.add_argument('--zingg-bbox', type=lambda s: [float(item) for item in s.split(',')],
                            default='0.01,1.0,0.01,1.0',
                            help='Bounding box of the Zingg diagram to restrict shapes to (min_x,max_x,min_y,max_y).')
        parser.add_argument('--distance-constraints', type=str, default=None,
                            help='Constraints to apply to the crystal face distances. Must be in the format "111>012>0".')
        parser.add_argument('--n-samples', type=int, default=1,
                            help='Number of samples to generate.')
        parser.add_argument('--obj-path', type=Path, default=None,
                            help='Path to the obj file to use (skipping the crystal generation).')
        parser.add_argument('--batch-size', type=int, default=100,
                            help='Number of meshes to save per obj file.')
        parser.add_argument('--image-size', type=int, default=512,
                            help='Image size.')
        parser.add_argument('--centre-crystals', type=str2bool, default=False,
                            help='Centre the crystals in the image.')
        parser.add_argument('--optimise-rotation', type=str2bool, default=True,
                            help='Optimise the rotation of the crystal to meet the area constraints. '
                                 'If False then just optimise the location and scale. (Default: True)')
        parser.add_argument('--rotation-mode', type=str, default=ROTATION_MODE_AXISANGLE, choices=ROTATION_MODES,
                            help='Which angles representation to use, "axisangle" or "quaternion".')
        parser.add_argument('--min-area', type=float, default=0.05,
                            help='Minimum area of the image covered by the crystal.')
        parser.add_argument('--max-area', type=float, default=0.3,
                            help='Maximum area of the image covered by the crystal.')
        parser.add_argument('--validate-n-samples', type=int, default=10,
                            help='Number of samples to re-generate for validation.')
        parser.add_argument('--generate-blender', type=str2bool, default=False,
                            help='Generate blender files (warning: file size).')
