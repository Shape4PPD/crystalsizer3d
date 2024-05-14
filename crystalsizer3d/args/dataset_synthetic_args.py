from argparse import ArgumentParser
from typing import List, Optional, Tuple

from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.crystal import ROTATION_MODES, ROTATION_MODE_AXISANGLE
from crystalsizer3d.util.utils import str2bool

CRYSTAL_IDS = ['LGLUAC01', 'LGLUAC02', 'LGLUAC11', 'AMBNAC01', 'AMBNAC04', 'IBPRAC', 'IBPRAC04', 'CEKBEZ']


class DatasetSyntheticArgs(BaseArgs):
    def __init__(
            self,

            # Dataset generation parameters
            n_samples: int = 100,
            generate_clean: bool = True,
            image_size: int = 200,
            batch_size: int = 100,
            validate_n_samples: int = 10,

            # Rendering settings
            spp: int = 256,
            integrator_max_depth: int = 64,
            integrator_rr_depth: int = 5,

            # Sensor parameters
            camera_distance: float = 100.,
            camera_fov: float = 25.,
            focus_distance: float = 90.,
            aperture_radius: float = 0.5,
            sampler_type: str = 'stratified',

            # Light parameters
            light_z_position: float = -10.1,
            light_scale: float = 50.,
            light_radiance_min: Tuple[float, float, float] = (0.5, 0.5, 0.5),
            light_radiance_max: Tuple[float, float, float] = (0.5, 0.5, 0.5),

            # Growth cell parameters
            cell_z_positions: List[float] = [-10., 0., 50., 60.],
            cell_surface_scale: float = 100.,

            # Crystal parameters
            crystal_id: str = 'LGLUAC01',
            miller_indices: List[Tuple[int, int, int]] = None,
            ratio_means: List[float] = None,
            ratio_stds: List[float] = None,
            zingg_bbox: List[float] = None,
            distance_constraints: Optional[str] = None,
            rotation_mode: str = ROTATION_MODE_AXISANGLE,

            # Crystal layout parameters
            crystal_area_min: float = 0.05,
            crystal_area_max: float = 0.5,
            centre_crystals: bool = False,
            crystal_min_x: float = -10.,
            crystal_max_x: float = 10.,
            crystal_min_y: float = -10.,
            crystal_max_y: float = 10.,

            # Crystal material properties
            min_ior: float = 1.1,
            max_ior: float = 1.6,
            min_roughness: float = 0.0,
            max_roughness: float = 0.4,

            # Bumpmap defects
            crystal_bumpmap_dim: int = -1,
            min_defects: int = 0,
            max_defects: int = 10,
            defect_min_width: float = 0.0001,
            defect_max_width: float = 0.001,
            defect_max_z: float = 0.1,

            # Bubble properties
            min_bubbles: int = 0,
            max_bubbles: int = 0,
            bubbles_min_x: float = -25.,
            bubbles_max_x: float = 25.,
            bubbles_min_y: float = -25.,
            bubbles_max_y: float = 25.,
            bubbles_min_scale: float = 0.01,
            bubbles_max_scale: float = 0.4,
            bubbles_min_roughness: float = 0.05,
            bubbles_max_roughness: float = 0.2,
            bubbles_min_ior: float = 1.1,
            bubbles_max_ior: float = 1.8,

            **kwargs
    ):
        # Dataset generation parameters
        assert n_samples > 0, f'Number of samples must be greater than 0. {n_samples} received.'
        self.n_samples = n_samples
        self.generate_clean = generate_clean
        assert image_size > 0, f'Image size must be greater than 0. {image_size} received.'
        self.image_size = image_size
        self.batch_size = batch_size
        self.validate_n_samples = validate_n_samples

        # Rendering settings
        self.spp = spp
        self.integrator_max_depth = integrator_max_depth
        self.integrator_rr_depth = integrator_rr_depth

        # Sensor parameters
        self.camera_distance = camera_distance
        self.camera_fov = camera_fov
        self.focus_distance = focus_distance
        self.aperture_radius = aperture_radius
        self.sampler_type = sampler_type

        # Light parameters
        self.light_z_position = light_z_position
        self.light_scale = light_scale
        self.light_radiance_min = light_radiance_min
        self.light_radiance_max = light_radiance_max

        # Growth cell parameters
        self.cell_z_positions = cell_z_positions
        self.cell_surface_scale = cell_surface_scale

        # Crystal parameters
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
        assert rotation_mode in ROTATION_MODES, f'Rotation mode must be one of {ROTATION_MODES}. {rotation_mode} received.'
        self.rotation_mode = rotation_mode

        # Crystal layout parameters
        self.centre_crystals = centre_crystals
        assert crystal_area_min > 0.01, f'Minimum area must be greater than 0.01. {crystal_area_min} received.'
        self.crystal_area_min = crystal_area_min
        assert crystal_area_max > crystal_area_min, f'Maximum area must be greater than minimum area. {crystal_area_max} received.'
        assert crystal_area_max < 1, f'Maximum area must be less than 1. {crystal_area_max} received.'
        self.crystal_area_max = crystal_area_max
        self.crystal_min_x = crystal_min_x
        self.crystal_max_x = crystal_max_x
        self.crystal_min_y = crystal_min_y
        self.crystal_max_y = crystal_max_y

        # Crystal material properties
        self.min_ior = min_ior
        self.max_ior = max_ior
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness

        # Bumpmap defects
        self.crystal_bumpmap_dim = crystal_bumpmap_dim
        self.min_defects = min_defects
        self.max_defects = max_defects
        self.defect_min_width = defect_min_width
        self.defect_max_width = defect_max_width
        self.defect_max_z = defect_max_z

        # Bubble properties
        self.min_bubbles = min_bubbles
        self.max_bubbles = max_bubbles
        self.bubbles_min_x = bubbles_min_x
        self.bubbles_max_x = bubbles_max_x
        self.bubbles_min_y = bubbles_min_y
        self.bubbles_max_y = bubbles_max_y
        self.bubbles_min_scale = bubbles_min_scale
        self.bubbles_max_scale = bubbles_max_scale
        self.bubbles_min_roughness = bubbles_min_roughness
        self.bubbles_max_roughness = bubbles_max_roughness
        self.bubbles_min_ior = bubbles_min_ior
        self.bubbles_max_ior = bubbles_max_ior

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Renderer Args')

        # Dataset generation parameters
        group.add_argument('--n-samples', type=int, default=100,
                           help='Number of samples to generate.')
        group.add_argument('--generate-clean', type=str2bool, default=True,
                           help='Generate clean images in addition to interference images.')
        group.add_argument('--image-size', type=int, default=200,
                           help='Image size.')
        group.add_argument('--batch-size', type=int, default=100,
                           help='Number of crystals to render per batch.')
        group.add_argument('--validate-n-samples', type=int, default=10,
                           help='Number of samples to re-generate for validation.')

        # Rendering settings
        group.add_argument('--spp', type=int, default=256,
                           help='Samples per pixel.')

        # Integrator parameters
        group.add_argument('--integrator-max-depth', type=int, default=64,
                           help='Maximum depth for the integrator.')
        group.add_argument('--integrator-rr-depth', type=int, default=5,
                           help='Russian roulette depth.')

        # Camera parameters
        group.add_argument('--camera-distance', type=float, default=100.,
                           help='Camera distance.')
        group.add_argument('--camera-fov', type=float, default=25.,
                           help='Camera field of view.')
        group.add_argument('--focus-distance', type=float, default=90.,
                           help='Focus distance.')
        group.add_argument('--aperture-radius', type=float, default=0.5,
                           help='Aperture radius.')
        group.add_argument('--sampler-type', type=str, default='stratified',
                           help='Sampler type.')

        # Light parameters
        group.add_argument('--light-z-position', type=float, default=-10.1,
                           help='Light z-position.')
        group.add_argument('--light-scale', type=float, default=50.,
                           help='Light scale.')
        group.add_argument('--light-radiance-min', type=lambda s: [float(item) for item in s.split(',')],
                           default='0.5,0.5,0.5', help='Minimum radiance.')
        group.add_argument('--light-radiance-max', type=lambda s: [float(item) for item in s.split(',')],
                           default='0.5,0.5,0.5', help='Maximum radiance.')

        # Growth cell parameters
        group.add_argument('--cell-z-positions', type=float, nargs='+', default=[-10., 0., 50., 60.],
                           help='Cell z-positions.')
        group.add_argument('--cell-surface-scale', type=float, default=100.,
                           help='Cell surface scale.')

        # Crystal parameters
        group.add_argument('--crystal-id', type=str, default='LGLUAC01',
                           help='Crystal ID to generate images for.')
        group.add_argument('--miller-indices', type=lambda s: [tuple(int(i) for i in item) for item in s.split(',')],
                           default='101,021,010', help='Miller indices of the canonical distances.')
        group.add_argument('--ratio-means', type=lambda s: [float(item) for item in s.split(',')],
                           default='1,1,1', help='Means of the ratios of growth rates.')
        group.add_argument('--ratio-stds', type=lambda s: [float(item) for item in s.split(',')],
                           default='0.5,0.5,0.5', help='Standard deviations of the growth rates.')
        group.add_argument('--zingg-bbox', type=lambda s: [float(item) for item in s.split(',')],
                           default='0.01,1.0,0.01,1.0',
                           help='Bounding box of the Zingg diagram to restrict shapes to (min_x,max_x,min_y,max_y).')
        group.add_argument('--distance-constraints', type=str, default=None,
                           help='Constraints to apply to the crystal face distances. Must be in the format "111>012>0".')
        group.add_argument('--rotation-mode', type=str, default=ROTATION_MODE_AXISANGLE, choices=ROTATION_MODES,
                           help='Which angles representation to use, "axisangle" or "quaternion".')

        # Crystal layout and rendering parameters
        group.add_argument('--centre-crystals', type=str2bool, default=False,
                           help='Centre the crystals in the image.')
        group.add_argument('--crystal-area-min', type=float, default=0.05,
                           help='Minimum area of the image covered by the crystal.')
        group.add_argument('--crystal-area-max', type=float, default=0.3,
                           help='Maximum area of the image covered by the crystal.')
        group.add_argument('--crystal-min-x', type=float, default=-10.0,
                           help='Minimum x-coordinate of the crystal origin.')
        group.add_argument('--crystal-max-x', type=float, default=10.0,
                           help='Maximum x-coordinate of the crystal origin.')
        group.add_argument('--crystal-min-y', type=float, default=-10.0,
                           help='Minimum y-coordinate of the crystal origin.')
        group.add_argument('--crystal-max-y', type=float, default=10.0,
                           help='Maximum y-coordinate of the crystal origin.')

        # Crystal material properties
        group.add_argument('--min-ior', type=float, default=1.0,
                           help='Minimum index of refraction.')
        group.add_argument('--max-ior', type=float, default=2.5,
                           help='Maximum index of refraction.')
        group.add_argument('--min-roughness', type=float, default=0.0,
                           help='Minimum roughness.')
        group.add_argument('--max-roughness', type=float, default=0.4,
                           help='Maximum roughness.')

        # Bumpmap defects
        group.add_argument('--crystal-bumpmap-dim', type=int, default=1000,
                           help='Bumpmap dimension. -1 for no bumpmap.')
        group.add_argument('--min-defects', type=int, default=0,
                           help='Minimum number of defects.')
        group.add_argument('--max-defects', type=int, default=10,
                           help='Maximum number of defects.')
        group.add_argument('--defect-min-width', type=float, default=0.0001,
                           help='Minimum defect width.')
        group.add_argument('--defect-max-width', type=float, default=0.001,
                           help='Maximum defect width.')
        group.add_argument('--defect-max-z', type=float, default=1,
                           help='Maximum defect z-coordinate.')

        # Bubbles
        group.add_argument('--min-bubbles', type=int, default=0,
                           help='Minimum number of bubbles.')
        group.add_argument('--max-bubbles', type=int, default=10,
                           help='Maximum number of bubbles.')
        group.add_argument('--bubbles-min-x', type=float, default=-25.,
                           help='Minimum x-coordinate of the bubble origin.')
        group.add_argument('--bubbles-max-x', type=float, default=25.,
                           help='Maximum x-coordinate of the bubble origin.')
        group.add_argument('--bubbles-min-y', type=float, default=-25.,
                           help='Minimum y-coordinate of the bubble origin.')
        group.add_argument('--bubbles-max-y', type=float, default=25.,
                           help='Maximum y-coordinate of the bubble origin.')
        group.add_argument('--bubbles-min-scale', type=float, default=0.01,
                           help='Minimum scale of the bubble.')
        group.add_argument('--bubbles-max-scale', type=float, default=0.4,
                           help='Maximum scale of the bubble.')
        group.add_argument('--bubbles-min-roughness', type=float, default=0.05,
                           help='Minimum roughness of the bubble.')
        group.add_argument('--bubbles-max-roughness', type=float, default=0.2,
                           help='Maximum roughness of the bubble.')
        group.add_argument('--bubbles-min-ior', type=float, default=1.1,
                           help='Minimum index of refraction of the bubble.')
        group.add_argument('--bubbles-max-ior', type=float, default=1.8,
                           help='Maximum index of refraction of the bubble.')

        return group
