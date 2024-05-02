from argparse import ArgumentParser
from typing import List, Tuple

from crystalsizer3d.args.base_args import BaseArgs


class RendererArgs(BaseArgs):
    def __init__(
            self,
            spp: int = 256,

            integrator_max_depth: int = 64,
            integrator_rr_depth: int = 5,

            camera_distance: float = 100.,
            camera_fov: float = 25.,
            focus_distance: float = 90.,
            aperture_radius: float = 0.5,
            sampler_type: str = 'stratified',

            light_z_position: float = -10.,
            light_scale: float = 50.,
            light_radiance_min: Tuple[float, float, float] = (0.5, 0.5, 0.5),
            light_radiance_max: Tuple[float, float, float] = (0.5, 0.5, 0.5),

            cell_z_positions: List[float] = [-10., 0., 50., 60.],
            cell_surface_scale: float = 100.,

            crystal_min_x: float = -10.,
            crystal_max_x: float = 10.,
            crystal_min_y: float = -10.,
            crystal_max_y: float = 10.,
            min_ior: float = 1.1,
            max_ior: float = 1.6,
            min_brightness: float = 0.75,
            max_brightness: float = 0.9,
            min_roughness: float = 0.0,
            max_roughness: float = 0.4,

            crystal_bumpmap_dim: int = -1,
            min_defects: int = 0,
            max_defects: int = 10,
            defect_min_width: float = 0.0001,
            defect_max_width: float = 0.001,
            defect_max_z: float = 0.1,

            min_bubbles: int = 0,
            max_bubbles: int = 0,
            bubbles_min_x: float = -10.,
            bubbles_max_x: float = 10.,
            bubbles_min_y: float = -10.,
            bubbles_max_y: float = 10.,
            bubbles_min_scale: float = 0.001,
            bubbles_max_scale: float = 1,
            bubbles_min_roughness: float = 0.05,
            bubbles_max_roughness: float = 0.2,
            bubbles_min_ior: float = 1.1,
            bubbles_max_ior: float = 1.8,

            **kwargs
    ):
        # Rendering settings
        self.spp = spp

        # Integrator parameters
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

        # Crystal properties
        self.crystal_min_x = crystal_min_x
        self.crystal_max_x = crystal_max_x
        self.crystal_min_y = crystal_min_y
        self.crystal_max_y = crystal_max_y
        self.min_ior = min_ior
        self.max_ior = max_ior
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
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
        group.add_argument('--light-z-position', type=float, default=-10.,
                           help='Light z-position.')
        group.add_argument('--light-scale', type=float, default=50.,
                           help='Light scale.')
        group.add_argument('--light-radiance-min', type=float, nargs=3, default=(0.5, 0.5, 0.5),
                           help='Minimum radiance.')
        group.add_argument('--light-radiance-max', type=float, nargs=3, default=(0.5, 0.5, 0.5),
                           help='Maximum radiance.')

        # Growth cell parameters
        group.add_argument('--cell-z-positions', type=float, nargs='+', default=[-10., 0., 50., 60.],
                           help='Cell z-positions.')
        group.add_argument('--cell-surface-scale', type=float, default=100.,
                           help='Cell surface scale.')

        # Crystal properties
        group.add_argument('--crystal-min-x', type=float, default=-10.0,
                           help='Minimum x-coordinate of the crystal origin.')
        group.add_argument('--crystal-max-x', type=float, default=10.0,
                           help='Maximum x-coordinate of the crystal origin.')
        group.add_argument('--crystal-min-y', type=float, default=-10.0,
                           help='Minimum y-coordinate of the crystal origin.')
        group.add_argument('--crystal-max-y', type=float, default=10.0,
                           help='Maximum y-coordinate of the crystal origin.')
        group.add_argument('--min-ior', type=float, default=1.0,
                           help='Minimum index of refraction.')
        group.add_argument('--max-ior', type=float, default=2.5,
                           help='Maximum index of refraction.')
        group.add_argument('--min-brightness', type=float, default=0.8,
                           help='Minimum brightness.')
        group.add_argument('--max-brightness', type=float, default=1.3,
                           help='Maximum brightness.')
        group.add_argument('--min-roughness', type=float, default=0.0,
                           help='Minimum roughness.')
        group.add_argument('--max-roughness', type=float, default=0.4,
                           help='Maximum roughness.')
        group.add_argument('--crystal-material-name', type=str, default='GLASS',
                           help='Crystal material')
        group.add_argument('--custom-material-name', type=str, default='default',
                           help='Choose custom material')
        group.add_argument('--remesh-mode', type=str, default='SHARP',
                           help='Remesh mode.')
        group.add_argument('--remesh-octree-depth', type=int, default=8,
                           help='Octree depth for remeshing.')

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
        group.add_argument('--min-bubbles', type=int, default=50,
                           help='Minimum number of bubbles.')
        group.add_argument('--max-bubbles', type=int, default=50,
                           help='Maximum number of bubbles.')
        group.add_argument('--bubbles-min-x', type=float, default=-10.,
                           help='Minimum x-coordinate of the bubble origin.')
        group.add_argument('--bubbles-max-x', type=float, default=10.,
                           help='Maximum x-coordinate of the bubble origin.')
        group.add_argument('--bubbles-min-y', type=float, default=-10.,
                           help='Minimum y-coordinate of the bubble origin.')
        group.add_argument('--bubbles-max-y', type=float, default=10.,
                           help='Maximum y-coordinate of the bubble origin.')
        group.add_argument('--bubbles-min-scale', type=float, default=0.001,
                           help='Minimum scale of the bubble.')
        group.add_argument('--bubbles-max-scale', type=float, default=1,
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
