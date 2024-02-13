from argparse import ArgumentParser

from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.util.utils import str2bool


class RendererArgs(BaseArgs):
    def __init__(
            self,
            device: str = 'CPU',
            min_ior: float = 1.1,
            max_ior: float = 1.6,
            min_brightness: float = 0.75,
            max_brightness: float = 0.9,
            min_roughness: float = 0.0,
            max_roughness: float = 0.4,
            light_type: str = 'AREA',
            light_angle_min: int = 0,
            light_angle_max: int = 0,
            use_bottom_light: bool = False,
            transmission_mode: bool = False,
            camera_distance: float = 15.0,
            remesh_mode: str = 'VOXEL',
            remesh_octree_depth: int = 4,
            crystal_material_name: str = 'GLASS',
            custom_material_name: str = 'default',
            **kwargs
    ):
        self.device = device
        self.min_ior = min_ior
        self.max_ior = max_ior
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.light_type = light_type
        self.light_angle_min = light_angle_min
        self.light_angle_max = light_angle_max
        self.use_bottom_light = use_bottom_light
        self.transmission_mode = transmission_mode
        self.camera_distance = camera_distance
        self.remesh_mode = remesh_mode
        self.remesh_octree_depth = remesh_octree_depth
        self.crystal_material_name = crystal_material_name
        self.custom_material_name = custom_material_name

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Renderer Args')
        group.add_argument('--device', type=str, default='GPU',
                           help='Device to use for rendering.')
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
        group.add_argument('--light-type', type=str, default='AREA',
                           help='Type of light to use.')
        group.add_argument('--light-angle-min', type=int, default=-90,
                           help='Minimum angle of the light.')
        group.add_argument('--light-angle-max', type=int, default=90,
                           help='Maximum angle of the light.')
        group.add_argument('--use-bottom-light', type=str2bool, default=False,
                           help='Use a light from below.')
        group.add_argument('--transmission-mode', type=str2bool, default=False,
                           help='Use transmission lighting mode.')
        group.add_argument('--camera-distance', type=float, default=15.0,
                           help='Set camera distance.')
        group.add_argument('--remesh-mode', type=str, default='SHARP',
                           help='Remesh mode.')
        group.add_argument('--remesh-octree-depth', type=int, default=8,
                           help='Octree depth for remeshing.')
        group.add_argument('--crystal-material-name', type=str, default='GLASS',
                           help='Crystal material')
        group.add_argument('--custom-material-name', type=str, default='default',
                           help='Choose custom material')
