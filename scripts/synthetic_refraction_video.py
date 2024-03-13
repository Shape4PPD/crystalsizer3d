import os
import time
from argparse import ArgumentParser, Namespace
from typing import List

import ffmpeg
import numpy as np
import torch
import yaml
from ccdc.io import EntryReader
from kornia.geometry import axis_angle_to_quaternion
from mayavi import mlab
import drjit as dr
import mitsuba as mi

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.util.utils import print_args, to_dict

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('llvm_ad_rgb')
    device = torch.device('cpu')

from mitsuba import ScalarTransform4f as T

# Off-screen rendering
mlab.options.offscreen = True

SHAPE_NAME = 'crystal'
VERTEX_KEY = SHAPE_NAME + '.vertex_positions'
FACES_KEY = SHAPE_NAME + '.faces'
BSDF_KEY = SHAPE_NAME + '.bsdf'
COLOUR_KEY = BSDF_KEY + '.reflectance.value'


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to generate a video of a rendered crystal rotating.')

    # Crystal
    parser.add_argument('--distances', type=lambda s: [float(item) for item in s.split(',')],
                        default='10,4,2', help='Crystal face distances.')

    # Video
    parser.add_argument('--width', type=int, default=1200, help='Width of video in pixels.')
    parser.add_argument('--height', type=int, default=900, help='Height of video in pixels.')
    parser.add_argument('--n-steps', type=int, default=1000, help='Number of frames to generate.')
    parser.add_argument('--duration', type=int, default=10, help='Video duration in seconds.')
    parser.add_argument('--fps', type=int, default=25, help='Video framerate.')
    parser.add_argument('--spp', type=int, default=2048, help='Samples per pixel.')

    # 3D plot
    parser.add_argument('--n-revolutions', type=float, default=2,
                        help='Number of revolutions across the clip.')

    args = parser.parse_args()

    return args


def _generate_crystal(
        distances: List[float] = [1.0, 0.5, 0.2],
        origin: List[float] = [0, 0, 20],
        rotvec: List[float] = [0, 0, 0],
) -> Crystal:
    """
    Generate a beta-form LGA crystal.
    """
    reader = EntryReader()
    crystal = reader.crystal('LGLUAC01')
    miller_indices = [(1, 1, 1), (0, 1, 2), (0, 0, 2)]
    lattice_unit_cell = [crystal.cell_lengths[0], crystal.cell_lengths[1], crystal.cell_lengths[2]]
    lattice_angles = [crystal.cell_angles[0], crystal.cell_angles[1], crystal.cell_angles[2]]
    point_group_symbol = '222'  # crystal.spacegroup_symbol

    crystal = Crystal(
        lattice_unit_cell=lattice_unit_cell,
        lattice_angles=lattice_angles,
        miller_indices=miller_indices,
        point_group_symbol=point_group_symbol,
        distances=distances,
        origin=origin,
        rotvec=rotvec
    )
    crystal.to(device)

    return crystal


def build_mitsuba_mesh(crystal: Crystal) -> mi.Mesh:
    """
    Convert a Crystal object into a Mitsuba mesh.
    """
    # Build the mesh in pytorch and convert the parameters to Mitsuba format
    vertices, faces = crystal.build_mesh()
    nv, nf = len(vertices), len(faces)
    vertices = mi.TensorXf(vertices)
    faces = mi.TensorXi64(faces)

    # Set up the material properties
    bsdf = {
        'type': 'roughdielectric',
        'distribution': 'beckmann',
        'alpha': 0.02,
        'int_ior': 1.78,
    }
    props = mi.Properties()
    props[BSDF_KEY] = mi.load_dict(bsdf)

    # Construct the mitsuba mesh and set the vertex positions and faces
    mesh = mi.Mesh(
        SHAPE_NAME,
        vertex_count=nv,
        face_count=nf,
        has_vertex_normals=False,
        has_vertex_texcoords=False,
        props=props
    )
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['faces'] = dr.ravel(faces)

    return mesh


def create_scene(crystal: Crystal, spp: int = 256, width: int = 400, height: int = 400) -> mi.Scene:
    """
    Create a Mitsuba scene containing the given crystal.
    """
    scene = mi.load_dict({
        'type': 'scene',

        # Camera and rendering parameters
        'integrator': {
            'type': 'path',
            'max_depth': 128,
            'rr_depth': 10,
            # 'sppi': 0,
        },
        'sensor': {
            'type': 'perspective',
            'to_world': T.look_at(
                origin=[0, 0, 80],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'stratified',
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': width,
                'height': height,
                'filter': {'type': 'gaussian'},
                'sample_border': True,
            },
        },

        # Emitters
        'light': {
            'type': 'rectangle',
            'to_world': T.scale(50),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': 0.7
                }
            },
        },

        # Shapes
        SHAPE_NAME: build_mitsuba_mesh(crystal)
    })

    return scene


def prepare_plot(args: Namespace):
    """
    Prepare the scene etc.
    """
    crystal = _generate_crystal(args.distances)

    # Get the rotations
    rx = np.linspace(0, 1, args.n_steps)
    ry = np.linspace(0, 1, args.n_steps) **2
    rz = np.linspace(0, 1, args.n_steps)
    v = np.stack([rx, ry, rz]).T
    thetas = np.linspace(0, args.n_revolutions * 2 * np.pi, args.n_steps)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True) * thetas[:, None]

    def update(step: int):
        rotvec = torch.from_numpy(v[step]).to(device).to(torch.float32)
        q = axis_angle_to_quaternion(rotvec)
        crystal.rotvec.data = q.to(crystal.rotvec.device)
        scene = create_scene(crystal=crystal, spp=args.spp, width=args.width, height=args.height)
        img = mi.render(scene)
        img = np.array(mi.util.convert_to_bitmap(img))
        return img

    return update


def generate_video():
    """
    Generate a video from a spec file.
    """
    args = get_args()
    print_args(args)

    # Write the args to the output dir
    output_dir = LOGS_PATH / START_TIMESTAMP
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / 'args.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    # Set up the plot
    update_fn = prepare_plot(args)

    # Initialise ffmpeg process
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'rotating_crystal'
    input_args = {
        'format': 'rawvideo',
        'pix_fmt': 'rgb24',
        's': f'{args.width}x{args.height}',
        'r': args.n_steps / args.duration,
    }
    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': args.fps,
        'metadata:g:0': f'title=Synthetic crystal rotation',
        'metadata:g:1': 'artist=Shape4PPD',
        'metadata:g:2': f'year={time.strftime("%Y")}',
    }
    process = (
        ffmpeg
        .input('pipe:', **input_args)
        .output(str(output_path) + '.mp4', **output_args)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    logger.info('Rendering steps.')
    for step in range(args.n_steps):
        if step > 0 and (step + 1) % 10 == 0:
            logger.info(f'Rendering step {step + 1}/{args.n_steps}.')

        # Update the frame and write to stream
        frame = update_fn(step)
        process.stdin.write(frame.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()

    logger.info(f'Generated video.')


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    generate_video()
