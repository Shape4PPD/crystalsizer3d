import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import yaml
from ccdc.io import EntryReader
from kornia.geometry import axis_angle_to_quaternion
from matplotlib import pyplot as plt
from mayavi import mlab
import drjit as dr
import mitsuba as mi

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.util.utils import print_args, to_dict, to_numpy, to_rgb

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
    parser = ArgumentParser(description='CrystalSizer3D script to generate a video of a digital crystal growing.')

    # Crystal
    parser.add_argument('--distances', type=lambda s: [float(item) for item in s.split(',')],
                        default='10,3,1.3', help='Crystal face distances.')

    # Images
    parser.add_argument('--res', type=int, default=1000, help='Width and height of images in pixels.')
    parser.add_argument('--spp', type=int, default=2048, help='Samples per pixel.')

    # 3D plot
    parser.add_argument('--azim', type=float, default=150,
                        help='Azimuthal angle of the camera.')
    parser.add_argument('--elev', type=float, default=160,
                        help='Elevation angle of the camera.')
    parser.add_argument('--roll', type=float, default=0,
                        help='Roll angle of the camera.')
    parser.add_argument('--distance', type=float, default=55,
                        help='Camera distance.')
    parser.add_argument('--surface-colour', type=str, default='skyblue',
                        help='Mesh surface colour.')
    parser.add_argument('--wireframe-colour', type=str, default='darkblue',
                        help='Mesh wireframe colour.')

    # Perspective renderings
    parser.add_argument('--n-rotations-per-axis', type=int, default=6,
                        help='Number of rotations to make per axis.')

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
        rotation=rotvec
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


def create_scene(crystal: Crystal, spp: int = 256, res: int = 400) -> mi.Scene:
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
                'width': res,
                'height': res,
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


def _plot_digital_crystal(
        crystal: Crystal,
        args: Namespace,
        output_dir: Path
):
    """
    Make a 3D digital crystal plot.
    """
    wireframe_radius_factor = 0.01

    # Set up mlab figure
    bg_col = 250 / 255
    fig = mlab.figure(size=(args.res * 2, args.res * 2), bgcolor=(bg_col, bg_col, bg_col))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Add initial crystal mesh
    origin = crystal.origin.clone()
    crystal.origin.data = torch.zeros_like(origin)
    v, f = crystal.build_mesh()
    v, f = to_numpy(v), to_numpy(f)
    mlab.triangular_mesh(*v.T, f, figure=fig, color=to_rgb(args.surface_colour), opacity=0.7)
    for fv_idxs in crystal.faces.values():
        fv = to_numpy(crystal.vertices[fv_idxs])
        fv = np.vstack([fv, fv[0]])  # Close the loop
        mlab.plot3d(*fv.T, color=to_rgb(args.wireframe_colour),
                        tube_radius=crystal.distances[0].item() * wireframe_radius_factor)
    crystal.origin.data = origin

    # Render
    mlab.view(figure=fig, azimuth=args.azim, elevation=args.elev, distance=args.distance, roll=args.roll,
              focalpoint=np.zeros(3))

    # # Useful for getting the view parameters when recording from the gui:
    # mlab.show()
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [-1.0976718374293786, 0.5730634321110751, -4.126732879628852]
    # scene.scene.camera.focal_point = [0.0, 0.0, -1.862645149230957e-09]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [0.47541220253052663, -0.8452964994640355, -0.24383819569321213]
    # scene.scene.camera.clipping_range = [3.460626768194046, 5.386199822951965]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    # mlab.show()
    # exit()

    # fig.scene.render()
    frame = mlab.screenshot(mode='rgb', antialiased=True, figure=fig)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / 'digital.png'), frame)


def _plot_perspectives(
        crystal: Crystal,
        args: Namespace,
        output_dir: Path
):
    """
    Render different perspectives of the crystal.
    """
    k = torch.linspace(0, 1, args.n_rotations_per_axis+1)[:-1] * np.pi
    for kx in k:
        for ky in k:
            for kz in k:
                logger.info(f'Rendering perspective [{kx:.2f}, {ky:.2f}, {kz:.2f}]')
                rotvec = torch.tensor([kx, ky, kz])
                q = axis_angle_to_quaternion(rotvec)
                crystal.rotation.data = q.to(crystal.rotation.device)
                scene = create_scene(crystal=crystal, spp=args.spp, res=args.res)
                img = mi.render(scene)
                img = mi.util.convert_to_bitmap(img)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_dir / f'perspective_{kx:.2f}_{ky:.2f}_{kz:.2f}.png'), img)


def plot_views():
    """
    Generate a video from a spec file.
    """
    args = get_args()
    print_args(args)

    # Write the args to the output dir
    output_dir = LOGS_PATH / START_TIMESTAMP
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'args.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    # Make crystal
    crystal = _generate_crystal(args.distances)

    # Make digital plot
    _plot_digital_crystal(crystal, args, output_dir)

    # Make perspective renderings
    # _plot_perspectives(crystal, args, output_dir)


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    plot_views()
