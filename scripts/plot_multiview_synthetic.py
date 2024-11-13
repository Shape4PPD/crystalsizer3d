import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import cv2
import mitsuba as mi
import numpy as np
import torch
import yaml
from PIL import Image
from mayavi import mlab
from scipy.spatial.transform import Rotation as R

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.util.utils import init_tensor, print_args, set_seed, to_dict, to_numpy, to_rgb

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('llvm_ad_rgb')
    device = torch.device('cpu')

# Off-screen rendering
mlab.options.offscreen = True


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to generate a video of a digital crystal growing.')
    parser.add_argument('--seed', type=int, default=2,
                        help='Seed for the random number generator.')

    # Crystal
    parser.add_argument('--crystal-path', type=Path, help='Path to a crystal json file.')
    parser.add_argument('--miller-indices', type=lambda s: [tuple(map(int, item.split(','))) for item in s.split(';')],
                        default='1,1,1;0,1,2;0,0,2', help='Miller indices of the crystal faces.')
    parser.add_argument('--distances', type=lambda s: [float(item) for item in s.split(',')],
                        default='10,3,1.3', help='Crystal face distances.')

    # Images
    parser.add_argument('--res', type=int, default=100, help='Width and height of images in pixels.')
    parser.add_argument('--spp', type=int, default=10, help='Samples per pass.')

    # 3D plot
    parser.add_argument('--azim', type=float, default=10,
                        help='Azimuthal angle of the camera.')
    parser.add_argument('--elev', type=float, default=65,
                        help='Elevation angle of the camera.')
    parser.add_argument('--roll', type=float, default=25,
                        help='Roll angle of the camera.')
    parser.add_argument('--distance', type=float, default=6,
                        help='Camera distance.')
    parser.add_argument('--surface-colour', type=str, default='skyblue',
                        help='Mesh surface colour.')
    parser.add_argument('--wireframe-colour', type=str, default='darkblue',
                        help='Mesh wireframe colour.')

    # Perspective renderings
    parser.add_argument('--n-rotations-per-axis', type=int, default=3,
                        help='Number of rotation increments to make per axis for the systematic rotation.')
    parser.add_argument('--n-frames', type=int, default=200,
                        help='Number of frames for the random rotation video.')
    parser.add_argument('--max-acc-change', type=float, default=0.01,
                        help='Maximum change in acceleration for the random rotation video.')
    parser.add_argument('--roughness', type=float, default=None,
                        help='Override the roughness of the crystal material.')

    args = parser.parse_args()

    # Set the random seed
    set_seed(args.seed)

    return args


def _generate_crystal(
        distances: List[float] = [1.0, 0.5, 0.2],
        origin: List[float] = [0, 0, 20],
        rotvec: List[float] = [0, 0, 0],
) -> Crystal:
    """
    Generate a beta-form LGA crystal.
    """
    csd_proxy = CSDProxy()
    cs = csd_proxy.load('LGLUAC11')
    miller_indices = [(1, 0, 1), (0, 2, 1), (0, 1, 0)]

    crystal = Crystal(
        lattice_unit_cell=cs.lattice_unit_cell,
        lattice_angles=cs.lattice_angles,
        miller_indices=miller_indices,
        point_group_symbol=cs.point_group_symbol,
        distances=distances,
        origin=origin,
        rotation=rotvec
    )
    crystal.to(device)

    return crystal


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
    # scene.scene.camera.position = [1.3763545093352523, -0.7546275653319254, -5.842831976214458]
    # scene.scene.camera.focal_point = [0.0, 0.0, 0.0]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [-0.9297293909126958, 0.26709182520570524, -0.253505851177823]
    # scene.scene.camera.clipping_range = [4.685894945101291, 7.845739989907185]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()

    # mlab.show()
    # exit()

    # fig.scene.render()
    frame = mlab.screenshot(mode='rgba', antialiased=True, figure=fig)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = Image.fromarray((frame * 255).astype(np.uint8), 'RGBA')
    img.save(output_dir / 'digital.png')


def _plot_perspectives(
        crystal: Crystal,
        args: Namespace,
        output_dir: Path
):
    """
    Render different perspectives of the crystal.
    """
    k = torch.linspace(0, 1, args.n_rotations_per_axis + 1)[:-1] * np.pi

    scene = Scene(
        crystal=crystal,
        spp=args.spp,
        res=args.res,
        integrator_max_depth=32,
        integrator_rr_depth=10,
        camera_distance=32,
        focus_distance=30,
        camera_fov=10.2,
        aperture_radius=0.3,
        light_z_position=-5.1,
        light_scale=8.0,
        light_radiance=(0.96, 0.96, 0.94),
    )

    for kx in k:
        for ky in k:
            for kz in k:
                logger.info(f'Rendering perspective [{kx:.2f}, {ky:.2f}, {kz:.2f}]')
                rotvec = torch.tensor([kx, ky, kz], device=device)
                crystal.rotation.data = rotvec
                scene.build_mi_scene()
                img = scene.render()
                cv2.imwrite(str(output_dir / f'perspective_{kx:.2f}_{ky:.2f}_{kz:.2f}.png'), img)


def _plot_random_rotation(
        crystal: Crystal,
        args: Namespace,
        output_dir: Path,
        make_video: bool = False
):
    """
    Render different perspectives of the crystal.
    """
    r0 = np.zeros((1, 3))
    momentum = np.random.uniform(-args.max_acc_change, args.max_acc_change, size=(args.n_frames - 1, 3))
    components = np.cumsum(np.concatenate([r0, np.cumsum(momentum, axis=0)]), axis=0)
    rotations = R.from_euler('xyz', components).as_rotvec()

    # light_st_texture = NoiseTexture(
    #     dim=args.res * 3,
    #     channels=3,
    #     perlin_freq=0.5,
    #     perlin_octaves=9,
    #     white_noise_scale=0.2,
    #     max_amplitude=0.1,
    #     zero_centred=True,
    #     shift=1.,
    #     seed=args.seed
    # )
    if args.roughness is not None:
        crystal.material_roughness.data.fill_(args.roughness)

    scene = Scene(
        crystal=crystal,
        spp=args.spp,
        res=args.res,
        integrator_max_depth=32,
        integrator_rr_depth=10,
        camera_distance=32,
        focus_distance=30,
        camera_fov=10.2,
        aperture_radius=0.3,
        light_z_position=-5.1,
        light_scale=8.0,
        light_radiance=(0.9, 0.9, 0.9),
        # light_st_texture=light_st_texture,
    )

    for i, r in enumerate(rotations):
        if (i + 1) % 5 == 0:
            logger.info(f'Rendering frame {i + 1}/{args.n_frames}')
        crystal.rotation.data = init_tensor(r, device=device)
        scene.build_mi_scene()
        img = scene.render()
        cv2.imwrite(str(output_dir / f'frame_{i:04d}.png'), img)

    if make_video:
        save_path = output_dir / 'rotation.mp4'
        logger.info(f'Making rotation video to {save_path}.')
        escaped_images_dir = str(output_dir.absolute()).replace('[', '\\[').replace(']', '\\]')
        cmd = f'ffmpeg -y -framerate 25 -pattern_type glob -i "{escaped_images_dir}/frame_*.png" -c:v libx264 -pix_fmt yuv420p "{save_path}"'
        logger.info(f'Running command: {cmd}')
        os.system(cmd)


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
    if args.crystal_path is not None and args.crystal_path.exists():
        crystal = Crystal.from_json(args.crystal_path)
        crystal.to(device)
    else:
        crystal = _generate_crystal(args.distances)

    # # Make digital plot
    # _plot_digital_crystal(crystal, args, output_dir)
    # 
    # # Make perspective renderings
    # _plot_perspectives(crystal, args, output_dir)
    _plot_random_rotation(crystal, args, output_dir, make_video=True)


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    plot_views()
