import os
import time
from argparse import ArgumentParser, Namespace

import cv2
import ffmpeg
import numpy as np
import torch
import yaml
from ccdc.io import EntryReader
from mayavi import mlab

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.util.utils import print_args, to_dict, to_numpy, to_rgb

# Off-screen rendering
mlab.options.offscreen = True


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to generate a video of a digital crystal growing.')

    # Video
    parser.add_argument('--width', type=int, default=1000, help='Width of video in pixels.')
    parser.add_argument('--height', type=int, default=800, help='Height of video in pixels.')
    parser.add_argument('--n-steps', type=int, default=500, help='Number of frames to generate.')
    parser.add_argument('--duration', type=int, default=15, help='Video duration in seconds.')
    parser.add_argument('--fps', type=int, default=25, help='Video framerate.')

    # 3D plot
    parser.add_argument('--revolutions', type=float, default=2,
                        help='Number of plot revolutions across the clip.')
    parser.add_argument('--distance', type=float, default=14.,
                        help='Camera distance.')
    parser.add_argument('--surface-colour', type=str, default='skyblue',
                        help='Mesh surface colour.')
    parser.add_argument('--wireframe-colour', type=str, default='darkblue',
                        help='Mesh wireframe colour.')

    # Morphology
    parser.add_argument('--distances-start', type=lambda s: [float(item) for item in s.split(',')],
                        default='1,0.1,0.05', help='Starting face distances.')
    parser.add_argument('--distances-end', type=lambda s: [float(item) for item in s.split(',')],
                        default='2,1.3,0.4', help='Ending face distances.')

    args = parser.parse_args()
    assert len(args.distances_start) == len(args.distances_end) == 3, 'Must provide 3 distances for start and end.'

    return args


def _generate_crystal() -> Crystal:
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
    )

    return crystal


def prepare_plot(args: Namespace):
    """
    Prepare the 3D figure.
    """
    azim_offset = 0
    elev_offset = 0
    wireframe_radius_factor = 0.01

    # Make crystal
    crystal = _generate_crystal()

    # Setup distance values
    distances_start = torch.tensor(args.distances_start)
    distances_end = torch.tensor(args.distances_end)
    distances = torch.linspace(0, 1, args.n_steps)[:, None] * (distances_end - distances_start) + distances_start

    # Set up mlab figure
    bg_col = 250 / 255
    fig = mlab.figure(size=(args.width * 2, args.height * 2), bgcolor=(bg_col, bg_col, bg_col))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Add initial crystal mesh
    v, f = crystal.build_mesh(distances=distances[0])
    v, f = to_numpy(v), to_numpy(f)
    mesh = mlab.triangular_mesh(*v.T, f, figure=fig, color=to_rgb(args.surface_colour), opacity=0.7)
    lines = []
    for fv_idxs in crystal.faces.values():
        fv = to_numpy(crystal.vertices[fv_idxs])
        fv = np.vstack([fv, fv[0]])  # Close the loop
        l = mlab.plot3d(*fv.T, color=to_rgb(args.wireframe_colour),
                        tube_radius=distances[0, 0].item() * wireframe_radius_factor)
        lines.append(l)

    # Aspects
    azims = azim_offset + np.linspace(start=0, stop=360 * args.revolutions, num=args.n_steps)
    elevs = elev_offset + np.fmod(np.linspace(start=0, stop=180 * args.revolutions / 2, num=args.n_steps), 180)
    mlab.view(figure=fig, azimuth=azims[0], elevation=elevs[0], distance=args.distance, focalpoint=np.zeros(3))

    def update(step: int):
        fig.scene.disable_render = True
        d = distances[step]
        v, f = crystal.build_mesh(distances=d)
        v, f = to_numpy(v).T, to_numpy(f)
        mesh.mlab_source.set(x=v[0], y=v[1], z=v[2], triangles=f)
        for i, fv_idxs in enumerate(crystal.faces.values()):
            fv = to_numpy(crystal.vertices[fv_idxs])
            fv = np.vstack([fv, fv[0]])  # Close the loop
            lines[i].mlab_source.set(points=fv, tube_radius=d[0].item() * wireframe_radius_factor)
        fig.scene.disable_render = False
        mlab.view(
            figure=fig,
            azimuth=azims[step],
            elevation=elevs[step],
            reset_roll=False,
            distance=args.distance,
            focalpoint=np.zeros(3)
        )
        fig.scene.render()
        frame = mlab.screenshot(mode='rgb', antialiased=True, figure=fig)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return frame

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
    output_path = output_dir / f'growth_vid'
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
        'metadata:g:0': f'title=Synthetic crystal evolution',
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
