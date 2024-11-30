import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from mayavi import mlab
from tvtk.tools import visual
import cv2  # must come after mayavi

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.scene_components.bubble import make_bubbles
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.util.utils import print_args, set_seed, to_dict, to_numpy, to_rgb

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
    parser.add_argument('--scene-path', type=Path, help='Path to a scene yml file.')
    parser.add_argument('--miller-indices', type=lambda s: [tuple(map(int, item.split(','))) for item in s.split(';')],
                        default='1,1,1;0,1,2;0,0,2', help='Miller indices of the crystal faces.')
    parser.add_argument('--distances', type=lambda s: [float(item) for item in s.split(',')],
                        default='10,3,1.3', help='Crystal face distances.')

    # 3D plot
    parser.add_argument('--res', type=int, default=2000,
                        help='Width and height of images in pixels.')
    parser.add_argument('--azim', type=float, default=30,
                        help='Azimuthal angle of the camera.')
    parser.add_argument('--elev', type=float, default=72,
                        help='Elevation angle of the camera.')
    parser.add_argument('--roll', type=float, default=-100,
                        help='Roll angle of the camera.')
    parser.add_argument('--distance', type=float, default=35,
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


def _init() -> Tuple[Namespace, Scene, Path]:
    """
    Initialise the scene and get the command line arguments.
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

    # Initialise the scene
    if args.scene_path is not None:
        scene = Scene.from_yml(args.scene_path)
    else:
        scene = _generate_scene(args)

    return args, scene, output_dir


def _generate_scene(
        args: Namespace,
        origin: List[float] = [0, 0, 0.1],
        rotvec: List[float] = [0, 0, 0],
) -> Scene:
    """
    Generate a basic scene with crystal in it.
    """
    csd_proxy = CSDProxy()
    cs = csd_proxy.load('LGLUAC11')
    miller_indices = [(1, 0, 1), (0, 2, 1), (0, 1, 0)]

    crystal = Crystal(
        lattice_unit_cell=cs.lattice_unit_cell,
        lattice_angles=cs.lattice_angles,
        miller_indices=miller_indices,
        point_group_symbol=cs.point_group_symbol,
        distances=args.distances,
        origin=origin,
        rotation=rotvec
    )

    # Build the scene
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
    )

    return scene


def _add_circle(
        x: float,
        y: float,
        z: float,
        radius: float,
        colour: Tuple[float, float, float] = (1.0, 1.0, 1.0)
):
    """
    Adds a circle to the 3D scene.
    """
    theta = np.linspace(0, 2 * np.pi, 40)
    r = np.linspace(0, radius, 2)

    # Create meshgrid for polar coordinates
    r, theta = np.meshgrid(r, theta)

    # Convert polar to Cartesian coordinates
    circle_x_vals = r * np.cos(theta) + x
    circle_y_vals = r * np.sin(theta) + y
    circle_z_vals = np.full_like(circle_x_vals, z)

    # Plot filled circle with mlab.mesh
    return mlab.mesh(circle_x_vals, circle_y_vals, circle_z_vals, color=colour)


def _add_camera(
        z_cam: float,
        cylinder_length: float = 2.0,
        cylinder_radius: float = 0.5,
        cone_length: float = 1.0,
        cone_radius: float = 0.5,
        cylinder_colour: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        cone_colour: Tuple[float, float, float] = (0.3, 0.3, 0.3)
):
    """
    Adds a camera-like shape consisting of a cylinder and cone to the 3D scene.

    Parameters:
    - z_cam (float): The z-coordinate for the camera's position.
    - cylinder_length (float): Length of the cylinder representing the camera body.
    - cylinder_radius (float): Radius of the cylinder representing the camera body.
    - cone_length (float): Length of the cone representing the camera lens.
    - cone_radius (float): Base radius of the cone representing the camera lens.
    """
    # Define cylinder parameters (camera body)
    cylinder_z = np.linspace(0, -cylinder_length, 20)  # Length of cylinder
    theta = np.linspace(0, 2 * np.pi, 40)
    theta, z = np.meshgrid(theta, cylinder_z)
    x = cylinder_radius * np.cos(theta)
    y = cylinder_radius * np.sin(theta)

    # Move cylinder to (0, 0, z_cam)
    z += z_cam

    # Plot the cylinder and cap it
    mlab.mesh(x, y, z, color=cylinder_colour, opacity=1.0)
    _add_circle(x=0, y=0, z=z_cam, radius=cylinder_radius, colour=cylinder_colour)

    # Define cone parameters (camera lens)
    cone_z = np.linspace(-cylinder_length, -cylinder_length - cone_length, 20)
    cone_radius_values = np.linspace(cylinder_radius, cone_radius, 20)  # Radius decreases to form the cone
    theta, cone_z = np.meshgrid(theta, cone_z)
    x = cone_radius_values[:, None] * np.cos(theta)
    y = cone_radius_values[:, None] * np.sin(theta)

    # Move cone to end of cylinder
    cone_z += z_cam

    # Plot the cone
    mlab.mesh(x, y, cone_z, color=cone_colour, opacity=1.0)


def _add_led_light(position: Tuple[float, float, float], square_size: float = 1.0, depth: float = 0.5,
                   inset_border: float = 0.1, inset_depth: float = 0.2, num_leds: int = 4,
                   led_colour: Tuple[float, float, float] = (1.0, 1.0, 1.0), arrow_height: float = 1.0) -> None:
    """
    Adds an LED-like shape, consisting of a solid cuboid with a carved inset on top, a grid of LED bulbs,
    and arrows to represent light emission.

    Parameters:
    - position (Tuple[float, float, float]): The (x, y, z) position of the center of the LED square.
    - square_size (float): The size of the LED square.
    - depth (float): Depth of the main cuboid.
    - inset_border (float): Border thickness for the inset carve-out.
    - inset_depth (float): Depth of the inset carved into the top of the cuboid.
    - num_leds (int): Number of circles in each row and column representing LEDs (total is num_leds^2).
    - led_colour (Tuple[float, float, float]): Colour of the LED bulbs.
    - inset_colour (Tuple[float, float, float]): Colour of the inset area.
    - arrow_height (float): The height of the arrows to represent emitted light.
    """

    # Create the main cuboid mesh with Trimesh
    main_cuboid = trimesh.creation.box(extents=[square_size, square_size, depth])
    main_cuboid.apply_translation([0, 0, -depth / 2])

    # Create the inset cuboid with Trimesh
    inset_size = square_size - 2 * inset_border
    inset_cuboid = trimesh.creation.box(extents=[inset_size, inset_size, inset_depth])
    inset_cuboid.apply_translation([0, 0, -inset_depth / 2])

    # Subtract inset from the main cuboid
    combined_cuboid = main_cuboid.difference(inset_cuboid)

    # Translate the combined cuboid to the given position
    combined_cuboid.apply_translation(position)

    # Convert the Trimesh mesh to a format that Mayavi can use
    vertices = combined_cuboid.vertices
    faces = combined_cuboid.faces

    # Extract x, y, z from vertices for Mayavi plotting
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # Plot the combined cuboid in Mayavi
    scene = mlab.gcf().scene
    mlab.triangular_mesh(x, y, z, faces, color=(0.3, 0.3, 0.3), opacity=1.0)

    # Add another surface at the base of the inset
    x = np.array([inset_size / 2, -inset_size / 2, -inset_size / 2, inset_size / 2])
    y = np.array([inset_size / 2, inset_size / 2, -inset_size / 2, -inset_size / 2])
    z = np.ones_like(x) * (-inset_depth + position[2] + 0.01)
    triangles = [[0, 1, 2], [0, 2, 3]]
    mlab.triangular_mesh(x, y, z, triangles, color=(0.8, 0.8, 0.8), opacity=1.0)

    # Plot grid of circles to represent LED components (inside the carved inset)
    led_radius = inset_size / (3 * num_leds)  # Proportional size for LED components
    spacing = inset_size / (num_leds + 1)
    for i in range(num_leds):
        for j in range(num_leds):
            x = position[0] - inset_size / 2 + spacing * (i + 1)
            y = position[1] - inset_size / 2 + spacing * (j + 1)
            z = -inset_depth + position[2] + 0.02

            # Add the bulb
            m = _add_circle(x=x, y=y, z=z, radius=led_radius, colour=(1., 1., 1.))

            # Access and adjust the mesh's actor properties to enhance brightness
            m.actor.property.specular = 1  # Increase specular reflectivity for a shiny look
            m.actor.property.specular_power = 50  # Control the sharpness of the shine (higher = sharper)
            m.actor.property.ambient = 0.9  # Increase ambient lighting contribution
            m.actor.property.diffuse = 1  # Control how much diffuse light the surface reflects
            # m.actor.property.opacity = 1.0  # Set to fully opaque
            m.actor.property.lighting = False

            arrows = mlab.quiver3d(x, y, z, 0, 0, arrow_height,
                                   color=led_colour, scale_factor=1, mode='arrow', opacity=1)
            # arrows.actor.property.specular = 1  # Increase specular reflectivity for a shiny look
            # arrows.actor.property.specular_power = 50  # Control the sharpness of the shine (higher = sharper)
            # arrows.actor.property.ambient = 0.9  # Increase ambient lighting contribution
            # arrows.actor.property.diffuse = 1  # Control how much diffuse light the surface reflects


def _add_bubbles(
        n_bubbles: int,
        cell_size: float,
        min_z: float,
        max_z: float,
        min_scale: float = 0.05,
        max_scale: float = 0.2
):
    """
    Add a number of random bubbles to the 3D scene.
    """
    if n_bubbles == 0:
        return

    bubbles = make_bubbles(
        n_bubbles=n_bubbles,
        min_x=-cell_size / 2,
        max_x=cell_size / 2,
        min_y=-cell_size / 2,
        max_y=cell_size / 2,
        min_z=min_z,
        max_z=max_z,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    for bubble in bubbles:
        vertices, faces = to_numpy(bubble.vertices), to_numpy(bubble.faces)

        # Extract x, y, z from vertices for Mayavi plotting
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        # Plot the bubble in Mayavi
        mlab.triangular_mesh(x, y, z, faces, color=(0.5, 0.5, 0.5), opacity=0.5)


def _add_plane(plane_size: float, position: Tuple[float, float, float], normal: Tuple[float, float, float],
               color: Tuple[float, float, float] = (0.5, 0.5, 0.5), opacity: float = 0.2) -> None:
    """
    Adds a plane to the 3D scene.

    Parameters:
    - plane_size (float): Length of the edges of the plane.
    - position (Tuple[float, float, float]): The position of the centre of the plane (x, y, z).
    - normal (Tuple[float, float, float]): The normal vector defining the orientation of the plane.
    - color (Tuple[float, float, float]): RGB values for the color of the plane.
    - opacity (float): Opacity of the plane.
    """
    # Create a square in the XY plane centered at origin
    x = np.array([plane_size / 2, -plane_size / 2, -plane_size / 2, plane_size / 2])
    y = np.array([plane_size / 2, plane_size / 2, -plane_size / 2, -plane_size / 2])
    z = np.zeros_like(x)
    triangles = [[0, 1, 2], [0, 2, 3]]

    # Define plane points
    points = np.vstack((x, y, z))

    # Normal vector of the plane
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

    # Reference normal vector (XY plane points upwards in Z direction)
    reference_normal = np.array([0, 0, 1])

    # Calculate rotation axis and angle using cross product and dot product
    rotation_axis = np.cross(reference_normal, normal)
    angle = np.arccos(np.clip(np.dot(reference_normal, normal), -1.0, 1.0))

    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Create rotation matrix using Rodrigues' rotation formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])

        rotation_matrix = (np.eye(3) +
                           np.sin(angle) * K +
                           (1 - np.cos(angle)) * np.dot(K, K))
    else:
        # If the normal is aligned, use identity rotation matrix
        rotation_matrix = np.eye(3)

    # Apply rotation to points
    rotated_points = rotation_matrix @ points

    # Translate plane to the specified position
    rotated_points[0, :] += position[0]
    rotated_points[1, :] += position[1]
    rotated_points[2, :] += position[2]

    # Plot the plane using the rotated points
    mlab.triangular_mesh(rotated_points[0, :], rotated_points[1, :], rotated_points[2, :],
                         triangles, color=color, opacity=opacity)


def _add_arrow(
        origin=(0, 0, 0),
        dest=(1, 0, 0),
        c=None
):
    x1, y1, z1 = origin
    x2, y2, z2 = dest
    ar1 = visual.arrow(x=x1, y=y1, z=z1, color=c, radius_shaft=0.01, radius_cone=0.05)
    arrow_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    ar1.actor.scale = [arrow_length, arrow_length, arrow_length]
    ar1.pos = ar1.pos / arrow_length
    ar1.length_cone = 0.1 / arrow_length
    ar1.radius_cone = 0.05 / arrow_length
    ar1.radius_shaft = 0.01 / arrow_length
    ar1.axis = [x2 - x1, y2 - y1, z2 - z1]
    return ar1


def draw_scene_schematic():
    """
    Draw a 3D schematic of the 3D rendering scene
    """
    args, scene, output_dir = _init()
    crystal = scene.crystal

    # Set up mlab figure
    wireframe_radius_factor = 0.05
    crystal_scale_factor = 1.2
    bg_col = 250 / 255
    cell_size = 9
    light_size = 8
    fig = mlab.figure(size=(args.res * 2, args.res * 2), bgcolor=(bg_col, bg_col, bg_col))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20

    # Add the cell surface
    x = np.array([cell_size / 2, -cell_size / 2, -cell_size / 2, cell_size / 2])
    y = np.array([cell_size / 2, cell_size / 2, -cell_size / 2, -cell_size / 2])
    z = np.zeros_like(x)
    triangles = [[0, 1, 2], [0, 2, 3]]
    mlab.triangular_mesh(x, y, z, triangles, color=(0.1, 0.1, 0.1), opacity=0.2)

    # Add camera
    cam_l = scene.camera_distance / 10
    _add_camera(
        z_cam=scene.camera_distance / 4,
        cylinder_length=cam_l / 2,
        cylinder_radius=cam_l / 6,
        cone_length=cam_l / 4,
        cone_radius=cam_l / 4
    )

    # Add the light
    _add_led_light(
        position=(0, 0, -2),
        square_size=light_size,
        depth=0.5,
        inset_border=0.2,
        inset_depth=0.3,
        num_leds=3,
        arrow_height=1
    )

    # Add crystal
    crystal.origin.data = torch.zeros_like(crystal.origin)
    crystal.scale.data = crystal_scale_factor * crystal.scale
    v, f = crystal.build_mesh()
    z_min = v[:, 2].amin(dim=-1)
    v[:, 2] -= z_min
    crystal.vertices.data[:, 2] -= z_min
    v, f = to_numpy(v), to_numpy(f)
    mlab.triangular_mesh(*v.T, f, figure=fig, color=to_rgb(args.surface_colour), opacity=0.9)
    for fv_idxs in crystal.faces.values():
        fv = to_numpy(crystal.vertices[fv_idxs])
        fv = np.vstack([fv, fv[0]])  # Close the loop
        mlab.plot3d(*fv.T, color=to_rgb(args.wireframe_colour),
                    tube_radius=crystal.distances[0].item() * wireframe_radius_factor)

    # Add some bubbles
    _add_bubbles(
        n_bubbles=10,
        cell_size=cell_size,
        min_z=1,
        max_z=scene.camera_distance / 4,
        min_scale=0.05,
        max_scale=0.2
    )

    # Render
    mlab.view(
        figure=fig,
        azimuth=30,
        elevation=72,
        distance=35,
        roll=-100
    )
    mlab.view(figure=fig, azimuth=args.azim, elevation=args.elev, distance=args.distance, roll=args.roll)
    frame = mlab.screenshot(mode='rgba', antialiased=True, figure=fig)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = Image.fromarray((frame * 255).astype(np.uint8), 'RGBA')
    img.save(output_dir / 'scene.png')

    # Useful for getting the view parameters when recording from the gui:
    # mlab.show()
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [28.9235516170804, 17.661483497882795, 14.249465607736322]
    # scene.scene.camera.focal_point = [0.13542003054347784, 0.07598590140548733, 3.5921460765424675]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [-0.26524733430646313, -0.1435855510994738, 0.95342909603115]
    # scene.scene.camera.clipping_range = [19.339086742284792, 55.652440215714165]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()


def draw_crystal_schematic():
    """
    Draw a 3D schematic of the 3D crystal builder
    """
    args, scene, output_dir = _init()
    crystal = scene.crystal

    # Set up mlab figure
    wireframe_radius_factor = 0.01
    wireframe_radius_factor2 = 0.04
    bg_col = 250 / 255
    plane_size = 3
    fig = mlab.figure(size=(args.res * 2, args.res * 2), bgcolor=(bg_col, bg_col, bg_col))

    # Depth peeling required for nice opacity, the rest don't seem to make any difference
    fig.scene.renderer.use_depth_peeling = True
    fig.scene.renderer.maximum_number_of_peels = 32
    fig.scene.render_window.point_smoothing = True
    fig.scene.render_window.line_smoothing = True
    fig.scene.render_window.polygon_smoothing = True
    fig.scene.render_window.multi_samples = 20
    fig.scene.anti_aliasing_frames = 20
    visual.set_viewer(fig)

    # Add crystal
    origin0 = crystal.mesh_vertices.mean(dim=0)
    crystal.adjust_origin(origin0)
    crystal.rotation.data = torch.zeros_like(crystal.rotation)
    v, f = crystal.build_mesh()
    z_min = v[:, 2].amin(dim=-1)
    v[:, 2] -= z_min
    crystal.vertices.data[:, 2] -= z_min
    colours = [np.array(to_rgb(c)) for c in ['red', 'green', 'blue']]
    vertex_dict = {}
    tolerance = 0.1

    # Process each plane and its vertices
    for i, fv_idxs in enumerate(crystal.faces.values()):
        if i > 2:
            break
        fvi = to_numpy(crystal.vertices[fv_idxs])

        for vertex in fvi:
            vertex_tuple = tuple(vertex)

            # Check if the vertex already exists in the dictionary
            found = False
            for key in list(vertex_dict.keys()):
                if np.linalg.norm(np.array(key) - vertex) < tolerance:
                    # If found, update color contribution
                    vertex_dict[key]['colour'] += colours[i]
                    vertex_dict[key]['count'] += 1
                    found = True
                    break

            if not found:
                # If not found, add a new entry in the dictionary
                vertex_dict[vertex_tuple] = {'colour': colours[i].copy(), 'count': 1}

    # Plot each vertex, adjusting color by averaging and size if it is the intersection
    for vertex, data in vertex_dict.items():
        average_color = tuple((data['colour'] / data['count']).tolist())
        scale_factor = 0.2 if data['count'] == 3 else 0.1  # Make intersection point larger
        p3d = mlab.points3d(*vertex, color=average_color, scale_factor=scale_factor)

        p3d.actor.property.specular = 0.6  # Increase specular reflectivity for a shiny look
        p3d.actor.property.specular_power = 25  # Control the sharpness of the shine (higher = sharper)
        p3d.actor.property.ambient = 0.9  # Increase ambient lighting contribution
        p3d.actor.property.diffuse = 1  # Control how much diffuse light the surface reflects
        p3d.actor.property.opacity = 1.0  # Set to fully opaque
        p3d.actor.property.lighting = False

    edge_dict = {}
    for i, fv_idxs in enumerate(crystal.faces.values()):
        fv_og = to_numpy(crystal.vertices[fv_idxs])
        fv = np.vstack([fv_og, fv_og[0]])  # Close the loop
        if i < 3:
            hkl = ','.join(str(hkl_) for hkl_ in crystal.all_miller_indices[i].tolist())
            logger.info(f'Drawing face {hkl}')
            for v0, v1 in zip(fv[:-1], fv[1:]):
                v0k = tuple(v0)
                v1k = tuple(v1)
                ek = (v0k, v1k)
                ek2 = (v1k, v0k)
                if ek in edge_dict or ek2 in edge_dict:
                    continue
                count_start = vertex_dict[v0k]['count']
                count_end = vertex_dict[v1k]['count']
                if count_start == 1 or count_end == 1:
                    colour = tuple(colours[i].tolist())
                elif count_start == 2 or count_end == 2:
                    colour = tuple((vertex_dict[v0k if count_start == 2 else v1k]['colour'] / 2).tolist())
                mlab.plot3d(*np.vstack([v0, v1]).T, color=colour,
                            tube_radius=crystal.distances[0].item() * wireframe_radius_factor2)
                edge_dict[ek] = True
                edge_dict[ek2] = True

            colour = tuple(colours[i].tolist())
            centroid = np.mean(fv, axis=0)
            fv = np.vstack((fv, centroid))
            centroid_idx = fv.shape[0] - 1
            triangles = [[j, (j + 1) % len(fv), centroid_idx] for j in range(len(fv))]
            mlab.triangular_mesh(*fv.T, triangles, color=colour, opacity=0.6)

            # Add arrow from the origin to the plane
            origin = to_numpy(crystal.origin)
            dest = to_numpy(crystal.origin + crystal.distances[i] * crystal.N[i])
            _add_arrow(origin=origin, dest=dest, c=colour)

            # Add plane
            _add_plane(
                plane_size=plane_size,
                position=dest,
                normal=crystal.N[i],
                color=colour,
                opacity=0.25
            )

        else:
            mlab.plot3d(*fv.T, color=to_rgb(args.wireframe_colour),
                        tube_radius=crystal.distances[0].item() * wireframe_radius_factor)

    # Render
    mlab.view(
        figure=fig,
        azimuth=60,
        elevation=70,
        distance=13,
        roll=-120
    )
    frame = mlab.screenshot(mode='rgba', antialiased=True, figure=fig)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = Image.fromarray((frame * 255).astype(np.uint8), 'RGBA')
    img.save(output_dir / 'crystal.png')

    # Useful for getting the view parameters when recording from the gui:
    # mlab.show()
    # scene = mlab.get_engine().scenes[0]
    # scene.scene.camera.position = [6.025470076080877, 10.659418476829046, 5.574685103285127]
    # scene.scene.camera.focal_point = [0.195185495665805, 0.1472335023203768, 0.6247609853744506]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [-0.23259960958193596, -0.3057383766330435, 0.9232667364722518]
    # scene.scene.camera.clipping_range = [7.357338693403784, 20.134304810563314]
    # scene.scene.camera.compute_view_plane_normal()
    # scene.scene.render()
    # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
    # print(mlab.roll())
    # exit()


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    # draw_scene_schematic()
    draw_crystal_schematic()
