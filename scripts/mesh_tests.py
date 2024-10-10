from crystalsizer3d.crystal import Crystal
import torch
from crystal_points import ProjectorPoints, plot_2d_projection
import numpy as np
import matplotlib.pyplot as plt
from crystalsizer3d.projector import Projector
import cv2
from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import project_to_image
from crystalsizer3d.util.utils import init_tensor, to_numpy

from scipy.spatial import cKDTree

device = torch.device('cuda')

TEST_CRYSTALS = {
    'cube': {
        'lattice_unit_cell': [1, 1, 1],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'point_group_symbol': '222',
        'scale': 1,
        'origin': [0.5, 0, 0],
        'distances': [1., 1., 1.],
        'rotation': [0.3, 0.3, 0.3],
        'material_ior': 1.2,
        'material_roughness': 1.5#0.01
    },
    'cube_test': {
        'lattice_unit_cell': [1, 1, 1],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'point_group_symbol': '222',
        'scale': 1,
        'origin': [0.5, 0, 0],
        'distances': [1.3, 1.0, 1.0],
        'rotation': [0.3, 0.3, 0.3],
        'material_ior': 1.2,
        'material_roughness': 1.5#0.01
    },
    'alpha6': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 0, 0), (1, 1, 0), (0, 0, -1), (0, -1, -1),
                           (0, 1, -1), (0, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1),
                           (1, 1, -1), (-1, 0, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'distances': [0.3830717206001282, 0.8166847825050354, 0.8026739358901978, 0.9758344292640686,
                      0.9103631377220154, 1.0181487798690796, 0.3933243453502655, 0.7772741913795471,
                      0.8740742802619934, 0.7110176682472229, 0.6107826828956604, 0.9051218032836914,
                      0.908871591091156, 1.1111396551132202, 0.9634890556335449, 0.9997269511222839,
                      1.1894351243972778, 0.9173557758331299, 1.2018373012542725, 1.1176774501800537],
        'origin': [-0.3571832776069641, -0.19568444788455963, 0.6160652711987495],
        'scale': 5.1607864066598905,
        'rotation': [-0.1091805174946785,-0.001362028531730175,1.4652847051620483],
        'material_ior': 1.7000342640124446,
        'material_roughness': 0.13993626928782799
    },
    'alpha': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 0, -1), (1, 1, 1), (1, 1, -1), (0, 1, 1), (0, 1, -1), (1, 0, 0)],
        'distances': [0.53, 0.50, 1.13, 1.04, 1.22, 1.00, 1.30],
        'rotation': [0.0, 0.3, 0.3],
        'point_group_symbol': '222',
        'scale': 3.0,
    },
    'alpha_test': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 0, -1), (1, 1, 1), (1, 1, -1), (0, 1, 1), (0, 1, -1), (1, 0, 0)],
        'distances': [0.6, 0.4, 1.0, 1.01, 1.28, 0.8, 1.00],
        'rotation': [0.0, 0.3, 0.3],
        'point_group_symbol': '222',
        'scale': 3.0,
    },
}

def create_closest_point_image(tensor_points, shape):
    """
    Creates an image where the closest point to each pixel from the tensor is colored green.

    Parameters:
    - tensor_points (torch.Tensor): An n x 2 tensor of points on the GPU, with the point coordinates 
                                    corresponding to (x, y) pixel positions.
    - image_height (int): The height of the image.
    - image_width (int): The width of the image.

    Returns:
    - image (np.ndarray): The generated image with green pixels for the closest points.
    """
    image_height, image_width = shape
    # Create a blank black image
    image = np.zeros((image_height, image_width, 4), dtype=np.uint8) #alpha

    # Convert tensor to CPU numpy array and round to nearest pixel
    points_np = tensor_points.cpu().detach().numpy().astype(int)

    # Ensure the points are within the bounds of the image
    points_np[:, 0] = np.clip(points_np[:, 0], 0, image_width - 1)
    points_np[:, 1] = np.clip(points_np[:, 1], 0, image_height - 1)

    for point in points_np:
        x, y = point
        image[y,x] = [0, 255, 0, 255] 
    # # Assign green color to pixels closest to any of the points

    return image

if __name__ == "__main__":
    
    res = 400
    crystal = Crystal(**TEST_CRYSTALS['alpha'])
    crystal.scale.data= init_tensor(1.2, device=crystal.scale.device)
    crystal.origin.data[:2] = torch.tensor([0, 0], device=crystal.origin.device)
    crystal.origin.data[2] -= crystal.vertices[:, 2].min()
    v, f = crystal.build_mesh()
    crystal.to(device)
    # m = Trimesh(vertices=to_numpy(v), faces=to_numpy(f))
    # m.show()

    # Create and render a scene
    scene = Scene(
        crystal=crystal,
        res=res,
        spp=512,

        camera_distance=32.,
        focus_distance=30.,
        # focal_length=29.27,
        camera_fov=10.2,
        aperture_radius=0.3,

        light_z_position=-5.1,
        # light_scale=5.,
        light_scale=10000.,
        light_radiance=.3,
        integrator_max_depth=3, # 2, default 64
        integrator_rr_depth=2, # default 5
        cell_z_positions=[-5, 0., 5., 10.],
        cell_surface_scale=3,
    )
    img = scene.render()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # todo: do we need this?

    # Get the unit scale factor
    z = crystal.vertices[:, 2].mean().item()
    _, (min_y, max_y) = scene.get_xy_bounds(z)
    zoom = 2 / (max_y - min_y)
    logger.info(f'Estimated zoom factor: {zoom:.3f}')
    pts2 = torch.tensor([[0, 1 / zoom, z], [0, -1 / zoom, z]], device=device)
    uv_pts2 = project_to_image(scene.mi_scene, pts2)  # these should appear at the top and bottom of the image

    # Save the original image with projected overlay
    projector = Projector(
        crystal=crystal,
        external_ior=1.333,
        image_size=(res, res),
        zoom=zoom,
        transparent_background=True,
        multi_line=True,
    )
    img_overlay = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    img_overlay[:, :, 3] = (img_overlay[:, :, 3] * 0.5).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(img)#,origin='lower')
    ax.imshow(img_overlay)#,origin='lower')
    
    
    # crystal = Crystal(**TEST_CRYSTALS['alpha'])
    # crystal.scale.data= init_tensor(1.2, device=crystal.scale.device)
    # crystal.origin.data[:2] = torch.tensor([0, 0], device=crystal.origin.device)
    # crystal.origin.data[2] -= crystal.vertices[:, 2].min()
    # v, f = crystal.build_mesh()
    # crystal.to(device)
    projector = ProjectorPoints(crystal,
                                external_ior=1.333,
                                zoom = zoom,
                                image_size=(res, res))
    points = projector.project()
    # point_image = create_closest_point_image(points.get_all_points_tensor(), (res, res))
    # point_image[:, :, 3] = (point_image[:, :, 3] * 0.5).astype(np.uint8)
    # ax.imshow(point_image)#,origin='lower')
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_2d_projection(points,ax=ax)
    plt.show()
    pass