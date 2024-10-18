import torch
from torch import nn 
import numpy as np
from pathlib import Path
from crystalsizer3d.crystal import Crystal
from crystalsizer3d import LOGS_PATH, ROOT_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.util.utils import print_args, to_numpy, init_tensor
import torch.nn.functional as F
import torchvision.transforms as transforms
from crystal_points import ProjectorPoints
from plot_mesh import multiplot, overlay_plot, plot_sampled_points_with_intensity
import torch.optim as optim
from crystalsizer3d.scene_components.scene import Scene
import cv2
from crystalsizer3d.scene_components.utils import project_to_image
from crystalsizer3d.projector import Projector
from edge_detection import load_rcf
from scipy.ndimage import distance_transform_edt
from torchvision.transforms.functional import to_tensor
from scipy.ndimage import gaussian_filter

from torch.utils.tensorboard import SummaryWriter
import io

# import matplotlib.pyplot as plt
from PIL import Image

if USE_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
from PIL import Image


class ContourDistanceNormalLoss(nn.Module):
    def __init__(self,
                 #distance_image,
        ):
        super(ContourDistanceNormalLoss,self).__init__()
        
    
    def forward(self,points,distance_image):
        
        N = len(points)  # Number of points
        
        # Initialize a tensor to accumulate the losses
        losses = []
        distances = []
        
        for i in range(N):
            # Extract the point and the normal vector for this iteration
            ref_point = points.get_point(i)
            
            line_points = self.sample_points_along_line_full_image(ref_point, 1000, distance_image.shape[-2:])
            
            
            # # sampled_values = self.sample_image_along_line(distance_image,line_points)
            reference_value = self.points_in_image(ref_point.point.unsqueeze(0), distance_image)
            
            nearest_minimum_posistion, nearest_minimum_value = self.find_nearest_local_minimum(distance_image, line_points, ref_point.point)
            if nearest_minimum_value == None:
                continue
            # get the distance to the closest peak
            distance = nearest_minimum_posistion - ref_point.point
            ref_point.set_distance(distance)
            # # Compute the mean squared error between the sampled values and the reference value
            loss = F.mse_loss(nearest_minimum_value.squeeze(), reference_value.squeeze())
            # loss = F.mse_loss(reference_point, reference_point)
            if torch.isnan(loss).any():
                print("Found NaN values in x!")
            losses.append(loss)
            
            # something is funnyt wioth how its getting the vallues from the image as x approches w and y approches h
        
        # Return the mean loss across all points
        return torch.mean(torch.stack(losses))
    
    def sample_points_along_line_full_image(self, ref_point, num_samples, image_shape):
        """
        Samples points along a line across the entire image, given a point and a normal vector.
        
        Args:
            point (torch.Tensor): A 2D tensor of shape (2,) representing the point on the line (x, y).
            normal_vector (torch.Tensor): A 2D tensor of shape (2,) representing the normal vector to the line.
            num_samples (int): Number of points to sample along the line.
            image_shape (tuple): A tuple representing the shape of the image (height, width).
            
        Returns:
            torch.Tensor: A tensor of sampled points along the line, of shape (num_samples, 2).
        """
        
        point = ref_point.point
        normal_vector = ref_point.normal.to(device)

        # # Compute the direction vector by rotating the normal vector by 90 degrees
        # direction_vector = torch.tensor([-normal_vector[1], normal_vector[0]],device=device)

        # Normalize the direction vector
        direction_vector_norm = normal_vector / torch.norm(normal_vector)
        # direction_vector_norm = direction_vector / torch.norm(direction_vector)

        # Find the intersection points with the image boundaries
        intersections = self.find_intersections_with_boundaries(point, direction_vector_norm, image_shape)

        # Sample points uniformly between the two intersection points
        sampled_points = torch.linspace(0, 1, steps=num_samples,device=device).unsqueeze(1) * (intersections[1] - intersections[0]) + intersections[0]

        # Clamp the points to ensure they are within image boundaries
        # sampled_points[:, 0] = torch.clamp(sampled_points[:, 0], 0, image_shape[1] - 1)  # x-coordinates
        # sampled_points[:, 1] = torch.clamp(sampled_points[:, 1], 0, image_shape[0] - 1)  # y-coordinates
        sampled_points = torch.stack([
            torch.clamp(sampled_points[:, 0], 0, image_shape[1] - 1),  # x-coordinates
            torch.clamp(sampled_points[:, 1], 0, image_shape[0] - 1)   # y-coordinates
        ], dim=1)

        return sampled_points
    
    def find_intersections_with_boundaries(self,point, direction_vector, image_shape):
        """
        Find the intersection points of the line with the image boundaries.

        Args:
            point (torch.Tensor): A 2D tensor of shape (2,) representing the point on the line (x, y).
            direction_vector (torch.Tensor): A 2D tensor of shape (2,) representing the direction vector of the line.
            image_shape (tuple): A tuple representing the shape of the image (height, width).
        
        Returns:
            list: A list of two points representing the intersection points with the image boundaries.
        """
        height, width = image_shape
        max_float = torch.finfo(direction_vector.dtype).max  # Maximum finite float value for the given dtype


        # Parametric line equation: p(t) = point + t * direction_vector
        # We need to find t such that p(t) lies on the image boundary

        # Handle intersections with left (x=0) and right (x=width-1) boundaries
        if torch.isclose(direction_vector[0], torch.tensor(0.0),atol=1e-3):
            t_left, t_right = -max_float, max_float
        else:
            t_left = (0 - point[0]) / direction_vector[0]  # Intersection with left boundary
            t_right = (width - 1 - point[0]) / direction_vector[0]  # Intersection with right boundary
        
        # Handle intersections with top (y=0) and bottom (y=height-1) boundaries
        if torch.isclose(direction_vector[1], torch.tensor(0.0),atol=1e-3):
            t_top, t_bottom = -max_float, max_float
        else:
            t_top = (0 - point[1]) / direction_vector[1]  # Intersection with top boundary
            t_bottom = (height - 1 - point[1]) / direction_vector[1]  # Intersection with bottom boundary

        # Take the maximum of left/top and the minimum of right/bottom
        t_min = max(t_left, t_top)
        t_max = min(t_right, t_bottom)

        # Compute the actual intersection points using the t values
        intersection_1 = point + t_min * direction_vector
        intersection_2 = point + t_max * direction_vector

        return torch.stack([intersection_1, intersection_2])
    
    def points_in_image(self,points,distance_image):
        N = len(points)  # Number of points
        H, W = distance_image.shape[-2], distance_image.shape[-1]  # Height and Width of the distance image
        try:
            grid = points.get_grid_sample_points(H,W)
        except:
            grid = points.clone()
            grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0  # x (width)
            grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0  # y (height)
            grid = grid.view(1, -1, 1, 2)
        interpolated_values = F.grid_sample(distance_image, grid, mode='bilinear', align_corners=True)
        return interpolated_values
    
    def find_nearest_local_minimum(self, image, sampled_points, reference_point):
        """
        Finds the nearest local minimum along the line relative to the reference point.

        Args:
            image (torch.Tensor): The image to sample from, of shape (C, H, W).
            sampled_points (torch.Tensor): A tensor of shape (num_samples, 2) with the points sampled along the line.
            reference_point (torch.Tensor): A 2D tensor representing the reference point (x, y).
        
        Returns:
            dict: Information about the nearest local minimum (position and value).
        """
        # Sample the image along the line to get the intensity values
        image_values = self.points_in_image(sampled_points, image).squeeze(0).squeeze(0) # .mean(dim=0)  # Get grayscale values (mean over channels)

        # Find the local minima along the line
        # local_minima_indices = self.find_local_minima_along_line(image_values) ###
        dist = torch.norm(sampled_points - reference_point, dim=1)
        ref_ind = torch.argmin(dist)
        local_minima_indices, min_value = self.find_nearest_minimum(image_values, ref_ind)
            
        # self.plot_min_line(image, sampled_points, reference_point)#, image_values) #debug
        # self.plot_slice(image_values) # debug
        
        if local_minima_indices == None:
            return None, None  # No local minima found

        # Calculate distances from the reference point to each local minimum
        distances = torch.norm(sampled_points[local_minima_indices] - reference_point)#, dim=1)

        # Find the index of the nearest local minimum
        nearest_minimum_index = torch.argmin(distances)
        nearest_minimum_position = sampled_points[local_minima_indices]
        nearest_minimum_value = image_values[local_minima_indices]
        plot_sampled_points_with_intensity(image, sampled_points, image_values, reference_point, nearest_minimum_position)
        return nearest_minimum_position, nearest_minimum_value
    
    def find_local_minima(self,tensor):
        local_minima_indices = []
        n = len(tensor)

        for i in range(1, n - 1):
            # Handle the case where the current value is less than both neighbors
            if tensor[i - 1] > tensor[i] < tensor[i + 1]:
                local_minima_indices.append(i)
            # Handle the case where adjacent values are equal and both are smaller than neighbors
            elif tensor[i - 1] > tensor[i] == tensor[i + 1]:
                local_minima_indices.append(i)
                local_minima_indices.append(i + 1)
        
        return list(set(local_minima_indices))  # Remove duplicates, if any

    # Function to find the nearest minimum on either side of a given start point
    def find_nearest_minimum(self,tensor, start_idx):
        # Find all local minima
        local_minima_indices = self.find_local_minima(tensor)
        
        # Separate minima into those left and right of the start point
        left_minima = [i for i in local_minima_indices if i < start_idx]
        right_minima = [i for i in local_minima_indices if i > start_idx]
        
        # Find the nearest minima on either side, if they exist
        nearest_left = min(left_minima, key=lambda i: start_idx - i, default=None)
        nearest_right = min(right_minima, key=lambda i: i - start_idx, default=None)
        
        # Return the nearest minima
        if nearest_left is not None and nearest_right is not None:
            if (start_idx - nearest_left) <= (nearest_right - start_idx):
                return nearest_left, tensor[nearest_left]
            else:
                return nearest_right, tensor[nearest_right]
        elif nearest_left is not None:
            return nearest_left, tensor[nearest_left]
        elif nearest_right is not None:
            return nearest_right, tensor[nearest_right]
        else:
            return None, None  # No local minima found
        
    def find_local_minima_along_line(self, image_values):
        """
        Finds the indices of local minima in the sampled image values along a line.

        Args:
            image_values (torch.Tensor): A 1D tensor of sampled image values along the line.
        
        Returns:
            torch.Tensor: A tensor of indices where local minima occur.
        """
        # Compare each point with its neighbors to find local minima
        # Shifted comparison for left and right neighbors
        left_shifted = image_values[:-2]  # Shift the values left
        right_shifted = image_values[2:]  # Shift the values right
        center_values = image_values[1:-1]  # Middle values

        # Condition for local minima: center_value is less than both neighbors
        local_minima_mask = (center_values < left_shifted) & (center_values < right_shifted)

        # Get indices of local minima and adjust indices because of shifting
        local_minima_indices = torch.nonzero(local_minima_mask.squeeze(1)) + 1  # Adjust by +1 to map to correct positions
        return local_minima_indices.flatten()
    
    def sample_image_along_line(self, image, sampled_points):
        """
        Samples the image along the provided points using bilinear interpolation.
        
        Args:
            image (torch.Tensor): The image to sample from, of shape (C, H, W).
            sampled_points (torch.Tensor): A tensor of points (x, y) along the line of shape (num_samples, 2).
        
        Returns:
            torch.Tensor: The sampled image values along the line, of shape (C, num_samples).
        """
        # Normalize sampled points to the range [-1, 1] for grid_sample
        height, width = image.shape[-2], image.shape[-1]
        
        # Convert points to the range [-1, 1] for both x and y
        # sampled_points_normalized = sampled_points.clone()
        # sampled_points_normalized[:, 0] = 2.0 * (sampled_points[:, 0] / (width - 1)) - 1.0  # Normalize x-coordinates
        # sampled_points_normalized[:, 1] = 2.0 * (sampled_points[:, 1] / (height - 1)) - 1.0  # Normalize y-coordinates
        sampled_points_normalized = torch.stack([
            2.0 * (sampled_points[:, 0] / (width - 1)) - 1.0,  # Normalize x-coordinate
            2.0 * (sampled_points[:, 1] / (height - 1)) - 1.0  # Normalize y-coordinate
        ])

        
        # Reshape points to match the grid_sample format
        grid = sampled_points_normalized.view(1, -1, 1, 2)  # Shape: (1, num_samples, 1, 2)

        # Use grid_sample to sample image values at the given points
        sampled_values = F.grid_sample(image.unsqueeze(0).unsqueeze(0), grid, align_corners=True, mode='bilinear', padding_mode='border')
        
        return sampled_values.squeeze(0).squeeze(1)  # Return shape (C, num_samples)
    
    def closest_point_loss(self,points_set_a, points_set_b):
        """
        Computes the loss between two sets of points using closest point matching.
        For each point in points_set_a, we find the closest point in points_set_b.
        """
        # Get the number of points in each set
        n_a = points_set_a.size(0)
        n_b = points_set_b.size(0)

        # Expand both sets to calculate pairwise distances between all points
        points_a_exp = points_set_a.unsqueeze(1).expand(n_a, n_b, -1)  # Shape: (n_a, n_b, 3)
        points_b_exp = points_set_b.unsqueeze(0).expand(n_a, n_b, -1)  # Shape: (n_a, n_b, 3)

        # Calculate pairwise distances between points in the two sets
        distances = torch.norm(points_a_exp - points_b_exp, dim=2)  # Shape: (n_a, n_b)

        # For each point in set A, find the closest point in set B
        min_distances_a_to_b, _ = torch.min(distances, dim=1)  # Shape: (n_a,)

        # For each point in set B, find the closest point in set A
        min_distances_b_to_a, _ = torch.min(distances, dim=0)  # Shape: (n_b,)

        # The loss is the sum of both matching directions
        loss = torch.mean(min_distances_a_to_b) + torch.mean(min_distances_b_to_a)
        return loss
    
    # def plot_min_line(self, image, sampled_points, reference_point):
    #     # Create a figure and axis
    #     plt.figure(figsize=(6, 6))
        
    #     plt.imshow(image.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='gray', alpha=0.5)

    #     # Plot the single point
    #     plt.scatter(reference_point[0].detach().cpu().detach().numpy(), reference_point[1].cpu().detach().numpy(), color='red', label='Single Point', s=30)

    #     # Plot the list of points
    #     plt.scatter(sampled_points[:, 0].cpu().detach().numpy(), sampled_points[:, 1].cpu().detach().numpy(), color='blue', label='List of Points', s=10)

    #     # Add grid lines, labels, and a legend
    #     plt.axhline(0, color='black',linewidth=0.5)
    #     plt.axvline(0, color='black',linewidth=0.5)
    #     # plt.xlim(-5, 5)
    #     # plt.ylim(-5, 5)
    #     plt.title("Tensor as Image, Single Point, and List of Points")
    #     plt.colorbar(plt.imshow(image.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='gray', alpha=0.5), label="Image Intensity")
    #     plt.legend()

    #     # Show the plot
    #     plt.show()
    
    # def plot_slice(self,tensor_1d):
    #     plt.figure(figsize=(6, 4))
        
    #     derivative = torch.diff(tensor_1d.squeeze(1))
    #     derivative2n = torch.diff(derivative)
    #     plt.plot(tensor_1d.cpu().detach().numpy(), marker='o', linestyle='-', color='b', label='1D Tensor')
        
    #     plt.plot(derivative.cpu().detach().numpy(), marker='o', linestyle='-', color='r', label='1D Tensor')

    #     plt.plot(derivative2n.cpu().detach().numpy(), marker='o', linestyle='-', color='g', label='1D Tensor')
    #     # Add title and labels
    #     plt.title("Plot of 1D Tensor")
    #     plt.xlabel("Index")
    #     plt.ylabel("Value")
    #     plt.grid(True)
    #     plt.legend()

    #     # Show the plot
    #     plt.show()
    
TEST_CRYSTALS = {
    'cube': {
        'lattice_unit_cell': [1, 1, 1],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'point_group_symbol': '222',
        'scale': 1,
        'origin': [0.5, 0, 0],
        'distances': [1., 1., 1.],
        'rotation': [0.2, 0.3, 0.3],
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
        'rotation': [0.2, 0.3, 0.3],
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
        # 'rotation': [0.3, 0.3, 0.3],
        'rotation': [0.0, 0.0, 0.0],
        'point_group_symbol': '222',
        'scale': 3.0,
    },
    'alpha_test': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 0, -1), (1, 1, 1), (1, 1, -1), (0, 1, 1), (0, 1, -1), (1, 0, 0)],
        'distances': [0.6, 0.4, 1.0, 1.01, 1.28, 0.8, 1.00],
        # 'rotation': [0.3, 0.3, 0.3],
        'rotation': [0.0, 0.0, 0.0],
        'point_group_symbol': '222',
        'scale': 3.0,
    },
}

def log_plot_to_tensorboard(writer, tag, figure, global_step):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    writer.add_image(tag, image, global_step)
    buf.close()

def generate_synthetic_crystal(
        crystal,
        save_dir,
        res = 400,
    ):
    # first generate mesh with cube
    
    # crystal = Crystal(**TEST_CRYSTALS['cube'])
    # crystal.scale.data= init_tensor(1.2, device=crystal.scale.device)
    # crystal.origin.data[:2] = torch.tensor([0, 0], device=crystal.origin.device)
    # crystal.origin.data[2] -= crystal.vertices[:, 2].min()
    # v, f = crystal.build_mesh()
    # crystal.to(device)


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
        integrator_max_depth=3,
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
        colour_facing_towards= [1.0,1.0,1.0],
        colour_facing_away = [1.0,1.0,1.0]
    )
    img_overlay = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    img_overlay[:, :, 3] = (img_overlay[:, :, 3] * 0.5).astype(np.uint8)
    # fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    # axs[0].imshow(img)
    img_og = img
    # axs[0].imshow(img_overlay)
    # fig.show()
    # get contour image
    model_path = Path(ROOT_PATH / 'tmp' / 'bsds500_pascal_model.pth')
    
    rcf = load_rcf(model_path)
    rcf.to(device)
    img = to_tensor(img).to(device)[None, ...]
    feature_maps = rcf(img, apply_sigmoid=False)
    #third one seems best for now
    
    dist_maps_arr = []
    # Save the feature maps
    for i, feature_map in enumerate(feature_maps):
        feature_map = to_numpy(feature_map).squeeze()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        if i == len(feature_maps) - 1:
            name = 'fused'
        else:
            name = f'feature_map_{i + 1}'
        img = Image.fromarray((feature_map * 255).astype(np.uint8))
        img.save(save_dir / 'rcf_featuremaps' / f'{name}.png')

        # Save the distance transform
        # feature_map[feature_map < 0.2] = 0
        # img = img.resize((200, 200))
        img = np.array(img).astype(np.float32)/255
        img = (img - img.min()) / (img.max() - img.min())
        thresh = 0.5
        img[img < thresh] = 0
        img[img >= thresh] = 1
        dist = distance_transform_edt(1-img)  #, metric='taxicab')
        dist = dist.astype(np.float32)
        dist = (dist - dist.min()) / (dist.max() - dist.min())
        dist_maps_arr.append(to_tensor(dist))
        Image.fromarray((dist * 255).astype(np.uint8)).save(save_dir / 'rcf_featuremaps' / f'dists_{name}.png')
    
    dist_maps = torch.stack(dist_maps_arr)
    dist_map = dist_maps[2].unsqueeze(0)
    f_map = torch.abs(feature_maps[2])
    
    f_map_np = to_numpy(f_map).squeeze()
    f_map_np = (f_map_np - f_map_np.min()) / (f_map_np.max() - f_map_np.min())
    del rcf
    torch.cuda.empty_cache() 
    
    return f_map, dist_map, img_og, img_overlay, zoom

def run():
    
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}'#_{args.image_path.name}'
    rcf_dir = save_dir / 'rcf_featuremaps'
    save_dir.mkdir(parents=True, exist_ok=True)
    rcf_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'tensorboard_logs'))

    # crystal_tar = Crystal(**TEST_CRYSTALS['cube'])
    crystal_tar = Crystal(**TEST_CRYSTALS['alpha'])
    crystal_tar.scale.data= init_tensor(1.2, device=crystal_tar.scale.device)
    crystal_tar.origin.data[:2] = torch.tensor([0, 0], device=crystal_tar.origin.device)
    crystal_tar.origin.data[2] -= crystal_tar.vertices[:, 2].min()
    v, f = crystal_tar.build_mesh()
    crystal_tar.to(device)

    # generate a synthetic crystal to compare too
    f_map, dist_map, img, img_overlay, zoom = generate_synthetic_crystal(
        crystal_tar,
        save_dir,
    )
    
    #convert image overlay
    overlay_pil = Image.fromarray(img_overlay.astype(np.uint8))
    overlay_pil.save(save_dir / 'overlay_image.png')
    
    img_gray = np.dot(img_overlay[..., :3], [0.2989, 0.5870, 0.1140])

    # Step 2: Invert the grayscale image
    img_inverted = 255 - img_gray # dont invert #  

    image_pil = Image.fromarray(img_inverted.astype(np.uint8))
    image_pil.save(save_dir / 'inverted_image.png')

    # Save the distance transform
    # feature_map[feature_map < 0.2] = 0
    # img = img.resize((200, 200))
    img_inverted = np.array(img_inverted).astype(np.float32)/255
    img_inverted = gaussian_filter(img_inverted,sigma=2)
    img_inverted = (img_inverted - img_inverted.min()) / (img_inverted.max() - img_inverted.min())
    image_pil = Image.fromarray((img_inverted*255).astype(np.uint8))
    image_pil.save(save_dir / 'inverted__step_image.png')
    thresh = 0.95
    img_inverted[img_inverted < thresh] = 0
    img_inverted[img_inverted >= thresh] = 1
    dist = distance_transform_edt(1-img_inverted)  #, metric='taxicab')
    dist = dist.astype(np.float32)
    dist = (dist - dist.min()) / (dist.max() - dist.min())
    dist = 1-dist # invert again
    distance_map_tensor = to_tensor(dist).squeeze(0)
    Image.fromarray((dist * 255).astype(np.uint8)).save(save_dir / 'distance_map.png')
    dist_map = dist



    # Step 3: Convert to tensor
    img_tensor = distance_map_tensor


    # Normalize the tensor to the range [0, 1] if needed
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    # f_map = f_map.to(device)
    # dist_map = dist_map.to(device)
    
    projector_tar = ProjectorPoints(crystal_tar,
                                external_ior=1.333,
                                zoom =zoom,
                                image_size=f_map.shape[-2:],)
    projector_tar.to(device)
    points_tar = projector_tar.project()


    # crystal_opt = Crystal(**TEST_CRYSTALS['cube_test'])
    # crystal_opt = Crystal(**TEST_CRYSTALS['alpha_test'])
    crystal_opt = crystal_tar.clone()
    crystal_opt.scale.data= init_tensor(1.2, device=crystal_opt.scale.device)
    crystal_opt.origin.data[:2] = torch.tensor([0, 0], device=crystal_opt.origin.device)
    crystal_opt.origin.data[2] -= crystal_opt.vertices[:, 2].min()
    crystal_tar_distances = crystal_tar.distances
    # Define the percentage range (e.g., ±5%)
    percentage = 0.05
    # Generate random values in the range [-percentage, +percentage]
    random_factors = torch.randn_like(crystal_opt.distances,device=crystal_opt.scale.device)  * percentage
    # Add the random amount to each value in the tensor
    modified_distances = crystal_tar_distances * (1 + random_factors)
    crystal_opt.distances.data = init_tensor(modified_distances, device=crystal_opt.scale.device)
    # crystal_opt.distances = modified_distances
    v, f = crystal_opt.build_mesh(distances=crystal_opt.distances)
    crystal_opt.to(device)

    projector_opt = ProjectorPoints(crystal_opt,
                                external_ior=1.333,
                                zoom = zoom,
                                image_size=f_map.shape[-2:])
    projector_opt.to(device)
    points_opt = projector_opt.project()

    params = {
            'distances': [crystal_opt.distances],
        }
    
    model = ContourDistanceNormalLoss()
    model.to(device)

    #inital 
    img_int = img_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    multiplot(img_int,points_opt,save_dir,'inital')

    optimizer = optim.Adam(params['distances'], lr=1e-2)
    target_dist = crystal_tar.distances
    # prev_dist = crystal_opt.distances
    # Training loop
    # with torch.autograd.detect_anomaly(False):
    for step in range(100):  # Run for 100 iterations
        print(f"Step {step}")
        # step.to(device)
        optimizer.zero_grad()  # Zero the gradients
        
        # Convert polar to Cartesian coordinates
        v, f = crystal_opt.build_mesh()
        
        projector_opt = ProjectorPoints(crystal_opt,
                            zoom = zoom,
                            image_size=f_map.shape[-2:],
                            external_ior=1.333,)
        points_opt = projector_opt.project()
        
        dist = crystal_opt.distances
        # a = points_opt.get_all_points_tensor()
        # print(f"points tensor {a}")
        # Forward pass: get the pixel value at the current point (x, y)
        loss = model(points_opt, img_tensor)  # Call model's forward method with Cartesian coordinates
        # Perform backpropagation (minimize the pixel value)
        
        loss.backward(retain_graph=True)

        # Check if the gradients for r and theta are non-zero
        print(f"Step {step}: {projector_opt.distances}")
        
        # Check if gradients are non-zero before optimizer step
        # if dist.grad.abs() < 1e-6:
        #     print(f"Warning: One of the gradients is very small at step {step}")
        
        # Update the radial parameters
        for group in optimizer.param_groups:
            print(group)
            
        if step % 1 == 0:
            print("plotting")
            # projector = Projector(
            #     crystal=crystal_opt,
            #     external_ior=1.333,
            #     image_size=f_map.shape[-2:],
            #     zoom=zoom,
            #     transparent_background=True,
            #     multi_line=True,
            # )
            # img_overlay = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
            # img_overlay[:, :, 3] = (img_overlay[:, :, 3] * 0.5).astype(np.uint8)
            # overlay_plot(img,points_opt,save_dir,'progress_' + str(step))
            multiplot(img_overlay,points_opt,save_dir,'progress_' + str(step).zfill(3),plot_loss_dist=True, writer=writer,global_step=step)
        
        optimizer.step()

        # Log the loss value
        writer.add_scalar('Loss', loss.item(), step)
        # Print the updated polar coordinates and the current loss
        print(f"Step {step}: loss: {loss}")


if __name__ == "__main__":
    print(f"running test")
    run()
