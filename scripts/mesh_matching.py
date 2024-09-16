# from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path

import numpy as np
import torch
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
# import yaml
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from torchvision.transforms.functional import to_tensor

from crystalsizer3d import LOGS_PATH, ROOT_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.projector import Projector
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.nn.models.rcf import RCF
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import project_to_image
from crystalsizer3d.util.utils import print_args, to_numpy, init_tensor


from edge_detection import load_rcf

if USE_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.autograd.set_detect_anomaly(True)
    
"""
Outline - 
1. This class takes a crystal image, calculates its contours using
RCF
2. Calculates a mesh projection as points with normal vectors
3. Calculates the loss 
"""

class ContourDistanceNormalLoss(nn.Module):
    def __init__(self,
                 model_path: Path = ROOT_PATH / 'tmp' / 'bsds500_pascal_model.pth',
                 ):
        super(ContourDistanceNormalLoss, self).__init__()
        
        #load RCF model            
        self.rcf = load_rcf(model_path)
        
        # Detect edges
        
        
    
    def forward(self, distance_image, points, normals):
        """
        Arguments:
        - distance_image (torch.Tensor): A tensor of shape (H, W) representing the distance map where 0 represents the contour.
        - points (torch.Tensor): A tensor of shape (N, 2), where N is the number of points, each defined by (x, y) coordinates.
        - normals (torch.Tensor): A tensor of shape (N, 2), where N is the number of points, each defined by a normal vector (nx, ny).
        
        Returns:
        - loss (torch.Tensor): The computed loss based on the change in the distance along the normal vectors.
        """
        
        N = points.shape[0]  # Number of points
        H, W = distance_image.shape[-2], distance_image.shape[-1]  # Height and Width of the distance image
        
        # Ensure points are within the bounds of the image
        points = points.clamp(min=0, max=max(H-1, W-1))
        
        # Initialize a tensor to accumulate the losses
        losses = []
        
        # Small step size for finite difference
        delta = 1e-3
        
        for i in range(N):
            # Extract the point and the normal vector for this iteration
            x, y = points[i]
            reference_point = points[i]
            nx, ny = normals[i]
            
            line_points = self.sample_points_along_line_full_image(points[i],normals[i], 100, distance_image.shape)
            # sampled_values = self.sample_image_along_line(distance_image,line_points)
            reference_value = self.sample_image_at_point(distance_image, reference_point)
            
            nearest_minimum_position, nearest_minimum_value = self.find_nearest_local_minimum(distance_image, line_points, reference_value)
                        
            # Compute the mean squared error between the sampled values and the reference value
            loss = F.mse_loss(nearest_minimum_value, reference_value)
            
            losses.append(loss)
            
            # something is funnyt wioth how its getting the vallues from the image as x approches w and y approches h
        
        # Return the mean loss across all points
        return torch.mean(torch.stack(losses))
    
    def sample_points_along_line_full_image(self, point, normal_vector, num_samples, image_shape):
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
        # # Ensure point and normal_vector are 2D tensors
        # point = point.float()
        # normal_vector = normal_vector.float()

        # Compute the direction vector by rotating the normal vector by 90 degrees
        direction_vector = torch.tensor([-normal_vector[1], normal_vector[0]],device=device)

        # Normalize the direction vector
        direction_vector_norm = direction_vector / torch.norm(direction_vector)

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
        if direction_vector[0] != 0:
            t_left = (0 - point[0]) / direction_vector[0]  # Intersection with left boundary
            t_right = (width - 1 - point[0]) / direction_vector[0]  # Intersection with right boundary
        else:
            t_left, t_right = -max_float, max_float

        # Handle intersections with top (y=0) and bottom (y=height-1) boundaries
        if direction_vector[1] != 0:
            t_top = (0 - point[1]) / direction_vector[1]  # Intersection with top boundary
            t_bottom = (height - 1 - point[1]) / direction_vector[1]  # Intersection with bottom boundary
        else:
            t_top, t_bottom = -max_float, max_float

        # Take the maximum of left/top and the minimum of right/bottom
        t_min = max(t_left, t_top)
        t_max = min(t_right, t_bottom)

        # Compute the actual intersection points using the t values
        intersection_1 = point + t_min * direction_vector
        intersection_2 = point + t_max * direction_vector

        return torch.stack([intersection_1, intersection_2])
   
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
    
    def sample_image_at_point(self,image, point):
        """
        Samples the image at a specific point using bilinear interpolation.

        Args:
            image (torch.Tensor): The image to sample from, of shape (C, H, W).
            point (torch.Tensor): A 2D tensor representing the point (x, y) to sample.

        Returns:
            torch.Tensor: The interpolated image value at the given point, of shape (C,).
        """
        # Normalize point to the range [-1, 1]
        height, width = image.shape[-2], image.shape[-1]
        
        point_normalized = torch.stack([
            2.0 * (point[0] / (width - 1)) - 1.0,  # Normalize x-coordinate
            2.0 * (point[1] / (height - 1)) - 1.0  # Normalize y-coordinate
        ])

        # Reshape the point to match the grid_sample format
        grid = point_normalized.view(1, 1, 1, 2)  # Shape: (1, 1, 1, 2)

        # Use grid_sample to sample image value at the given point
        sampled_value = F.grid_sample(image.unsqueeze(0).unsqueeze(0), grid, align_corners=True, mode='bilinear', padding_mode='border')
        
        return sampled_value.squeeze(0).squeeze(1).squeeze(1)  # Return shape (C,)
  
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
        image_values = self.sample_image_along_line(image, sampled_points).squeeze(0) # .mean(dim=0)  # Get grayscale values (mean over channels)

        # Find the local minima along the line
        local_minima_indices = self.find_local_minima_along_line(image_values)

        if len(local_minima_indices) == 0:
            return None  # No local minima found

        # Calculate distances from the reference point to each local minimum
        distances = torch.norm(sampled_points[local_minima_indices] - reference_point, dim=1)

        # Find the index of the nearest local minimum
        nearest_minimum_index = torch.argmin(distances)
        nearest_minimum_position = sampled_points[local_minima_indices[nearest_minimum_index]]
        nearest_minimum_value = image_values[local_minima_indices[nearest_minimum_index]]

        return nearest_minimum_position, nearest_minimum_value
        # return {
        #     "position": nearest_minimum_position,
        #     "value": nearest_minimum_value,
        #     "index": local_minima_indices[nearest_minimum_index],
        #     "distance": distances[nearest_minimum_index]
        # }
    
def run(distance_image):
    
    # Create an output directory
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}'#_{args.image_path.name}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the input image
    # img = Image.open(args.image_path)
    # img.save(save_dir / 'input.png')
    # Initial points, not necessarily on the contour
    points = torch.tensor([[98, 50],  # Near bottom of the circle
                        [50, 98],  # Near right of the circle
                        [1, 50],  # Near top of the circle
                        [50, 1]],
                        dtype=torch.float,
                        device=device,
                        requires_grad=True)  # Near left of the circle

    # Plot the distance image to visualize the contour
    plot(distance_image,points,save_dir,'start')


    # Corresponding normals (pointing towards the center)
    normals = torch.tensor([[0, 1],   # Pointing upwards
                            [-1, 0],  # Pointing left
                            [0, -1],  # Pointing downwards
                            [1, 0]], 
                           device=device,
                           dtype=torch.float)

    # Instantiate the loss function
    loss_fn = ContourDistanceNormalLoss()

    # Define an optimizer to update the points
    optimizer = optim.Adam([points], lr=0.1)
    distance_image = distance_image.to(device)
    # Optimization loop
    num_iterations = 1000
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Compute the loss
        loss = loss_fn(distance_image, points, normals)
        
        # Print loss every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
        
        # Backpropagate the loss and update the points
        loss.backward()
        optimizer.step()

        # Optional: Clamp points to keep them within the image bounds
        points.data.clamp_(0, max(H - 1, W - 1))
        
        if i % 100 == 0:
            plot(distance_image,points,save_dir,'progress_' + str(i))

    # Final points after optimization
    print(f"Final Points:\n{points.detach().cpu().numpy()}")

    # Visualize final points on the distance image
    plot(distance_image,points,save_dir,'end')
    
        
def plot(distance_image,points,save_dir,name):
    plt.imshow(distance_image.cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.scatter(points[:, 0].detach().cpu().numpy(), points[:, 1].detach().cpu().numpy(), c='blue')
    plt.title('Final Points After Optimization')
    plt.savefig(save_dir / (name + '.png'))
    plt.close()

def plot_step(distance_image,points,minimum):
    plt.imshow(distance_image.cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.scatter(points[:, 0].detach().cpu().numpy(), points[:, 1].detach().cpu().numpy(), c='blue', s=1)
    plt.scatter(minimum[0].detach().cpu().numpy(), minimum[1].detach().cpu().numpy(), c='green')
    plt.title('In progress')
    plt.show()

TEST_CRYSTALS = {
    'cube': {
        'lattice_unit_cell': [1, 1, 1],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'point_group_symbol': '222',
        'scale': 1,
        'origin': [0.5, 0, 0],
        'distances': [1., 1., 1.],
        'rotation': [0.0, 0.0, 0.0],
        'material_ior': 1.2,
        'material_roughness': 1.5#0.01
    },
}

def run_cube():
    # first generate mesh with cube
    
    res = 400
    crystal = Crystal(**TEST_CRYSTALS['cube'])
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
    )
    img_overlay = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    img_overlay[:, :, 3] = (img_overlay[:, :, 3] * 0.5).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.imshow(img_overlay)


    fig.tight_layout()
    plt.show()
        
if __name__ == "__main__":
    # need a better way of loading image or atleast optionsd
    # image_path = "data\LGA_000239.jpg"
    # img = Image.open(image_path)
    # img = to_tensor(img).to(device)[None, ...]

    # Create a synthetic distance image (e.g., distance from a circle)
    # Create a synthetic distance image (e.g., distance from a circle)
    # H, W = 100, 100
    # center = torch.tensor([50.0, 50.0])
    # radius = 30.0

    # # Create a grid for the image
    # y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    # distance_image = torch.abs(torch.sqrt((x - center[0])**2 + (y - center[1])**2) - radius)
    # distance_image = (distance_image - distance_image.min() ) /  (distance_image.max() - distance_image.min())
    # run(distance_image)
    
    run_cube()
