import torch
from torch import nn 
import numpy as np
from crystalsizer3d.crystal import Crystal
from crystalsizer3d import USE_CUDA
from crystalsizer3d.util.utils import print_args, to_numpy, init_tensor
import torch.nn.functional as F
import torchvision.transforms as transforms
from crystal_points import ProjectorPoints
import torch.optim as optim
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
        # loss = self.closest_point_loss(dist1, dist2)
        loss = self.points_in_image(points,distance_image)
        return loss
    
    def points_in_image(self,points,distance_image):
        N = points.shape[0]  # Number of points
        H, W = distance_image.shape[-2], distance_image.shape[-1]  # Height and Width of the distance image
        
        # Ensure points are within the bounds of the image
        points = points.clamp(min=0, max=max(H-1, W-1))
        loss = 0
        return loss
    
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
    'cube_test': {
        'lattice_unit_cell': [1, 1, 1],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'point_group_symbol': '222',
        'scale': 1,
        'origin': [0.5, 0, 0],
        'distances': [1.3, 1.0, 1.0],
        'rotation': [0.0, 0.0, 0.0],
        'material_ior': 1.2,
        'material_roughness': 1.5#0.01
    },
}



if __name__ == "__main__":
    print(f"running test")
    
    test_image = "logs\\scripts\\mesh_matching\\20240919_1146\\rcf_featuremaps\\feature_map_3.png"    
    
    # Read a PIL image
    image = Image.open(test_image)

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)

    # print the converted Torch tensor
    
    
    crystal_tar = Crystal(**TEST_CRYSTALS['cube'])
    crystal_tar.scale.data= init_tensor(1.2, device=crystal_tar.scale.device)
    crystal_tar.origin.data[:2] = torch.tensor([0, 0], device=crystal_tar.origin.device)
    crystal_tar.origin.data[2] -= crystal_tar.vertices[:, 2].min()
    v, f = crystal_tar.build_mesh()
    crystal_tar.to(device)

    projector_tar = ProjectorPoints(crystal_tar,
                                external_ior=1.333,)
    projector_tar.to(device)
    points_tar = projector_tar.project()


    
    crystal_opt = Crystal(**TEST_CRYSTALS['cube_test'])
    crystal_opt.scale.data= init_tensor(1.2, device=crystal_opt.scale.device)
    crystal_opt.origin.data[:2] = torch.tensor([0, 0], device=crystal_opt.origin.device)
    crystal_opt.origin.data[2] -= crystal_opt.vertices[:, 2].min()
    v, f = crystal_opt.build_mesh()
    crystal_opt.to(device)

    projector_opt = ProjectorPoints(crystal_opt,
                                external_ior=1.333,)
    projector_opt.to(device)
    points_opt = projector_opt.project()

    params = {
            'distances': [crystal_opt.distances],
        }
    
    model = ContourDistanceNormalLoss()
    model.to(device)

    optimizer = optim.Adam(params['distances'], lr=0.01)
    target_dist = crystal_tar.distances
    # prev_dist = crystal_opt.distances
    # Training loop
    with torch.autograd.detect_anomaly():
        for step in range(100):  # Run for 100 iterations
            print(f"Step {step}")
            # step.to(device)
            optimizer.zero_grad()  # Zero the gradients
            
            # Convert polar to Cartesian coordinates
            v, f = crystal_opt.build_mesh()
            
            projector_opt = ProjectorPoints(crystal_opt,
                                external_ior=1.333,)
            points_opt = projector_opt.project()
            
            dist = crystal_opt.distances
            a = points_opt.get_all_points_tensor()
            b = points_tar.get_all_points_tensor()
            # Forward pass: get the pixel value at the current point (x, y)
            loss = model(a, b)  # Call model's forward method with Cartesian coordinates
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
            
            optimizer.step()

            # Print the updated polar coordinates and the current loss
            print(f"Step {step}: loss: {loss}")