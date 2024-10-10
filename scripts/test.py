
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class MyModel(nn.Module):
    def __init__(self, n):
        super(MyModel, self).__init__()
        # Define the distance tensor as a learnable parameter
        self.distance = nn.Parameter(torch.randn(n, requires_grad=True))

    def forward(self, indices):
        # Dynamically rearrange distance at each forward pass using the indices
        d_2 = self.distance[indices]
        return d_2

# Define the model
n = 5
model = MyModel(n)

# Define some indices for rearrangement (dynamic indexing)
indices = torch.tensor([2, 1, 0, 4, 3])

# Define an optimizer (optimizing the distance parameter)
optimizer = optim.SGD(model.parameters(), lr=0.1)
for step in range(100):
    # Forward pass: compute d_2
    d_2 = model(indices)

    # Some loss function (for example, sum of d_2)
    loss = d_2.sum()

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    # Check updated distance tensor
    print("Updated distance tensor:", model.distance)




# import torch
# from torch.autograd import gradcheck
# from crystalsizer3d.crystal import Crystal
# from crystalsizer3d.util.utils import print_args, to_numpy, init_tensor
# from crystal_points import ProjectorPoints, plot_2d_projection
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# USE_CUDA = True
# if USE_CUDA:
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()

#     def forward(self, distances):
#         return distances.sum()



# TEST_CRYSTALS = {
#     'cube': {
#         'lattice_unit_cell': [1, 1, 1],
#         'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
#         'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
#         'point_group_symbol': '222',
#         'scale': 1,
#         'origin': [0.5, 0, 0],
#         'distances': [1., 1., 1.],
#         'rotation': [0.0, 0.0, 0.0],
#         'material_ior': 1.2,
#         'material_roughness': 1.5#0.01
#     },
#     'cube_test': {
#         'lattice_unit_cell': [1, 1, 1],
#         'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
#         'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
#         'point_group_symbol': '222',
#         'scale': 1,
#         'origin': [0.5, 0, 0],
#         'distances': [1.3, 1.0, 1.0],
#         'rotation': [0.0, 0.0, 0.0],
#         'material_ior': 1.2,
#         'material_roughness': 1.5#0.01
#     },
# }

# crystal_opt = Crystal(**TEST_CRYSTALS['cube_test'])
# crystal_opt.scale.data= init_tensor(1.2, device=crystal_opt.scale.device)
# crystal_opt.origin.data[:2] = torch.tensor([0, 0], device=crystal_opt.origin.device)
# crystal_opt.origin.data[2] -= crystal_opt.vertices[:, 2].min()
# v, f = crystal_opt.build_mesh()
# crystal_opt.to(device)

# projector = ProjectorPoints(crystal_opt,
#                             external_ior=1.333,)
# points = projector.project()

# params = {
#         'distances': [crystal_opt.distances],
#     }

# model = SimpleModel()
# model.to(device)

# optimizer = optim.Adam(params['distances'], lr=0.001)
# prev_dist = crystal_opt.distances
# # Training loop
# for step in range(100):  # Run for 100 iterations
#     print(f"Step {step}")
#     optimizer.zero_grad()  # Zero the gradients
    
#     # Convert polar to Cartesian coordinates
#     v, f = crystal_opt.build_mesh()
    
#     dist = crystal_opt.distances
    
#     # Forward pass: get the pixel value at the current point (x, y)
#     loss = model(dist)  # Call model's forward method with Cartesian coordinates
    
#     # Perform backpropagation (minimize the pixel value)
#     loss.backward()

#     # Check if the gradients for r and theta are non-zero
#     print(f"Step {step}: Gradients - dist.grad: {dist.grad}")
    
#     # Check if gradients are non-zero before optimizer step
#     # if dist.grad.abs() < 1e-6:
#     #     print(f"Warning: One of the gradients is very small at step {step}")
    
#     # Update the radial parameters
#     optimizer.step()

#     # Print the updated polar coordinates and the current loss
#     print(f"Step {step}: Polar Coordinates (r: {dist}")
#         #   f"Cartesian Coordinates (x: {x.item():.2f}, y: {y.item():.2f}), Loss {loss.item():.2f}")
    
#     # Check if the parameters are actually being updated
#     if torch.equal(dist,prev_dist):
#         print(f"Warning: r and theta haven't changed at step {step}")
#     else:
#         print(f"Parameters updated: dist_change: {dist - prev_dist}")
    
#     # Update the previous values
#     prev_dist = dist