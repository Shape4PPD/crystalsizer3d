### delete this file
import matplotlib
matplotlib.use('TkAgg')
from crystalsizer3d import LOGS_PATH, ROOT_PATH, START_TIMESTAMP, USE_CUDA, logger
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image
import torchvision.transforms as transforms

# def multiplot(distance_image,points,save_dir, tag, distances=None, plot_loss_dist=False, writer=None, global_step=0):
#     fig, ax = plt.subplots()
#     if isinstance(distance_image, torch.Tensor):
#         plt.imshow(distance_image.squeeze(0).squeeze(0).detach().cpu().numpy())
#     else:
#         plt.imshow(distance_image)
    
#     if plot_loss_dist:
#         points_array, distances_array = points.cpu().detach(), distances.cpu().detach()
#         for point, distance in zip(points_array, distances_array):
#             x, y = point
#             dx, dy = distance

#             if np.array_equal(distance, [0, 0]):
#                 # If the distance is [0, 0], plot a point with a different color
#                 ax.plot(x, y, 'go',markersize=2)  # 'ro' is red circle for stationary points
#             else:
#                 # Otherwise, draw a line from the point to the point + distance
#                 ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
#                 ax.plot(x, y, 'ro',markersize=2)  # 'ro' is red circle for stationary points
#     else:
#         try:
#             points_array = points.get_points_array()
#         except:
#             points_array = points.cpu().detach().numpy()

            
#         # plt.colorbar()

#         plt.scatter(points_array[:, 0], points_array[:, 1], c='blue',s=3)
#     plt.title(f'Step {global_step}')
#     plt.savefig(save_dir / f'{tag}.png')
#     if writer is not None:
#         log_plot_to_tensorboard(writer, tag, fig, global_step)
#     plt.close()
    
def overlay_plot(image1,image2,save_dir,name):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image1)#,origin='lower')
    ax.imshow(image2)#,origin='lower')
    plt.savefig(save_dir / (name + '.png'))
    plt.close()
    
def log_plot_to_tensorboard(writer, tag, figure, global_step):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    writer.add_image(tag, image, global_step)
    buf.close()

# def plot_2d_projection(point_collections: PointCollection, plot_normals=False,ax=None):
#     """
#     Plots 2D points from a PyTorch tensor or a list of tensors.
    
#     Args:
#     points (torch.Tensor or list of torch.Tensor): A tensor of 2D points or a list of tensors of 2D points.
#     """
#     try:
#         plot_2d_projection.counter += 1
#     except:
#          plot_2d_projection.counter = 0
#     if isinstance(point_collections, PointCollection):
#         point_collections = [point_collections]  # Convert to list for uniform processing
    
#     # Define a color map
#     colours = plt.get_cmap('Set1')
#     # colours = plt.get_cmap('tab20')
    
#     for i, point_collection in enumerate(point_collections):
#         if len(point_collection) == 0:
#             continue
#         tensor = point_collection.get_points_and_normals_tensor()
#         if tensor.dim() != 3 or tensor.size(1) != 2:
#             raise ValueError("Each tensor must be of shape (N, 2, 2) where N is the number of points.")
        
#         pointx = tensor[:, 0, 0].detach().cpu().numpy()
#         pointy = tensor[:,0, 1].detach().cpu().numpy()
#         normalx = tensor[:, 1, 0].detach().cpu().numpy()
#         normaly = tensor[:, 1, 1].detach().cpu().numpy()
#         # Plot the points # plot_2d_projection.counter % 20
#         if ax == None:
#             plt.scatter(pointx, pointy, color=colours(i), label=f'Tensor {i+1}',s=2)
#         else:
#             ax.scatter(pointx, pointy, color=colours(i), label=f'Tensor {i+1}',s=2)
            
#         if plot_normals:
#             scale = 0.2
#             for px,py,nx,ny in zip(pointx,pointy,normalx,normaly):
#                 if ax == None:
#                     plt.arrow(px, py, nx*scale, ny*scale, head_width=0.1, head_length=0.1, fc=colours(i), ec=colours(i))
#                 else:
#                     ax.arrow(px, py, nx*scale, ny*scale, head_width=0.1, head_length=0.1, fc=colours(i), ec=colours(i))
    
#     if ax == None:
#         plt.xlim((-1.5,1.5))
#         plt.ylim((2.0,-2.0))
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.legend()
#         plt.show()
        
        

def plot_sampled_points_with_intensity(distance_image, sampled_points, intensity_values, reference_point, nearest_minimum_position):
    """
    Plots an image with sampled points overlaid. The points are colored based on intensity values.
    Adds the reference_point and nearest_minimum_position with specific markers. Saves the plot to a file.

    Args:
        distance_image (np.array): The background image (2D array).
        sampled_points (torch.Tensor): An n x 2 tensor of (x, y) points.
        intensity_values (torch.Tensor): A tensor of length n with intensity values corresponding to each point.
        reference_point (torch.Tensor): A 1 x 2 tensor representing the reference point (x, y).
        nearest_minimum_position (torch.Tensor): A 1 x 2 tensor representing the nearest minimum position (x, y).
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(sampled_points, torch.Tensor):
        sampled_points = sampled_points.cpu().detach().numpy()
    
    if isinstance(intensity_values, torch.Tensor):
        intensity_values = intensity_values.cpu().detach().numpy()
        
    if isinstance(reference_point, torch.Tensor):
        reference_point = reference_point.cpu().detach().numpy()
    
    if isinstance(nearest_minimum_position, torch.Tensor):
        nearest_minimum_position = nearest_minimum_position.cpu().detach().numpy()

    # Display the distance image in the background
    plt.imshow(distance_image.cpu().detach().squeeze(0).squeeze(0).numpy(), cmap='gray', origin='upper')

    # Scatter plot for sampled points with intensity as color
    x = sampled_points[:, 0]
    y = sampled_points[:, 1]
    
    # Create a scatter plot, with color mapped to the intensity values
    scatter = plt.scatter(x, y, c=intensity_values, cmap='jet', s=30)

    # Add a colorbar to represent the intensity scale
    plt.colorbar(scatter, label='Intensity')

    # Plot the reference_point with a green 'x'
    plt.plot(reference_point[0], reference_point[1], 'gx', markersize=20, label='Reference Point')

    # Plot the nearest_minimum_position with a yellow 'x'
    plt.plot(nearest_minimum_position[0], nearest_minimum_position[1], 'yx', markersize=20, label='Nearest Minimum')


    # Display the plot with labeled axes
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Sampled Points with Intensity Overlay')

    save_dir = LOGS_PATH / f'{START_TIMESTAMP}'
    img_dir = save_dir / ("tests")
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_dir / (f"{reference_point}test.png"))
    plt.close()