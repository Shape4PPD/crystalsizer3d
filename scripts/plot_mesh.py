
from crystal_points import PointCollection
import matplotlib.pyplot as plt
import torch
import numpy as np


def multiplot(distance_image,points,save_dir,name):
    try:
        points_array = points.get_points_array()
    except:
        points_array = points.cpu().detach().numpy()
    if isinstance(distance_image, torch.Tensor):
        plt.imshow(distance_image.squeeze(0).squeeze(0).detach().cpu().numpy())
    else:
        plt.imshow(distance_image,origin='lower')
        
    # plt.colorbar()
    plt.scatter(points_array[:, 0], points_array[:, 1], c='blue')
    plt.title('Final Points After Optimization')
    plt.savefig(save_dir / (name + '.png'))
    plt.close()
    
def overlay_plot(image1,image2,save_dir,name):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image1)#,origin='lower')
    ax.imshow(image2)#,origin='lower')
    plt.savefig(save_dir / (name + '.png'))
    plt.close()

def plot_2d_projection(point_collections: PointCollection, plot_normals=False,ax=None):
    """
    Plots 2D points from a PyTorch tensor or a list of tensors.
    
    Args:
    points (torch.Tensor or list of torch.Tensor): A tensor of 2D points or a list of tensors of 2D points.
    """
    try:
        plot_2d_projection.counter += 1
    except:
         plot_2d_projection.counter = 0
    if isinstance(point_collections, PointCollection):
        point_collections = [point_collections]  # Convert to list for uniform processing
    
    # Define a color map
    colours = plt.get_cmap('Set1')
    # colours = plt.get_cmap('tab20')
    
    for i, point_collection in enumerate(point_collections):
        if len(point_collection) == 0:
            continue
        tensor = point_collection.get_points_and_normals_tensor()
        if tensor.dim() != 3 or tensor.size(1) != 2:
            raise ValueError("Each tensor must be of shape (N, 2, 2) where N is the number of points.")
        
        pointx = tensor[:, 0, 0].detach().cpu().numpy()
        pointy = tensor[:,0, 1].detach().cpu().numpy()
        normalx = tensor[:, 1, 0].detach().cpu().numpy()
        normaly = tensor[:, 1, 1].detach().cpu().numpy()
        # Plot the points # plot_2d_projection.counter % 20
        if ax == None:
            plt.scatter(pointx, pointy, color=colours(i), label=f'Tensor {i+1}',s=2)
        else:
            ax.scatter(pointx, pointy, color=colours(i), label=f'Tensor {i+1}',s=2)
            
        if plot_normals:
            scale = 0.2
            for px,py,nx,ny in zip(pointx,pointy,normalx,normaly):
                if ax == None:
                    plt.arrow(px, py, nx*scale, ny*scale, head_width=0.1, head_length=0.1, fc=colours(i), ec=colours(i))
                else:
                    ax.arrow(px, py, nx*scale, ny*scale, head_width=0.1, head_length=0.1, fc=colours(i), ec=colours(i))
    
    if ax == None:
        plt.xlim((-1.5,1.5))
        plt.ylim((2.0,-2.0))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()