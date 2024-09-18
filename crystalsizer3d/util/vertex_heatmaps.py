import torch
from torch import Tensor


def generate_vertex_heatmap(
        vertices: Tensor,
        image_size: int,
        blob_height: float = 1.0,
        blob_variance: float = 10.0
) -> Tensor:
    """
    Generates a heatmap with Gaussian blobs at each vertex location.

    Args:
        vertices (torch.Tensor): Tensor of shape (N, 2) containing vertex coordinates (x, y).
        image_size (int): Size of the square image (image will be image_size x image_size).
        blob_height (float): The maximum height of the Gaussian blob.
        blob_variance (float): Variance (sigma^2) of the Gaussian blobs.

    Returns:
        heatmap (torch.Tensor): Generated heatmap of shape (image_size, image_size).
    """
    device = vertices.device
    heatmap = torch.zeros((image_size, image_size), device=device)

    # Create a grid of coordinates (image_size x image_size)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(image_size, device=device),
        torch.arange(image_size, device=device),
        indexing='ij'
    )
    y_coords = y_coords.float()
    x_coords = x_coords.float()

    # Iterate over each vertex to add a Gaussian blob
    for vertex in vertices:
        x, y = vertex

        # Compute the squared distance from the vertex to every pixel in the image
        dist_squared = (x_coords - x)**2 + (y_coords - y)**2

        # Generate a Gaussian blob at this vertex
        gaussian_blob = blob_height * torch.exp(-dist_squared / (2 * blob_variance))

        # Update the heatmap using max to avoid exceeding the blob height when blobs overlap
        heatmap = torch.max(heatmap, gaussian_blob)

    return heatmap
