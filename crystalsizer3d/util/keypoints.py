import torch
from torch import Tensor


def generate_keypoints_heatmap(
        keypoints: Tensor,
        image_size: int,
        blob_variance: float = 10.0
) -> Tensor:
    """
    Generates a heatmap with Gaussian blobs at each keypoint location.

    Args:
        keypoints (Tensor): Tensor of shape (N, 2) containing keypoint coordinates (x, y).
        image_size (int): Size of the square image (image will be image_size x image_size).
        blob_variance (float): Variance of the Gaussian blobs.

    Returns:
        heatmap (Tensor): Generated heatmap of shape (image_size, image_size).
    """
    device = keypoints.device
    heatmap = torch.zeros((image_size, image_size), device=device)

    # Create a grid of coordinates (image_size x image_size)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(image_size, device=device),
        torch.arange(image_size, device=device),
        indexing='ij'
    )
    y_coords = y_coords.float()
    x_coords = x_coords.float()

    # Iterate over each keypoint to add a Gaussian blob
    for keypoint in keypoints:
        x, y = keypoint

        # Compute the squared distance from the keypoint to every pixel in the image
        dist_squared = (x_coords - x)**2 + (y_coords - y)**2

        # Generate a Gaussian blob at this location
        gaussian_blob = torch.exp(-dist_squared / (2 * blob_variance))

        # Update the heatmap using max to avoid exceeding the blob height when blobs overlap
        heatmap = torch.max(heatmap, gaussian_blob)

    return heatmap
