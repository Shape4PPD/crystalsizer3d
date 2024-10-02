from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from crystalsizer3d.nn.manager import Manager


def tile_image(image: Tensor, n_tiles: int, overlap: float = 0.):
    """
    Split an image into a square grid of patches.
    """
    assert int(np.sqrt(n_tiles))**2 == n_tiles, 'n_tiles must be a square number.'
    img_size = image.shape[-1]
    n_patches_per_side = int(np.sqrt(n_tiles))
    overlap = round(overlap * img_size)
    patch_size = (img_size + overlap * (n_patches_per_side - 1)) // n_patches_per_side
    patches = []
    patch_positions = []
    patch_coords = []

    for i in range(n_patches_per_side):
        start_x = round(i * (patch_size - overlap))
        start_x = max(0, min(start_x, img_size - patch_size))
        end_x = start_x + patch_size

        for j in range(n_patches_per_side):
            start_y = round(j * (patch_size - overlap))
            start_y = max(0, min(start_y, img_size - patch_size))
            end_y = start_y + patch_size

            patch = image[:, start_x:end_x, start_y:end_y]
            patches.append(patch)
            patch_positions.append((i, j))
            patch_coords.append((start_x, start_y, end_x, end_y))

    return torch.stack(patches), patch_positions


def stitch_image(patches: Tensor, patch_positions: List[Tuple[int, int]], overlap: float = 0.):
    """
    Stitch patches back together into a single image.
    """
    n_patches_per_side = int(np.sqrt(len(patch_positions)))
    patch_size = patches.shape[-1]
    img_size = round(n_patches_per_side * patch_size / (1 + (n_patches_per_side - 1) * overlap))
    overlap = round(overlap * img_size)
    stitched_image = torch.zeros((patches.shape[1], img_size, img_size), device=patches.device)
    full_weights = torch.zeros(img_size, img_size, device=patches.device)

    for patch, (i, j) in zip(patches, patch_positions):
        start_x = round(i * (patch_size - overlap))
        start_x = max(0, min(start_x, img_size - patch_size))
        end_x = start_x + patch_size

        start_y = round(j * (patch_size - overlap))
        start_y = max(0, min(start_y, img_size - patch_size))
        end_y = start_y + patch_size

        # Blend the patch into the stitched image
        weights = torch.ones(patch_size, patch_size, device=patches.device)
        for k in range(overlap):
            f = (k + 1) / (overlap + 1)
            if i > 0:
                weights[k, :] *= f
            if j > 0:
                weights[:, k] *= f
            if i < n_patches_per_side - 1:
                weights[-(k + 1), :] *= f
            if j < n_patches_per_side - 1:
                weights[:, -(k + 1)] *= f

        existing_weights = full_weights[start_x:end_x, start_y:end_y]
        weights_adj = torch.clamp(existing_weights + weights - 1, min=0)
        weights = weights - weights_adj
        stitched_image[:, start_x:end_x, start_y:end_y] += patch * weights
        full_weights[start_x:end_x, start_y:end_y] += weights

    # Check that the blending weights all sum to 1
    assert torch.allclose(full_weights, torch.ones_like(full_weights))

    return stitched_image


@torch.no_grad()
def denoise_batch(manager: Manager, X: Tensor, batch_size: int = -1):
    """
    Helper function to denoise a batch of images (or patches).
    """
    n_batches = (X.shape[0] + batch_size - 1) // batch_size
    X_denoised = []
    for i in range(n_batches):
        X_ = X[i * batch_size:(i + 1) * batch_size] if batch_size > 0 else X
        X_denoised.append(manager.denoise(X_))
    return torch.cat(X_denoised, dim=0)


def denoise_image(
        manager: Manager,
        X: Tensor,
        n_tiles: int = 1,
        overlap: float = 0.,
        batch_size: int = -1,
        return_patches: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, List[Tuple[int, int]]]]:
    """
    Denoise the image by splitting it into patches, denoising the patches, and stitching them back together.
    """
    assert int(np.sqrt(n_tiles))**2 == n_tiles, 'N must be a square number.'
    assert 0 <= overlap < 1, 'Overlap must be in [0, 1).'
    assert X.shape[-1] == X.shape[-2], 'Image must be square.'

    # Split it up into patches
    X_patches, patch_positions = tile_image(X, n_tiles=n_tiles, overlap=overlap)

    # Resize the patches to the working image size
    X_patches = F.interpolate(
        X_patches,
        size=manager.image_shape[-1],
        mode='bilinear',
        align_corners=False
    ).to(manager.device)

    # Denoise the patches
    X_patches_denoised = denoise_batch(manager, X_patches, batch_size=batch_size)

    # Plot the reconstituted image from the patches
    X_stitched = stitch_image(X_patches_denoised, patch_positions, overlap=overlap)

    # Resize the reconstituted denoised image to the working image size
    X_denoised = F.interpolate(
        X_stitched[None, ...],
        size=manager.image_shape[-1],
        mode='bilinear',
        align_corners=False
    )[0]

    if return_patches:
        return X_denoised, X_patches, X_patches_denoised, patch_positions

    return X_denoised
