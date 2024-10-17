from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


def tile_image(image: Tensor, n_tiles: int, overlap: float = 0.):
    """
    Split an image into a square grid of patches.
    """
    squeeze_channel_dim = False
    squeeze_batch_dim = False
    if image.ndim == 2:
        image = image[None, ...]
        squeeze_channel_dim = True
    if image.ndim == 3:
        image = image[None, ...]
        squeeze_batch_dim = True
    assert image.ndim == 4 and image.shape[1] in [1, 3], 'Image must be ([B,] [C,] H, W).'
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

            patch = image[:, :, start_x:end_x, start_y:end_y]
            patches.append(patch)
            patch_positions.append((i, j))
            patch_coords.append((start_x, start_y, end_x, end_y))

    patches = torch.stack(patches).permute(1, 0, 2, 3, 4)  # (B, N, C, H, W)
    if squeeze_batch_dim:
        patches = patches.squeeze(0)
    if squeeze_channel_dim:
        patches = patches.squeeze(1)

    return patches, patch_positions


def stitch_image(patches: Tensor, patch_positions: List[Tuple[int, int]], overlap: float = 0.):
    """
    Stitch patches back together into a single image.
    """
    squeeze_batch_dim = False
    if patches.ndim == 4:
        patches = patches[None, ...]
        squeeze_batch_dim = True
    assert patches.ndim == 5 and patches.shape[2] in [1, 3], 'Patches must be ([B,] N, C, H, W).'
    patches = patches.permute(1, 0, 2, 3, 4)  # (N, B, C, H, W)
    n_patches_per_side = int(np.sqrt(len(patch_positions)))
    patch_size = patches.shape[-1]
    img_size = round(n_patches_per_side * patch_size / (1 + (n_patches_per_side - 1) * overlap))
    overlap = round(overlap * img_size)
    stitched_images = torch.zeros(
        (patches.shape[1], patches.shape[2], img_size, img_size),  # (B, C, H, W)
        device=patches.device
    )
    full_weights = torch.zeros(img_size, img_size, device=patches.device)  # (H, W)

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
        stitched_images[:, :, start_x:end_x, start_y:end_y] += patch * weights
        full_weights[start_x:end_x, start_y:end_y] += weights

    # Check that the blending weights all sum to 1
    assert torch.allclose(full_weights, torch.ones_like(full_weights))

    if squeeze_batch_dim:
        stitched_images = stitched_images.squeeze(0)

    return stitched_images
