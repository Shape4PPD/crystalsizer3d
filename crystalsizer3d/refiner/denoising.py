from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.refiner.tiling import stitch_image, tile_image

# Resize arguments used for interpolation
resize_args = dict(mode='bilinear', align_corners=False)


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


@torch.no_grad()
def denoise_image(
        manager: Manager,
        X: Tensor,
        n_tiles: int = 1,
        overlap: float = 0.,
        oversize_input: bool = False,
        max_img_size: int = 1024,
        batch_size: int = -1,
        return_patches: bool = False
) -> Tensor | Tuple[Tensor, Tensor, Tensor, List[Tuple[int, int]]]:
    """
    Denoise the image by splitting it into patches, denoising the patches, and stitching them back together.
    """
    assert int(np.sqrt(n_tiles))**2 == n_tiles, 'N must be a square number.'
    assert 0 <= overlap < 1, 'Overlap must be in [0, 1).'
    assert X.shape[-1] == X.shape[-2], 'Image must be square.'

    # Set up the resizing
    resize_input_old = manager.keypoint_detector_args.dn_resize_input
    manager.keypoint_detector_args.dn_resize_input = False
    img_size = max_img_size if oversize_input else manager.image_shape[-1]

    # Split it up into patches
    X_patches, patch_positions = tile_image(X, n_tiles=n_tiles, overlap=overlap)

    # Resize the patches to the input image size
    if oversize_input and X_patches.shape[-1] > img_size \
            or not oversize_input and X_patches.shape[-1] != img_size:
        X_patches = F.interpolate(X_patches, size=img_size, **resize_args)

    # Denoise the patches
    X_patches_denoised = denoise_batch(manager, X_patches.to(manager.device), batch_size=batch_size)

    # Stitch the image back together from the patches
    X_stitched = stitch_image(X_patches_denoised, patch_positions, overlap=overlap)

    # Resize the reconstituted denoised image to the input image size
    X_denoised = F.interpolate(X_stitched[None, ...], size=X.shape[-1], **resize_args)[0]

    # Reset the resizing
    manager.keypoint_detector_args.dn_resize_input = resize_input_old

    if return_patches:
        return X_denoised, X_patches, X_patches_denoised, patch_positions

    return X_denoised
