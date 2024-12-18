from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from crystalsizer3d import logger
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.refiner.keypoint_detection import generate_attention_patches

# Resize arguments used for interpolation
resize_args = dict(mode='bilinear', align_corners=False)


def calculate_edge_heatmaps(
        X: Tensor,
        manager: Manager,
        oversize_input: bool = False,
        max_img_size: int = 512,
        batch_size: int = -1,
) -> Tensor:
    """
    Calculate the edge heatmaps for a batch of images.
    """
    squeeze_batch_dim = False
    if X.ndim == 3:
        X = X[None, ...]
        squeeze_batch_dim = True

    # Set up the resizing
    resize_input_old = manager.keypoint_detector_args.kd_resize_input
    manager.keypoint_detector_args.kd_resize_input = False
    img_size = max_img_size if oversize_input else manager.image_shape[-1]

    # Resize the input if needed
    restore_size = False
    if oversize_input and X.shape[-1] > img_size \
            or not oversize_input and X.shape[-1] != img_size:
        restore_size = X.shape[-1]
        X = F.interpolate(X, size=img_size, **resize_args)

    # Detect the edges
    X_wf = detect_edges_batch(manager, X, batch_size=batch_size)

    # Restore the size if needed
    if restore_size:
        X_wf = F.interpolate(X_wf[:, None], size=restore_size, **resize_args)[:, 0]

    # Restore the resize input setting
    manager.keypoint_detector_args.kd_resize_input = resize_input_old

    if squeeze_batch_dim:
        X_wf = X_wf[0]

    return X_wf


@torch.no_grad()
def find_edges(
        X_target: Tensor,
        X_target_denoised: Tensor,
        manager: Manager,
        oversize_input: bool = True,
        max_img_size: int = 512,
        batch_size: int = 4,
        threshold: float = 0,
        exclude_border: float = 0,
        n_patches: int = 16,
        patch_size: int = 512,
        patch_search_res: int = 256,
        attenuation_sigma: float = 0.5,
        max_attenuation_factor: float = 1.5,
        return_everything: bool = False,
        quiet: bool = False,
) -> Tensor | Dict[str, Tensor]:
    """
    Find the edges using a targeted approach.
    """
    X_target = X_target.cpu()
    X_target_denoised = X_target_denoised.cpu()
    image_size = X_target.shape[-1]
    heatmap_args = dict(
        manager=manager,
        oversize_input=oversize_input,
        max_img_size=max_img_size,
        batch_size=batch_size,
    )
    patch_args = dict(
        patch_search_res=patch_search_res,
        n_patches=n_patches,
        patch_size=patch_size,
        attenuation_sigma=attenuation_sigma,
        max_attenuation_factor=max_attenuation_factor
    )

    def quiet_log(msg):
        if not quiet:
            logger.info(msg)

    def zero_border(X_):
        if exclude_border > 0:
            exclude_border_px = round(exclude_border * image_size)
            X_[:exclude_border_px] = 0
            X_[-exclude_border_px:] = 0
            X_[:, :exclude_border_px] = 0
            X_[:, -exclude_border_px:] = 0

    # Check the input shapes and ensure (C, H, W) format
    assert X_target.ndim == 3 and X_target_denoised.ndim == 3, 'Input images must have 3 dimensions.'
    if X_target.shape[-1] == 3:
        X_target = X_target.permute(2, 0, 1)
    if X_target_denoised.shape[-1] == 3:
        X_target_denoised = X_target_denoised.permute(2, 0, 1)
    assert X_target.shape == X_target_denoised.shape, 'Input images must have the same shape.'

    # First we detect edges on the full size original
    quiet_log('Detecting initial edges in the original image.')
    X_lr_wf = calculate_edge_heatmaps(X=X_target, **heatmap_args)
    zero_border(X_lr_wf)

    # Combine the original and denoised images into a batch
    X_combined = torch.stack([X_target, X_target_denoised], dim=0)

    # Use the edge heatmap from the low-res image to select patches to focus on
    quiet_log('Generating patches to focus in on the low-res edge image.')
    X_patches_combined, patch_centres = generate_attention_patches(X_combined, X_lr_wf, **patch_args)
    X_patches, X_patches_dn = X_patches_combined

    # Calculate edge heatmaps in the high-res patches
    quiet_log('Calculating edge heatmaps in the high-res patches.')
    X_patches_combined_wf = calculate_edge_heatmaps(
        X=X_patches_combined.reshape(-1, *X_patches_combined.shape[2:]),
        **heatmap_args
    )
    X_patches_wf, X_patches_dn_wf = X_patches_combined_wf.reshape(
        *X_patches_combined.shape[:2], *X_patches_combined_wf.shape[-2:]
    )

    # Combine the edges from all the patches together
    X_combined_wf = torch.zeros(image_size, image_size)
    pixel_counts = torch.zeros_like(X_combined_wf)
    for i, (x, y) in enumerate(patch_centres):
        x0, x1 = x - patch_size // 2, x + patch_size // 2
        y0, y1 = y - patch_size // 2, y + patch_size // 2
        for patch in [X_patches_wf[i].clone().cpu(), X_patches_dn_wf[i].clone().cpu()]:
            counts = torch.ones_like(patch)
            patch[patch < threshold] = 0
            counts[patch < threshold] = 0
            X_combined_wf[y0:y1, x0:x1] += patch
            pixel_counts[y0:y1, x0:x1] += counts
    X_combined_wf = torch.where(
        pixel_counts > 0,
        X_combined_wf / pixel_counts,
        torch.zeros_like(X_combined_wf)
    )
    zero_border(X_combined_wf)

    if return_everything:
        return {
            'X_lr_wf': X_lr_wf,
            'X_patches': X_patches,
            'X_patches_dn': X_patches_dn,
            'X_patches_wf': X_patches_wf,
            'X_patches_dn_wf': X_patches_dn_wf,
            'X_combined_wf': X_combined_wf,
        }

    return X_combined_wf


@torch.no_grad()
def detect_edges_batch(manager: Manager, X: Tensor, batch_size: int = -1) -> Tensor:
    """
    Helper function to detect edges in a batch of images (or patches).
    """
    n_batches = (X.shape[0] + batch_size - 1) // batch_size
    X_wf = []
    for i in range(n_batches):
        X_ = X[i * batch_size:(i + 1) * batch_size] if batch_size > 0 else X
        _, X_wf_i = manager.detect_keypoints(X_)
        X_wf_i = X_wf_i.sum(dim=1).clamp(0, 1)
        X_wf.append(X_wf_i)
    return torch.cat(X_wf, dim=0)
