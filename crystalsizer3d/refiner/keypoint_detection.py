from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import peak_local_max
from torch import Tensor
from torchvision.transforms.functional import gaussian_blur

from crystalsizer3d import N_WORKERS, logger
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.util.geometry import merge_vertices
from crystalsizer3d.util.utils import to_numpy

# Resize arguments used for interpolation
resize_args = dict(mode='bilinear', align_corners=False)


def to_relative_coordinates(coords: Tensor, image_size: int) -> Tensor:
    """
    Convert absolute coordinates to relative coordinates.
    """
    return torch.stack([
        (coords[:, 0] / image_size - 0.5) * 2,
        (0.5 - coords[:, 1] / image_size) * 2
    ], dim=1)


def to_absolute_coordinates(coords: Tensor, image_size: int) -> Tensor:
    """
    Convert relative coordinates to absolute coordinates.
    """
    return torch.stack([
        (coords[:, 0] * 0.5 + 0.5) * image_size,
        (0.5 - coords[:, 1] * 0.5) * image_size
    ], dim=1)


def calculate_keypoint_heatmaps(
        X: Tensor,
        manager: Manager,
        oversize_input: bool = False,
        max_img_size: int = 512,
        batch_size: int = -1,
) -> Tensor:
    """
    Calculate the keypoint heatmaps for a batch of images.
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

    # Detect the keypoints
    X_kp = detect_keypoints_batch(manager, X, batch_size=batch_size)

    # Restore the size if needed
    if restore_size:
        X_kp = F.interpolate(X_kp[:, None], size=restore_size, **resize_args)[:, 0]

    # Restore the resize input setting
    manager.keypoint_detector_args.kd_resize_input = resize_input_old

    if squeeze_batch_dim:
        X_kp = X_kp[0]

    return X_kp


def get_keypoint_coordinates(
        X_kp: np.ndarray | Tensor,
        min_distance: int = 1,
        threshold: float = 0,
        exclude_border: float = 0,
        n_workers: int = N_WORKERS,
) -> Tensor | List[Tensor]:
    """
    Find the peaks in the keypoint heatmaps.
    """
    if isinstance(X_kp, Tensor):
        X_kp = to_numpy(X_kp)
    if X_kp.ndim == 3:
        args = (min_distance, threshold, exclude_border, 0)
        if n_workers > 1:
            with Pool(n_workers) as p:
                return p.starmap(
                    get_keypoint_coordinates,
                    [(X_kp_i, *args) for X_kp_i in X_kp]
                )
        else:
            return [get_keypoint_coordinates(X_kp_i, *args) for X_kp_i in X_kp]
    assert X_kp.ndim == 2, 'Input must have shape (H, W).'
    if exclude_border > 0:
        exclude_border = round(exclude_border * X_kp.shape[-1])
    else:
        exclude_border = False
    coords = peak_local_max(
        X_kp,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=exclude_border,
    )
    coords = coords[:, ::-1].copy()  # Flip the coordinates to (y, x) to match the projector

    return torch.from_numpy(coords)


def generate_attention_patches(
        X: Tensor,
        X_kp: Tensor,
        patch_search_res: int = 256,
        n_patches: int = 16,
        patch_size: int = 512,
        attenuation_sigma: float = 0.5,
        max_attenuation_factor: float = 1.5,
) -> Tuple[Tensor, Tensor]:
    """
    Use the keypoint heatmap to select the best patches to focus in on.
    """
    if X.ndim == 3:
        X = X[None, ...]
    assert X.ndim == 4 and X.shape[1] in [1, 3], 'X must be a 4D tensor ([B,] C, H, W).'
    assert X_kp.ndim == 2, 'X_kp must be a 2D tensor (H, W).'
    assert X_kp.shape == X.shape[-2:], 'X_kp must have the same (H, W) shape as X.'

    # Resize the keypoint heatmap to the patch search resolution
    X_kp = X_kp.cpu().clone()[None, None, ...]
    sf = 1
    if X_kp.shape[-1] > patch_search_res:
        sf = X_kp.shape[-1] / patch_search_res
        X_kp = F.interpolate(X_kp, size=patch_search_res, **resize_args)

    # Sizes
    patch_size = min(patch_size, X.shape[-1] - 20)  # Ensure the patch size is not too large
    wis = X_kp.shape[-1]
    ps = round(patch_size / sf)
    ps2 = round(ps / 2)
    ps2_full = round(patch_size / 2)

    # Create a Gaussian blob to reduce the heatmap around the selected patch
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, ps2 * 2),
        torch.linspace(-1, 1, ps2 * 2),
        indexing='ij'
    )
    gaussian_blob = torch.exp(-(xx**2 + yy**2) / (2 * attenuation_sigma**2))
    focal_attenuator = 1 + (gaussian_blob * (max_attenuation_factor - 1)
                            - (gaussian_blob * (max_attenuation_factor - 1)).min())
    focal_attenuator = focal_attenuator[None, None, ...]

    # Generate patches iteratively
    patch_centres = []
    X_patches = []
    for i in range(n_patches):
        # Calculate the heatmap sum for each possible patch position
        patch_sums = F.conv2d(X_kp, focal_attenuator, padding=ps2).squeeze()
        if patch_sums.shape[-1] != wis:
            patch_sums = F.interpolate(patch_sums[None, None, ...], size=wis, **resize_args).squeeze()

        # Find the index of the maximum error sum
        max_idx = torch.argmax(patch_sums).item()
        y = max_idx // wis
        x = max_idx % wis
        y = min(max(ps2, y), wis - ps2)
        x = min(max(ps2, x), wis - ps2)

        # Reduce the heatmap around the selected patch
        X_kp[..., y - ps2:y + ps2, x - ps2:x + ps2] /= focal_attenuator

        # Scale the coordinates back to the original image size
        y = min(max(ps2_full, round(y * sf)), X.shape[-1] - ps2_full)
        x = min(max(ps2_full, round(x * sf)), X.shape[-1] - ps2_full)
        centre = torch.tensor([x, y])
        patch_centres.append(centre)

        # Crop patch from the original
        X_patch = X[..., y - ps2_full:y + ps2_full, x - ps2_full:x + ps2_full]
        X_patches.append(X_patch)

    return torch.stack(X_patches, dim=1), torch.stack(patch_centres)


@torch.no_grad()
def find_keypoints(
        X_target: Tensor,
        X_target_denoised: Tensor,
        manager: Manager,
        oversize_input: bool = True,
        max_img_size: int = 512,
        batch_size: int = 512,
        min_distance: int = 1,
        threshold: float = 0,
        exclude_border: float = 0,
        blur_kernel_relative_size: float = 0.01,
        n_patches: int = 16,
        patch_size: int = 512,
        patch_search_res: int = 256,
        attenuation_sigma: float = 0.5,
        max_attenuation_factor: float = 1.5,
        low_res_catchment_distance: int = 100,
        return_everything: bool = False,
        quiet: bool = False,
) -> Tensor | Dict[str, Tensor]:
    """
    Find the keypoints using a targeted approach.
    """
    heatmap_args = dict(
        manager=manager,
        oversize_input=oversize_input,
        max_img_size=max_img_size,
        batch_size=batch_size,
    )
    coords_args = dict(
        min_distance=min_distance,
        threshold=threshold,
        exclude_border=exclude_border,
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

    # Check the input shapes and ensure (C, H, W) format
    assert X_target.ndim == 3 and X_target_denoised.ndim == 3, 'Input images must have 3 dimensions.'
    if X_target.shape[-1] == 3:
        X_target = X_target.permute(2, 0, 1)
    if X_target_denoised.shape[-1] == 3:
        X_target_denoised = X_target_denoised.permute(2, 0, 1)
    assert X_target.shape == X_target_denoised.shape, 'Input images must have the same shape.'
    image_size = X_target.shape[-1]

    # First we take the denoised image, blur it and detect keypoints
    quiet_log('Detecting initial keypoints in the blurred denoised image.')
    ks = round(image_size * blur_kernel_relative_size)
    if ks % 2 == 0:
        ks += 1
    X_lr = gaussian_blur(X_target_denoised, kernel_size=[ks, ks])
    X_lr_kp = calculate_keypoint_heatmaps(X=X_lr, **heatmap_args)
    Y_lr = get_keypoint_coordinates(X_lr_kp, **coords_args)

    # Combine the original and denoised images into a batch
    X_combined = torch.stack([X_target, X_target_denoised], dim=0)

    # Use the keypoint heatmap from the low-res image to select patches to focus on
    quiet_log('Generating patches to focus in on the low-res keypoints.')
    X_patches_combined, patch_centres = generate_attention_patches(X_combined, X_lr_kp, **patch_args)
    X_patches, X_patches_dn = X_patches_combined

    # Calculate keypoint heatmaps in the high-res patches
    quiet_log('Calculating keypoint heatmaps in the high-res patches.')
    X_patches_combined_kp = calculate_keypoint_heatmaps(
        X=X_patches_combined.reshape(-1, *X_patches_combined.shape[2:]),
        **heatmap_args
    )
    X_patches_kp, X_patches_dn_kp = X_patches_combined_kp.reshape(
        *X_patches_combined.shape[:2], *X_patches_combined_kp.shape[-2:]
    )

    # Extract the keypoint coordinates from the heatmaps
    quiet_log('Extracting keypoint coordinates from the heatmaps.')
    Y_patches_combined = get_keypoint_coordinates(X_kp=X_patches_combined_kp, **coords_args)
    Y_patches = Y_patches_combined[:n_patches]
    Y_patches_dn = Y_patches_combined[n_patches:]

    # Collate the keypoints from all the patches together
    Y_candidates_all = []
    for Yp, pc in zip(Y_patches, patch_centres):
        Y_candidates_all.append(Yp + pc[None, :] - patch_size // 2)
    for Yp, pc in zip(Y_patches_dn, patch_centres):
        Y_candidates_all.append(Yp + pc[None, :] - patch_size // 2)
    Y_candidates_all = torch.cat(Y_candidates_all, axis=0)

    # Merge nearby keypoints
    quiet_log('Merging nearby keypoints.')
    Y_candidates_merged, _ = merge_vertices(
        Y_candidates_all.to(torch.float32), epsilon=min_distance
    )

    # Discard any keypoints which are too far from any detected in the low-res image
    quiet_log('Discarding keypoints which are too far from the low-res keypoints.')
    Ylr_Yc_dist = torch.cdist(Y_lr.to(torch.float32), Y_candidates_merged)
    Y_candidates_final = Y_candidates_merged[Ylr_Yc_dist.amin(dim=0) < low_res_catchment_distance]
    Y_candidates_final_rel = to_relative_coordinates(Y_candidates_final, image_size)

    if return_everything:
        return {
            'X_lr': X_lr,
            'X_lr_kp': X_lr_kp,
            'Y_lr': Y_lr,
            'X_patches': X_patches,
            'X_patches_dn': X_patches_dn,
            'X_patches_kp': X_patches_kp,
            'X_patches_dn_kp': X_patches_dn_kp,
            'Y_patches': Y_patches,
            'Y_patches_dn': Y_patches_dn,
            'Y_candidates_all': Y_candidates_all,
            'Y_candidates_merged': Y_candidates_merged,
            'Y_candidates_final': Y_candidates_final,
            'Y_candidates_final_rel': Y_candidates_final_rel,
        }

    return Y_candidates_final_rel


@torch.no_grad()
def detect_keypoints_batch(manager: Manager, X: Tensor, batch_size: int = -1) -> Tensor:
    """
    Helper function to detect keypoints in a batch of images (or patches).
    """
    n_batches = (X.shape[0] + batch_size - 1) // batch_size
    X_kp = []
    for i in range(n_batches):
        X_ = X[i * batch_size:(i + 1) * batch_size] if batch_size > 0 else X
        X_kp.append(manager.detect_keypoints(X_)[0])
    return torch.cat(X_kp, dim=0)
