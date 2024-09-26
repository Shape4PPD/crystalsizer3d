import os
import shutil
import time
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.draw import line_aa
from skimage.filters import gaussian

import crystalsizer3d

crystalsizer3d.USE_CUDA = False  # Ensure we only use CPU for this script

from crystalsizer3d import logger
from crystalsizer3d.projector import Projector
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.util.keypoints import generate_keypoints_heatmap
from crystalsizer3d.util.utils import print_args, set_seed, str2bool, to_numpy
from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs
from crystalsizer3d.nn.dataset import Dataset


def parse_args(printout: bool = True) -> Namespace:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Generate the keypoints for an existing synthetic dataset.')
    parser.add_argument('--ds-path', type=str, required=True,
                        help='Set the path to the existing dataset.')
    parser.add_argument('--overwrite-existing', type=str2bool, default=False,
                        help='Overwrite existing keypoints if they exists, otherwise try to resume.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Set a random seed.')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Set the number of parallel workers to use.')
    parser.add_argument('--batch-size', type=int, default=15,
                        help='Set the batch size for generating keypoints images.')
    parser.add_argument('--heatmap-blob-variance', type=float, default=10.0,
                        help='Variance of the Gaussian blobs in the keypoint heatmap.')
    parser.add_argument('--wireframe-blur-variance', type=float, default=1.0,
                        help='Variance of the Gaussian blur applied to the wireframe images.')

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    return args


@torch.no_grad()
def _generate_keypoints_image(
        idx: int,
        c_params: Dict[str, Any],
        projector: Projector,
        scene: Scene,
        dst_args: DatasetTrainingArgs,
        save_dir: Path
):
    """
    Generate a projected wireframe image with keypoints heatmap.
    3 channel image output: 1st channel is the front-facing wireframe edges, 2nd channel is the back-facing wireframe
    edges, 3rd channel is the keypoints heatmap.
    """
    # Set the crystal parameters
    projector.crystal.scale.data = torch.tensor(c_params['scale'])
    projector.crystal.distances.data = torch.tensor(c_params['distances'])
    projector.crystal.origin.data = torch.tensor(c_params['origin'])
    projector.crystal.rotation.data = torch.tensor(c_params['rotation'])
    projector.crystal.material_ior.data = torch.tensor(c_params['material_ior'])

    # Rebuild the crystal mesh
    projector.crystal.build_mesh(update_uv_map=False)

    # Recalculate the projector zoom based on the crystal vertices and reproject the mesh
    projector.update_zoom(orthographic_scale_factor(
        scene, z=projector.crystal.vertices[:, 2].mean().item()
    ))
    projector.project(generate_image=False)

    # Create the blank image with 3 channels, 2 for the wireframe and 1 for the keypoints
    h, w = projector.image_size.tolist()
    assert h == w
    image = np.zeros((3, h, w))

    # Draw the wireframe edges
    for ref_face_idx, face_segments in projector.edge_segments.items():
        if len(face_segments) == 0:
            continue
        for segment in face_segments:
            segment_clamped = segment.clone()
            segment_clamped[:, 0] = torch.clamp(segment_clamped[:, 0], 0, w - 1)
            segment_clamped[:, 1] = torch.clamp(segment_clamped[:, 1], 0, h - 1)
            channel = 0 if ref_face_idx == 'facing' else 1
            points = segment_clamped.round().to(torch.uint32).tolist()
            rr, cc, val = line_aa(points[0][1], points[0][0], points[1][1], points[1][0])
            image[channel, rr, cc] = np.clip(val, 0, 1)

    # Apply a gaussian blur and then re-normalise to "fatten" the midline
    if dst_args.wireframe_blur_variance > 0:
        image = gaussian(image, sigma=dst_args.wireframe_blur_variance, channel_axis=0)

        # Normalise to [0-1] with float32 dtype
        image_range = image.max() - image.min()
        if image_range > 0:
            image = (image - image.min()) / image_range

    # Create the keypoints heatmap
    heatmap = generate_keypoints_heatmap(
        keypoints=projector.keypoints,
        image_size=h,
        blob_variance=dst_args.heatmap_blob_variance
    )
    heatmap = heatmap.clamp(min=0)
    heatmap = heatmap / heatmap.max()
    image[2] = to_numpy(heatmap)

    # Convert to uint8 and transpose to (H, W, C)
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    image = image.transpose(1, 2, 0)

    # Save the keypoints image
    keypoints_path = save_dir / f'{idx:010d}.png'
    Image.fromarray(image).save(keypoints_path)


def _generate_keypoints_images_batch(
        batch_idx: int,
        idxs_batch: List[int],
        c_params_batch: List[Dict[str, Any]],
        n_batches: int,
        ds: Dataset,
        dst_args: DatasetTrainingArgs,
        save_dir: Path
):
    """
    Generate a batch of keypoints images.
    """
    assert len(idxs_batch) == len(c_params_batch)
    process_name = current_process().name
    if process_name == 'MainProcess':
        worker_id = 'Worker'
    else:
        worker_idx = int(process_name.split('-')[-1])
        worker_id = f'Worker-{worker_idx}'
    logger.info(f'{worker_id} starting on batch {batch_idx + 1}/{n_batches}.')

    # Load the first crystal to use as the template, with grad turned off
    r_params = ds.data[0]['rendering_parameters']
    crystal = ds.load_crystal(r_params=r_params)
    for param in crystal.parameters():
        param.requires_grad = False

    # Instantiate the scene, doesn't need a crystal, just used for calculating the orthographic scale factor
    scene = Scene(
        res=ds.dataset_args.image_size,
        **ds.dataset_args.to_dict(),
    )

    # Instantiate the projector
    projector = Projector(
        crystal,
        image_size=(ds.dataset_args.image_size, ds.dataset_args.image_size),
        zoom=orthographic_scale_factor(scene, z=crystal.vertices[:, 2].mean().item()),
        multi_line=True
    )

    # Loop over the idxs and generate the keypoints images
    for i, (idx, c_params) in enumerate(zip(idxs_batch, c_params_batch)):
        if (i + 1) % 10 == 0:
            logger.info(f'{worker_id} at {i + 1}/{len(idxs_batch)} (batch {batch_idx + 1}/{n_batches}).')
        _generate_keypoints_image(
            idx=idx,
            c_params=c_params,
            projector=projector,
            scene=scene,
            dst_args=dst_args,
            save_dir=save_dir
        )
    logger.info(f'{worker_id} finished on batch {batch_idx + 1}/{n_batches}.')


def _generate_keypoints_images_batch_wrapper(args: Tuple):
    """
    Wrapper for generating keypoints images in parallel.
    """
    return _generate_keypoints_images_batch(*args)


def generate_keypoints():
    """
    Generate keypoint images for an existing dataset of synthetic crystal images.
    """
    runtime_args = parse_args()
    set_seed(runtime_args.seed)

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Load the dataset
    dst_args = DatasetTrainingArgs(
        dataset_path=runtime_args.ds_path,
        train_keypoint_detector=True,
        train_edge_detector=True,
        heatmap_blob_variance=runtime_args.heatmap_blob_variance,
        wireframe_blur_variance=runtime_args.wireframe_blur_variance
    )
    ds = Dataset(dst_args=dst_args)
    keypoints_dir = ds.path / f'keypoints_wfv={dst_args.wireframe_blur_variance:.1f}_kpv={dst_args.heatmap_blob_variance:.1f}'

    # Remove existing keypoint images if required
    if runtime_args.overwrite_existing and keypoints_dir.exists():
        logger.info(f'Removing existing keypoints at {keypoints_dir}.')
        shutil.rmtree(keypoints_dir)

    # Ensure that the output directory exists
    keypoints_dir.mkdir(parents=True, exist_ok=True)

    # Loop over dataset and get all the idxs and crystal params that are missing keypoint images
    logger.info('Checking for existing keypoints images.')
    existing_images = {entry.name for entry in os.scandir(keypoints_dir) if entry.is_file()}
    idxs = [idx for idx in ds.data.keys() if ds.data[idx]['keypoints_image'].name not in existing_images]
    c_params = [ds.data[idx]['rendering_parameters']['crystal'] for idx in idxs]
    logger.info(f'Found {len(idxs)} images without keypoints.')

    # Split the idxs into batches
    batch_size = runtime_args.batch_size
    idx_batches = [idxs[i:i + batch_size] for i in range(0, len(idxs), batch_size)]
    c_params_batches = [c_params[i:i + batch_size] for i in range(0, len(c_params), batch_size)]

    # Build the shared args
    shared_args = (len(idx_batches), ds, dst_args, keypoints_dir)

    # Generate the keypoint images in parallel
    if runtime_args.n_workers > 1:
        logger.info(f'Generating keypoints images in parallel with {runtime_args.n_workers} workers.')
        with Pool(runtime_args.n_workers) as pool:
            pool.map(
                _generate_keypoints_images_batch_wrapper,
                [(i, idx_batch, c_params_batch, *shared_args)
                 for i, (idx_batch, c_params_batch) in enumerate(zip(idx_batches, c_params_batches))]
            )

    # Generate the keypoint images in serial
    else:
        logger.info('Generating keypoints images in serial.')
        for i, (idx_batch, c_params_batch) in enumerate(zip(idx_batches, c_params_batches)):
            _generate_keypoints_images_batch(i, idx_batch, c_params_batch, *shared_args)

    # Show how long this took, formatted nicely
    elapsed_time = time.time() - start_time
    logger.info(f'Finished in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s.')


if __name__ == '__main__':
    generate_keypoints()
