import copy
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import shuffle
from typing import Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs
from crystalsizer3d.nn.dataset import Dataset
from crystalsizer3d.refiner.keypoint_detection import get_keypoint_coordinates
from crystalsizer3d.sequence.utils import get_image_paths
from crystalsizer3d.util.keypoints import generate_keypoints_heatmap
from crystalsizer3d.util.utils import print_args, set_seed, to_dict, to_numpy


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to plot some figure assets.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for the random number generator.')

    # Image sequence
    parser.add_argument('--images-dir', type=Path,
                        help='Directory containing the sequence of images.')
    parser.add_argument('--image-ext', type=str, default='jpg',
                        help='Image extension.')

    # Dataset
    parser.add_argument('--dataset-path', type=Path, help='Path to the dataset.')
    parser.add_argument('--start-image', type=int, default=0,
                        help='Start with this image.')
    parser.add_argument('--end-image', type=int, default=20,
                        help='End with this image.')

    # Stack args
    parser.add_argument('--res', type=int, default=2000,
                        help='Width and height of images in pixels.')
    parser.add_argument('--border-size', type=int, default=2,
                        help='Size of the border in pixels.')
    parser.add_argument('--border-colour', type=lambda s: (float(c) for c in s.split(',')), default=(0, 0, 0, 200),
                        help='RGBA colour of the border.')
    parser.add_argument('--n-images', type=int, default=33,
                        help='Number of images to show.')
    parser.add_argument('--img-spacing', type=int, default=20,
                        help='Spacing between images in pixels.')
    parser.add_argument('--chunk-spacing', type=int, default=80,
                        help='Spacing between chunks of images in pixels.')
    parser.add_argument('--show-first-n-images', type=int, default=4,
                        help='Number of images to show at the front.')
    parser.add_argument('--show-last-n-images', type=int, default=2,
                        help='Number of images to show at the back.')
    parser.add_argument('--n-dots', type=int, default=3,
                        help='Number of dots to show.')
    parser.add_argument('--dot-size', type=int, default=2,
                        help='Size of the dots in pixels.')
    parser.add_argument('--dot-colour', type=lambda s: (float(c) for c in s.split(',')), default=(0, 0, 0, 200),
                        help='RGBA colour of the dots.')
    parser.add_argument('--highlight-idx', type=int, default=None,
                        help='Highlight this image.')
    parser.add_argument('--highlight-offset', type=int, default=100,
                        help='Offset for the highlighted image.')
    parser.add_argument('--highlight-border-size', type=int, default=2,
                        help='Size of the border for the highlighted image.')

    args = parser.parse_args()

    # Set the random seed
    set_seed(args.seed)

    return args


def _init(load_ds: bool = True) -> Tuple[Namespace, Dataset | None, Path]:
    """
    Initialise the dataset and get the command line arguments.
    """
    args = get_args()
    print_args(args)

    # Write the args to the output dir
    output_dir = LOGS_PATH / START_TIMESTAMP
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'args.yml', 'w') as f:
        spec = to_dict(args)
        spec['created'] = START_TIMESTAMP
        yaml.dump(spec, f)

    # Initialise the dataset
    if load_ds:
        dst_args = DatasetTrainingArgs(
            dataset_path=args.dataset_path,
            train_keypoint_detector=True,
            wireframe_blur_variance=1.0,
            heatmap_blob_variance=10.0,
        )
        ds = Dataset(dst_args)
    else:
        ds = None

    return args, ds, output_dir


def _make_bordered_img(
        img: np.ndarray,
        border_size: int = 0,
        border_colour: np.ndarray = np.array([0, 0, 0])
) -> Image:
    """
    Make a blob image using colours from the look-up-table with a border as defined.
    """
    b_img = img.copy()
    if border_size > 0:
        b_img[:border_size, :] = border_colour
        b_img[-border_size:, :] = border_colour
        b_img[:, :border_size] = border_colour
        b_img[:, -border_size:] = border_colour
    b_img = Image.fromarray(b_img)
    return b_img


def _make_image_stack(
        images: np.ndarray,
        border_size: int = 1,
        border_colour: Tuple[int] = (0, 0, 0, 200),
        n_images: int = 33,
        img_spacing: int = 6,
        chunk_spacing: int = 40,
        show_first_n_images: int = 5,
        show_last_n_images: int = 1,
        n_dots: int = 3,
        dot_size: int = 2,
        dot_colour: Tuple[int] = (0, 0, 0, 200),
        centre_dots: bool = False,
        highlight_idx: int = None,
        highlight_offset: int = 100,
        highlight_border_size: int = 2,
        **kwargs
):
    """
    Plot stacks of rendered blobs for a midline.
    """
    N = len(images)
    h, w = int(images.shape[1]), int(images.shape[2])

    if border_colour is None:
        cmap = plt.get_cmap('jet')
        border_colours = cmap(np.linspace(0, 1, N)) * 255
    else:
        border_colours = np.ones((N, 4)) * border_colour
    if show_last_n_images == 0:
        chunk_spacing = 0

    # Get the idxs of the blobs to use
    if 0 < n_images < N:
        img_idxs = np.round(np.linspace(0, N - 1, n_images)).astype(int)
    else:
        img_idxs = range(N)

    # Calculate the offsets
    offsets_first = np.arange(show_first_n_images) * img_spacing
    offsets_last = np.arange(show_last_n_images) * img_spacing
    highlight_rhs = w + highlight_offset \
                    + int((show_first_n_images - 0.5) * img_spacing + chunk_spacing / 2)

    dim_y = h + (show_first_n_images + show_last_n_images - 1) * img_spacing + chunk_spacing
    if highlight_idx is not None and dim_y < highlight_rhs:
        dim_x = highlight_rhs
    elif h == w:
        dim_x = dim_y
    else:
        dim_x = w + (show_first_n_images + show_last_n_images - 1) * img_spacing + chunk_spacing
    bg = np.ones((dim_y, dim_x, 4), dtype=np.uint8) * 255
    bg[..., -1] = 0
    img_stack = Image.fromarray(bg)

    # Draw the back of the stack first
    for i, n in enumerate(img_idxs[::-1][:show_last_n_images]):
        b_img = _make_bordered_img(images[n], border_size, border_colours[n])
        img_stack.paste(
            b_img,
            box=(
                chunk_spacing + show_first_n_images * img_spacing + offsets_last[show_last_n_images - i - 1],
                offsets_last[i]
            ),
            mask=b_img
        )

    # Add a highlighted frame if requested
    if highlight_idx is not None:
        n = img_idxs[highlight_idx]
        b_img = _make_bordered_img(images[n], highlight_border_size, border_colours[n])
        img_stack.paste(
            b_img,
            box=(
                highlight_rhs - h,
                highlight_rhs - w - highlight_offset
            ),
            mask=b_img
        )

    # Draw the front of the stack
    for i, n in enumerate(img_idxs[:show_first_n_images][::-1]):
        b_img = _make_bordered_img(images[n], border_size, border_colours[n])
        img_stack.paste(
            b_img,
            box=(
                offsets_first[show_first_n_images - i - 1],
                chunk_spacing + show_last_n_images * img_spacing + offsets_first[i]
            ),
            mask=b_img
        )

    # Add the dots
    dot_offsets = np.linspace(0, chunk_spacing, n_dots + 2).round().astype(np.uint8)[1:-1]
    draw = ImageDraw.Draw(img_stack)

    # Centre dots
    if centre_dots:
        for i in range(n_dots):
            x = -w / 2 + img_spacing * show_first_n_images + dot_offsets[i]
            y = h / 2 + img_spacing * show_last_n_images + chunk_spacing - dot_offsets[i]
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=dot_colour)

    # Corner dots
    else:
        # Top-left dots
        for i in range(n_dots):
            x = img_spacing * show_first_n_images + dot_offsets[i]
            y = img_spacing * show_last_n_images + chunk_spacing - dot_offsets[i]
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=dot_colour)

        if highlight_idx is None:
            # Top-right dots
            for i in range(n_dots):
                x = w + img_spacing * (show_first_n_images - 1) + dot_offsets[i]
                y = img_spacing * show_last_n_images + chunk_spacing - dot_offsets[i]
                draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=dot_colour)

            # Bottom-right dots
            for i in range(n_dots):
                x = w + img_spacing * (show_first_n_images - 1) + dot_offsets[i]
                y = h + img_spacing * (show_last_n_images - 1) + chunk_spacing - dot_offsets[i]
                draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=dot_colour)

    return img_stack


def make_real_image_stacks():
    """
    Draw a stack of images from a sequence.
    """
    args, _, output_dir = _init(load_ds=False)
    image_paths = get_image_paths(args, load_all=True)
    img0 = Image.open(image_paths[0][1])
    w, h = img0.size
    crop_size = int(w * 0.5)
    out_size = (400, 400)

    # Calculate cropping box
    left = (w - crop_size) / 2
    top = (h - crop_size) / 2
    right = (w + crop_size) / 2
    bottom = (h + crop_size) / 2

    # Load the images and keypoint heatmaps
    logger.info(f'Loading images {args.start_image} to {len(image_paths) if args.end_image == -1 else args.end_image}.')
    idxs = list(range(args.start_image, len(image_paths) if args.end_image == -1 else args.end_image))
    images = []
    for idx in idxs:
        image = Image.open(image_paths[idx][1])
        image = image.crop((left, top, right, bottom))
        image = image.resize(out_size, Image.Resampling.LANCZOS)
        images.append(image)
    images = np.stack(images)

    # Add alpha channels to images
    alpha = (np.ones(images.shape[:3] + (1,)) * 255).astype(np.uint8)
    images = np.concatenate([images, alpha], axis=-1)

    # Make image stacks
    image_stack = _make_image_stack(images=images, **to_dict(args))
    image_stack.save(output_dir / 'image_stack.png')


def make_synthetic_image_stacks():
    """
    Draw a stack of images from the dataset.
    """
    args, ds, output_dir = _init()
    image_size = ds.dataset_args.image_size

    # Load the images and keypoint heatmaps
    idxs = list(range(args.start_image, ds.size_all if args.end_image == -1 else args.end_image))
    shuffle(idxs)
    images_noisy = []
    images_clean = []
    keypoints = []
    parameters = []
    for idx in idxs:
        item = ds.load_item(idx)
        images_noisy.append(np.array(item[1]))
        images_clean.append(np.array(item[2]))

        # Regenerate the keypoints heatmap
        X_kp = np.array(item[3]['kp_heatmap'])
        kp_coords = get_keypoint_coordinates(X_kp=X_kp)
        heatmap = generate_keypoints_heatmap(
            keypoints=kp_coords,
            image_size=image_size,
            blob_variance=100.0
        )
        heatmap = (to_numpy(heatmap)**3 * 255).astype(np.uint8)
        keypoints.append(heatmap)

        # Make a parameter vector
        p_vec = np.concatenate([
            item[3]['distances'],
            item[3]['transformation'],
            item[3]['material'],
            item[3]['light']
        ])
        parameters.append(p_vec)

    images_noisy = np.stack(images_noisy)
    images_clean = np.stack(images_clean)
    keypoints = np.stack(keypoints)
    parameters = np.stack(parameters)

    # Add alpha channels to images
    alpha = (np.ones(images_noisy.shape[:3] + (1,)) * 255).astype(np.uint8)
    images_noisy = np.concatenate([images_noisy, alpha], axis=-1)
    images_clean = np.concatenate([images_clean, alpha], axis=-1)

    # Map the keypoint heatmap to use reds
    cm = plt.get_cmap('Reds')
    reds = cm(np.linspace(0, 1, 256))
    reds = (reds * 255).astype(np.uint8)
    keypoints = np.take(reds, keypoints, axis=0)

    # Map the parameter vectors to use blues
    parameters = (parameters - parameters.min(axis=0)) / (parameters.max(axis=0) - parameters.min(axis=0))
    parameters = parameters * 0.7 + 0.15
    parameters = (parameters * 255).astype(np.uint8)
    cm = plt.get_cmap('Blues')
    blues = cm(np.linspace(0, 1, 256))
    blues = (blues * 255).astype(np.uint8)
    parameters = np.take(blues, parameters, axis=0)[:, :, None]
    sf = image_size / parameters.shape[1]
    zoom_factors = (1, sf, sf, 1)
    parameters = zoom(parameters, zoom_factors, order=0)
    args2 = copy.deepcopy(args)
    args2.border_size = 0
    args2.centre_dots = True

    # Make stacks
    noisy_stack = _make_image_stack(images=images_noisy, **to_dict(args))
    clean_stack = _make_image_stack(images=images_clean, **to_dict(args))
    keypoints_stack = _make_image_stack(images=keypoints, **to_dict(args))
    param_stack = _make_image_stack(images=parameters, **to_dict(args2))

    # Save stacks
    noisy_stack.save(output_dir / 'noisy_stack.png')
    clean_stack.save(output_dir / 'clean_stack.png')
    keypoints_stack.save(output_dir / 'keypoints_stack.png')
    param_stack.save(output_dir / 'param_stack.png')


def make_parameter_vector_image(n: int = 1):
    """
    Make images of random parameter vectors.
    """
    n_params = 12
    img_height = 200
    for i in range(n):
        p_vec = np.random.normal(0, 1, n_params)
        p_vec = (p_vec - p_vec.min()) / (p_vec.max() - p_vec.min())
        p_vec = p_vec * 0.7 + 0.15
        p_vec = (p_vec * 255).astype(np.uint8)
        cm = plt.get_cmap('Blues')
        blues = cm(np.linspace(0, 1, 256))
        blues = (blues * 255).astype(np.uint8)
        p_vec = np.take(blues, p_vec, axis=0)[:, None]
        sf = img_height / p_vec.shape[0]
        zoom_factors = (sf, sf, 1)
        p_vec = zoom(p_vec, zoom_factors, order=0)
        p_vec = Image.fromarray(p_vec)
        p_vec.save(LOGS_PATH / f'{START_TIMESTAMP}_param_vec_{i:02d}.png')


def make_parameter_vector_stack():
    """
    Make images of random parameter vectors.
    """
    args, _, output_dir = _init(load_ds=False)
    n_params = 12
    img_height = 200
    p_vecs = []
    for i in range(args.n_images):
        p_vec = np.random.normal(0, 1, n_params)
        p_vec = (p_vec - p_vec.min()) / (p_vec.max() - p_vec.min())
        p_vec = p_vec * 0.7 + 0.15
        p_vec = (p_vec * 255).astype(np.uint8)
        cm = plt.get_cmap('Blues')
        blues = cm(np.linspace(0, 1, 256))
        blues = (blues * 255).astype(np.uint8)
        p_vec = np.take(blues, p_vec, axis=0)[:, None]
        sf = img_height / p_vec.shape[0]
        zoom_factors = (sf, sf, 1)
        p_vec = zoom(p_vec, zoom_factors, order=0)
        p_vecs.append(p_vec)

    p_vecs = np.stack(p_vecs)
    stack = _make_image_stack(images=p_vecs, centre_dots=True, **to_dict(args))
    stack.save(output_dir / 'param_vec_stack.png')


def make_smooth_parameter_vector_images():
    """
    Make images of random parameter vectors smoothed from one end to the other
    """
    args, _, output_dir = _init(load_ds=False)
    n_params = 12
    img_height = 200
    args.n_images = 12

    # Linearly interpolate between random p0 and pT
    p0 = np.random.normal(0, 1, n_params)
    pT = np.random.normal(0, 1, n_params)
    w = np.linspace(0, 1, args.n_images)
    p_vecs = np.outer(w, pT) + np.outer(1 - w, p0)

    # Normalise and add colour
    p_vecs = (p_vecs - p_vecs.min()) / (p_vecs.max() - p_vecs.min())
    p_vecs = p_vecs * 0.7 + 0.15
    p_vecs = (p_vecs * 255).astype(np.uint8)
    cm = plt.get_cmap('Blues')
    blues = cm(np.linspace(0, 1, 256))
    blues = (blues * 255).astype(np.uint8)
    p_vecs = np.take(blues, p_vecs, axis=0)[:, :, None]

    # Scale and save
    sf = img_height / p_vecs.shape[1]
    zoom_factors = (1, sf, sf, 1)
    p_vecs = zoom(p_vecs, zoom_factors, order=0)
    for i, p_vec in enumerate(p_vecs):
        Image.fromarray(p_vec).save(output_dir / f'param_vec_smooth_{i:02d}.png')


def make_smooth_parameter_vector_stack():
    """
    Make an image stack of random parameter vectors smoothed from one end to the other
    """
    args, _, output_dir = _init(load_ds=False)
    n_params = 12
    img_height = 200

    # Linearly interpolate between random p0 and pT
    p0 = np.random.normal(0, 1, n_params)
    pT = np.random.normal(0, 1, n_params)
    w = np.linspace(0, 1, args.n_images)
    p_vecs = np.outer(w, pT) + np.outer(1 - w, p0)

    # Normalise and add colour
    p_vecs = (p_vecs - p_vecs.min()) / (p_vecs.max() - p_vecs.min())
    p_vecs = p_vecs * 0.7 + 0.15
    p_vecs = (p_vecs * 255).astype(np.uint8)
    cm = plt.get_cmap('Blues')
    blues = cm(np.linspace(0, 1, 256))
    blues = (blues * 255).astype(np.uint8)
    p_vecs = np.take(blues, p_vecs, axis=0)[:, :, None]

    # Scale and save
    sf = img_height / p_vecs.shape[1]
    zoom_factors = (1, sf, sf, 1)
    p_vecs = zoom(p_vecs, zoom_factors, order=0)
    stack = _make_image_stack(images=p_vecs, centre_dots=True, **to_dict(args))
    stack.save(output_dir / 'param_vec_smooth_stack.png')


def make_blank_stack(image_size: int = 100):
    """
    Draw a stack of blank images.
    """
    args, _, output_dir = _init(load_ds=False)
    # colour = np.array([255, 212, 42, 255], dtype=np.uint8)
    colour = np.array([255, 255, 255, 255], dtype=np.uint8)
    images = np.ones((args.n_images, image_size, image_size * 2, 4), dtype=np.uint8) * colour
    stack = _make_image_stack(images=images, centre_dots=False, **to_dict(args))
    stack.save(output_dir / 'blank_stack.png')


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    make_real_image_stacks()
    # make_synthetic_image_stacks()
    # make_parameter_vector_image(n=5)
    # make_parameter_vector_stack()
    # make_smooth_parameter_vector_images()
    # make_smooth_parameter_vector_stack()
    # make_blank_stack()
