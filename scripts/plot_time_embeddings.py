import math
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP


def fourier_encode(time_points: np.ndarray, bands: np.ndarray):
    """
    Apply Fourier encoding to continuous time points.
    """
    time_points = time_points[:, None] * bands
    sin_features = np.sin(time_points)
    cos_features = np.cos(time_points)
    return np.concatenate([sin_features, cos_features], axis=-1)


def plot_time_embeddings():
    """
    Make images of the time embeddings.
    """
    hidden_dim = 32
    max_freq = 4
    n_timesteps = 100
    save_n_timesteps = 20
    img_height = 200
    cm = plt.get_cmap('YlOrBr')
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}_dim={hidden_dim}_freq={max_freq}_n={n_timesteps}'
    save_dir.mkdir(exist_ok=True)

    # Define frequency bands for Fourier features
    n_fourier_features = hidden_dim // 2  # Number of sin/cos pairs
    bands = 2**np.linspace(0, math.log2(max_freq), n_fourier_features)

    # Encode time points
    time_points = np.linspace(0, 2 * np.pi, n_timesteps)
    time_points = time_points[::-1]
    t_enc = fourier_encode(time_points, bands)

    # Save the full matrix
    plt.imshow(t_enc.T, aspect='auto', cmap=cm)
    plt.savefig(save_dir / 'time_embedding.png', bbox_inches='tight', pad_inches=0)

    # Plot the embeddings
    save_idxs = np.linspace(0, n_timesteps - 1, save_n_timesteps).astype(int)
    t_enc = (t_enc - t_enc.min()) / (t_enc.max() - t_enc.min())
    colours = cm(np.linspace(0, 1, 256))
    colours = (colours * 255).astype(np.uint8)
    for i, idx in enumerate(save_idxs):
        t_vec = t_enc[idx]
        t_vec = (t_vec * 255).astype(np.uint8)
        t_vec = np.take(colours, t_vec, axis=0)[:, None]
        sf = img_height / t_vec.shape[0]
        zoom_factors = (sf, sf, 1)
        t_vec = zoom(t_vec, zoom_factors, order=0)
        t_vec = Image.fromarray(t_vec)
        t_vec.save(save_dir / f't_vec_{idx:04d}.png')


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    # make_image_stacks()
    plot_time_embeddings()
