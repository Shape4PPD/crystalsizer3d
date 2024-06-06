import cv2
import numpy as np
import pyfastnoisesimd as fns
import torch
from torch.nn.functional import interpolate

from crystalsizer3d.crystal import Crystal
from crystalsizer3d.util.geometry import line_equation_coefficients, line_intersection, normalise
from crystalsizer3d.util.utils import SEED, is_power_of_two, to_dict, to_numpy


class NoiseTexture:
    def __init__(
            self,
            dim: int = 512,
            channels: int = 1,
            perlin_freq: float = 0.1,
            perlin_octaves: int = 5,
            white_noise_scale: float = 0.2,
            max_amplitude: float = 1.0,
            zero_centred: bool = False,
            shift: float = 0.0,
            seed: int = SEED
    ):
        self.dim = dim
        self.channels = channels
        self.perlin_freq = perlin_freq
        self.perlin_octaves = perlin_octaves
        self.white_noise_scale = white_noise_scale
        self.max_amplitude = max_amplitude
        self.zero_centred = zero_centred
        self.shift = shift
        self.seed = seed

    def build(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        noise = generate_noise_map(
            dim=self.dim,
            channels=self.channels,
            perlin_freq=self.perlin_freq,
            perlin_octaves=self.perlin_octaves,
            white_noise_scale=self.white_noise_scale,
            max_amplitude=self.max_amplitude,
            zero_centred=self.zero_centred,
            seed=self.seed,
            device=device
        ) + self.shift
        return noise.to(device)

    def to_dict(self) -> dict:
        """
        Convert the class to a dictionary.
        """
        return to_dict(self)


class NormalMapNoiseTexture(NoiseTexture):
    def __init__(
            self,
            dim: int = 512,
            perlin_freq: float = 0.1,
            perlin_octaves: int = 5,
            white_noise_scale: float = 0.1,
            max_amplitude: float = 0.5,
            seed: int = SEED
    ):
        super().__init__(
            dim=dim,
            perlin_freq=perlin_freq,
            perlin_octaves=perlin_octaves,
            white_noise_scale=white_noise_scale,
            max_amplitude=max_amplitude,
            seed=seed
        )

    def build(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        return generate_surface_normalmap(
            dim=self.dim,
            perlin_freq=self.perlin_freq,
            perlin_octaves=self.perlin_octaves,
            white_noise_scale=self.white_noise_scale,
            max_amplitude=self.max_amplitude,
            seed=self.seed,
            device=device
        ).to(device)

    def to_dict(self) -> dict:
        """
        Convert the class to a dictionary.
        """
        return {
            'dim': self.dim,
            'perlin_freq': self.perlin_freq,
            'perlin_octaves': self.perlin_octaves,
            'white_noise_scale': self.white_noise_scale,
            'max_amplitude': self.max_amplitude,
            'seed': self.seed,
        }


def generate_noise_map(
        dim: int = 512,
        channels: int = 1,
        perlin_freq: float = 0.02,
        perlin_octaves: int = 5,
        white_noise_scale: float = 0.2,
        max_amplitude: float = 1.0,
        zero_centred: bool = False,
        seed: int = SEED,
        device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generate a composite noise map.
    """
    args = dict(
        dim=dim,
        perlin_freq=perlin_freq,
        perlin_octaves=perlin_octaves,
        white_noise_scale=white_noise_scale,
        max_amplitude=max_amplitude,
        zero_centred=zero_centred,
        seed=seed,
        device=device
    )

    # Stack multiple channels of noise
    if channels > 1:
        noise_channels = []
        for c in range(channels):
            args['seed'] = seed + c
            noise = generate_noise_map(**args)
            noise_channels.append(noise)
        return torch.stack(noise_channels, axis=-1)

    # If the dimension is not a power of two, generate at a higher dimension and then downsample
    if not is_power_of_two(dim):
        dim_tmp = 1 << (dim - 1).bit_length()
        args['dim'] = dim_tmp
        noise = generate_noise_map(**args)
        noise = interpolate(noise[None, None, ...], size=dim, mode='bilinear', align_corners=False).squeeze()
        return noise

    # Normalise the perlin frequency
    perlin_freq = perlin_freq / dim

    # Initialise a fractal Perlin noise generator
    perlin_noise = fns.Noise(seed=seed)
    perlin_noise.noiseType = fns.NoiseType.PerlinFractal
    perlin_noise.frequency = perlin_freq
    perlin_noise.fractal.octaves = perlin_octaves
    perlin_noise.fractal.lacunarity = 2.0
    perlin_noise.fractal.gain = 0.5

    # Initialise white noise generator
    white_noise = fns.Noise(seed=0)
    white_noise.noiseType = fns.NoiseType.WhiteNoise

    # Generate noise arrays
    shape = (dim, dim)
    base_noise = perlin_noise.genAsGrid(shape)
    imperfections = white_noise.genAsGrid(shape)

    # Combine the noises
    combined_noise = base_noise + white_noise_scale * imperfections

    # Rescale the combined noise to the range
    # [-max_amplitude, max_amplitude] if zero_centred is True
    # [0, max_amplitude] otherwise
    norm_noise = (combined_noise - np.min(combined_noise)) / np.max(np.abs(combined_noise))
    if zero_centred:
        norm_noise = 2 * norm_noise - 1
    noise = norm_noise * max_amplitude

    # Convert to tensor
    noise = torch.from_numpy(noise).to(device)

    return noise


def generate_crystal_bumpmap(
        crystal: Crystal,
        n_defects: int,
        defect_min_width: float = 0.0001,
        defect_max_width: float = 0.001,
        defect_max_z: float = 1,
) -> torch.Tensor:
    """
    Create a bumpmap for the crystal with some line defects.
    """
    dim = crystal.bumpmap_dim
    device = crystal.origin.device
    bumpmap = np.zeros((dim, dim), dtype=np.float32)

    n_defects_added = 0
    n_attempts = 0
    while n_defects_added < n_defects and n_attempts < 1000:
        n_attempts += 1

        # Pick a random face
        face_idx = np.random.randint(0, len(crystal.faces))
        centroid_uv = crystal.uv_faces[face_idx][0]
        face_uv = crystal.uv_faces[face_idx][1:]

        # Merge nearby uv coordinates
        face_uv = torch.unique(face_uv, dim=0)

        # Pick two random adjacent vertices
        v0_idx = np.random.randint(0, len(face_uv))
        v1_idx = (v0_idx + 1) % len(face_uv)
        v0 = face_uv[v0_idx]
        v1 = face_uv[v1_idx]
        edge = v1 - v0
        if edge.norm() < 1e-6:
            continue
        midpoint = (v0 + v1) / 2

        # Find where a perpendicular line from the middle of this edge would intersect another edge
        l_perp = line_equation_coefficients(v0, v1, perpendicular=True)
        max_dist = np.inf
        for j in range(len(face_uv)):
            if j == v0_idx:
                continue
            u0 = face_uv[j]
            u1 = face_uv[(j + 1) % len(face_uv)]
            l = line_equation_coefficients(u0, u1)
            intersect = line_intersection(l, l_perp)
            if intersect is None:
                continue
            if torch.dot(intersect - midpoint, centroid_uv - midpoint) < 0:
                continue
            max_dist = min(max_dist, (intersect - midpoint).norm().item())
        if max_dist == np.inf or max_dist == 0:
            continue

        # Pick a random perpendicular distance
        d = np.random.uniform(0, max_dist)

        # Pick a random line length
        edge_length = edge.norm().item()
        l = np.random.uniform(edge_length * 0.1, edge_length * 2)

        # Pick a random midpoint offset
        offset = np.random.uniform(-0.2, 0.2)

        # Calculate end points of the defect line parallel to the edge
        perp_vec = normalise(torch.tensor([-l_perp[1], l_perp[0]], device=device))
        if torch.dot(perp_vec, centroid_uv - midpoint) < 0:
            perp_vec = -perp_vec
        defect_start = midpoint + d * perp_vec - l * (0.5 + offset) * edge / edge_length
        defect_end = midpoint + d * perp_vec + l * (0.5 - offset) * edge / edge_length

        # Convert to image coordinates
        x0, y0 = to_numpy((defect_start * (dim - 1)).round().to(torch.int64))
        x1, y1 = to_numpy((defect_end * (dim - 1)).round().to(torch.int64))

        # Draw line between points
        w = max(1, int(np.random.uniform(defect_min_width, defect_max_width) * dim))
        defect = np.zeros((dim, dim), dtype=np.uint8)
        cv2.line(defect, (x0, y0), (x1, y1), 255, w, cv2.LINE_AA)

        # Scale the bump depth and add some noise to the line
        defect = defect.astype(np.float32) / 255  # * z
        line = defect > 0.1
        noise = np.random.normal(np.zeros(line.sum()), 0.2)
        defect[line] += noise
        z = (np.random.uniform() * 2 - 1) * defect_max_z
        defect[line] *= z

        # Add the defect to the bumpmap
        defect = defect * to_numpy(crystal.uv_mask)
        bumpmap += defect
        bumpmap = np.clip(bumpmap, -defect_max_z, defect_max_z)

        # Reset the counter
        n_defects_added += 1
        n_attempts = 0

    return torch.from_numpy(bumpmap).to(device)


def generate_surface_normalmap(
        dim: int = 200,
        perlin_freq: float = 10.,
        perlin_octaves: int = 5,
        white_noise_scale: float = 0.1,
        max_amplitude: float = 0.5,
        seed: int = SEED,
        device=torch.device('cpu')
) -> torch.Tensor:
    """
    Generate a randomised surface normal map.
    """
    spherical_angles = []
    for i in range(2):
        theta_phi = generate_noise_map(
            dim=dim,
            perlin_freq=perlin_freq,
            perlin_octaves=perlin_octaves,
            white_noise_scale=white_noise_scale,
            max_amplitude=np.pi / 2 * max_amplitude if i == 0 else 2 * np.pi,
            seed=seed + i,
            device=device
        )
        spherical_angles.append(theta_phi)
    spherical_angles = torch.stack(spherical_angles)
    normalmap = (torch.stack([
        torch.sin(spherical_angles[0]) * torch.cos(spherical_angles[1]),
        torch.sin(spherical_angles[0]) * torch.sin(spherical_angles[1]),
        torch.cos(spherical_angles[0])
    ], dim=-1) + 1) / 2

    return normalmap
