import distutils.util
import hashlib
import json
import os
import random
from argparse import Namespace
from json import JSONEncoder
from math import log2
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from matplotlib.axes import Axes

from crystalsizer3d import logger

SEED = 0


def set_seed(seed: Optional[int] = None):
    """Set the random seed everywhere."""
    global SEED
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    logger.info(f'Setting random seed = {seed}.')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    SEED = seed


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Converts a torch tensor to a numpy array."""
    return t.detach().cpu().numpy()


def str2bool(v: str) -> bool:
    """Converts truthy and falsey strings to their boolean values."""
    return bool(distutils.util.strtobool(v))


def print_args(args: Namespace):
    """Logs all the keys and values present in an argument namespace."""
    log = '--- Arguments ---\n'
    for arg_name, arg_value in vars(args).items():
        log += f'{arg_name}: {arg_value}\n'
    log += '-----------------\n'
    logger.info(log)


def to_dict(obj) -> dict:
    """Returns a dictionary representation of an object"""
    # Get any attrs defined on the instance (self)
    inst_vars = vars(obj)
    # Filter out private attributes
    public_vars = {k: v for k, v in inst_vars.items() if not k.startswith('_')}
    # Replace any Paths with their string representations
    for k, v in public_vars.items():
        if isinstance(v, Path):
            public_vars[k] = str(v)
    return public_vars


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Converts a float32 numpy array to a uint8 numpy array"""
    # Normalize the array between 0 and 1
    normalized_array = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    # Scale the values to the range [0, 255] and round to the nearest integer
    scaled_array = np.round(normalized_array * 255)

    # Convert the array to uint8 data type
    uint8_array = scaled_array.astype(np.uint8)

    return uint8_array


class NumpyCompatibleJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def hash_data(data) -> str:
    """Generates a generic md5 hash string for arbitrary data."""
    return hashlib.md5(
        json.dumps(data, sort_keys=True, cls=NumpyCompatibleJSONEncoder).encode('utf-8')
    ).hexdigest()


def normalise(v: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Normalise an array along its final dimension."""
    if isinstance(v, torch.Tensor):
        return v / torch.norm(v, dim=-1, keepdim=True)
    else:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def is_bad(t: torch.Tensor) -> bool:
    """Checks if any of the elements in the tensor are infinite or nan."""
    if torch.isinf(t).any():
        return True
    if torch.isnan(t).any():
        return True
    return False


def from_preangles(t: torch.Tensor) -> torch.Tensor:
    """Converts an array of pre-angles to an array of angles."""
    if t.shape[-1] != 2:
        t = t.reshape((*t.shape[:-1], -1, 2))
    return torch.atan2(t[..., 0], t[..., 1])


def euler_to_quaternion(angles: np.ndarray) -> np.ndarray:
    """Convert euler angles to quaternion (XYZ order)."""
    roll, pitch, yaw = angles

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a quaternion to euler angles (XYZ order)."""
    if type(q) == torch.Tensor:
        q = to_numpy(q)
    qw, qx, qy, qz = normalise(q)

    # Roll (x-axis rotation)
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

    # Pitch (y-axis rotation)
    sin_pitch = 2 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sin_pitch, -1, 1))

    # Yaw (z-axis rotation)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    return np.array([roll, pitch, yaw])


def compose_quaternions(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compose two quaternions."""
    v1, a1 = q1[:3], q1[3]
    v2, a2 = q2[:3], q2[3]
    a = a1 * a2 - np.dot(v1, v2)
    v = a1 * v2 + a2 * v1 + np.cross(v1, v2)
    return np.array([*v, a])


def equal_aspect_ratio(ax: Axes, zoom: float = 1.0):
    """Fix equal aspect ratio for 3D plots."""
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1), zoom=zoom)


def is_power_of_two(n: int) -> bool:
    """Check if the given integer is a power of two."""
    return log2(n).is_integer()
