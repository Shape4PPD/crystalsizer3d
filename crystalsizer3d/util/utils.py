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

import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation

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
        return normalise_pt(v)
    else:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


@torch.jit.script
def normalise_pt(v: torch.Tensor) -> torch.Tensor:
    """Normalise a tensor along its final dimension."""
    return v / torch.norm(v, dim=-1, keepdim=True)


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
    """Convert euler angles (XYZ order) to quaternion in scalar-first format."""
    r = Rotation.from_euler('XYZ', angles)
    return r.as_quat(canonical=True)[[3, 0, 1, 2]]


def quaternion_to_euler(q: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a quaternion (scalar-first) to euler angles (XYZ order)."""
    if type(q) == torch.Tensor:
        q = to_numpy(q)
    q = normalise(q)
    r = Rotation.from_quat(q[[1, 2, 3, 0]])
    return r.as_euler('XYZ')


def euler_to_axisangle(angles: np.ndarray) -> np.ndarray:
    """Convert euler angles to an axis-angle representation."""
    r = Rotation.from_euler('XYZ', angles)
    return r.as_rotvec()


def axisangle_to_euler(v: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert an axis-angle representation to euler angles."""
    if type(v) == torch.Tensor:
        v = to_numpy(v)
    r = Rotation.from_rotvec(v)
    return r.as_euler('XYZ')


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Create the skew symmetric matrix for the (batched) input vectors
    v = (v_1, v_2, v_3)
    v_ss = | 0    -v_3    v_2 |
           | v_3     0   -v_1 |
           | -v_2  v_1     0  |
    """
    M = torch.zeros(v.shape[0], 3, 3, device=v.device)
    M[:, 0, 1] = -v[:, 2]
    M[:, 0, 2] = v[:, 1]
    M[:, 1, 0] = v[:, 2]
    M[:, 1, 2] = -v[:, 0]
    M[:, 2, 0] = -v[:, 1]
    M[:, 2, 1] = v[:, 0]
    return M


def axisangle_to_matrix(v: torch.Tensor, EPS: float = 1e-2) -> torch.Tensor:
    """Convert axis-angle representation to rotation matrix."""
    raise RuntimeError('TODO: change to kornia library')
    ss = skew_symmetric(v)
    theta_sq = torch.sum(v**2, dim=1)
    is_angle_small = theta_sq < EPS

    theta = torch.sqrt(theta_sq)
    theta_pow_4 = theta_sq * theta_sq
    theta_pow_6 = theta_sq * theta_sq * theta_sq
    theta_pow_8 = theta_sq * theta_sq * theta_sq * theta_sq

    term_1 = torch.where(is_angle_small,
                         1 - (theta_sq / 6) + (theta_pow_4 / 120) - (theta_pow_6 / 5040) + (theta_pow_8 / 362880),
                         torch.sin(theta) / theta)

    term_2 = torch.where(is_angle_small,
                         0.5 - (theta_sq / 24) + (theta_pow_4 / 720) - (theta_pow_6 / 40320) + (theta_pow_8 / 3628800),
                         (1 - torch.cos(theta)) / theta_sq)

    term_1_expand = term_1.view(-1, 1, 1)
    term_2_expand = term_2.view(-1, 1, 1)
    I = torch.eye(3, device=v.device).expand(v.shape[0], -1, -1)

    v_exp = I + term_1_expand * ss + term_2_expand * torch.matmul(ss, ss)

    return v_exp


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor, EPS: float = 1e-4) -> torch.Tensor:
    """Compute the geodesic distance between two rotation matrices."""
    R = torch.matmul(R2, R1.transpose(-2, -1))
    trace = torch.einsum('...ii', R)
    trace_temp = (trace - 1) / 2
    trace_temp = torch.clamp(trace_temp, -1 + EPS, 1 - EPS)
    theta = torch.acos(trace_temp)
    return theta


def compose_quaternions(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compose two quaternions (scalar-first format)."""
    a1, v1 = q1[0], q1[1:]
    a2, v2 = q2[0], q2[1:]
    a = a1 * a2 - np.dot(v1, v2)
    v = a1 * v2 + a2 * v1 + np.cross(v1, v2)
    return np.array([a, *v])


def equal_aspect_ratio(ax: Axes, zoom: float = 1.0):
    """Fix equal aspect ratio for 3D plots."""
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1), zoom=zoom)


def is_power_of_two(n: int) -> bool:
    """Check if the given integer is a power of two."""
    return log2(n).is_integer()


def to_rgb(c: Union[str, np.ndarray]):
    if type(c) == str:
        return mcolors.to_rgb(c)
    return c
