import distutils.util
import hashlib
import json
import os
import random
from argparse import Namespace
from json import JSONEncoder
from math import log2
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.colors as mcolors
import numpy as np
import torch
from filelock import FileLock
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


def init_tensor(tensor: Union[torch.Tensor, np.ndarray, List[float], float, int], dtype=torch.float32) -> torch.Tensor:
    """
    Create a clone of a tensor or numpy array.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if isinstance(tensor, list) or isinstance(tensor, float) or isinstance(tensor, int):
        tensor = torch.tensor(tensor)
    return tensor.to(dtype).detach().clone()


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


def is_bad(t: torch.Tensor) -> bool:
    """Checks if any of the elements in the tensor are infinite or nan."""
    if torch.isinf(t).any():
        return True
    if torch.isnan(t).any():
        return True
    return False


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


def append_json(file_path: Path, new_data: dict, timeout: int = 30):
    """
    Append new data to a JSON file.
    """
    if not file_path.exists():
        with open(file_path, 'w') as f:
            json.dump({}, f)
    lock_path = file_path.with_suffix('.lock')
    lock = FileLock(lock_path, timeout=timeout)
    lock.acquire()
    with open(file_path, 'r') as f:
        data = json.load(f)
    if len(data) > 0:
        for k in new_data.keys():
            if k in data and hash_data(data[k]) != hash_data(new_data[k]):
                raise ValueError(f'Key "{k}" already exists in {file_path} and is not the same!')
    data.update(new_data)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    lock.release()
