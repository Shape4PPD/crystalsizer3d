import distutils.util
import hashlib
import json
import os
import random
import threading
from argparse import Namespace
from collections import OrderedDict
from json import JSONEncoder
from math import log2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from torch import Tensor
from torchvision.transforms import GaussianBlur

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


def get_seed() -> int:
    return SEED


def to_numpy(t: Tensor) -> np.ndarray:
    """Converts a torch tensor to a numpy array."""
    return t.detach().cpu().numpy().copy()


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

def init_tensor(
        tensor: Union[Tensor, np.ndarray, List[float], Tuple[float, ...], float, int],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
) -> Tensor:
    """
    Create a clone of a tensor or numpy array.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if (isinstance(tensor, list) or isinstance(tensor, tuple) or isinstance(tensor, float)
            or isinstance(tensor, int) or isinstance(tensor, np.floating)):
        tensor = torch.tensor(tensor)
    tensor = tensor.to(dtype).detach().clone()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class FlexibleJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Tensor):
            obj = to_numpy(obj)
        if isinstance(obj, np.generic) or isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return JSONEncoder.default(self, obj)


class ArgsCompatibleJSONEncoder(JSONEncoder):
    def default(self, obj):
        from crystalsizer3d.args.base_args import BaseArgs
        if isinstance(obj, BaseArgs):
            return obj.to_dict()
        return JSONEncoder.default(self, obj)


def json_to_numpy(data: Any) -> Any:
    """
    Convert json data to numpy arrays where possible.
    """
    if isinstance(data, dict):
        return {k: json_to_numpy(v) for k, v in data.items()}
    if isinstance(data, list):
        types = [type(v) for v in data]
        if all([t in [int, float] for t in types]):
            return np.array(data)
        if all([t in [list, dict] for t in types]):
            n_entries = [len(v) for v in data]
            if len(set(n_entries)) == 1:
                return np.array(data)
        return [json_to_numpy(v) for v in data]

def json_to_torch(data: Any) -> Any:
    """
    Convert json data to torch tensors where possible.
    """
    if isinstance(data, dict):
        return {k: json_to_torch(v) for k, v in data.items()}
    if isinstance(data, list):
        types = [type(v) for v in data]
        if all([t in [int, float] for t in types]):
            return torch.tensor(data)
        if all([t in [list, dict] for t in types]):
            n_entries = [len(v) for v in data]
            if len(set(n_entries)) == 1:
                return torch.tensor(data)
        return [json_to_torch(v) for v in data]


def hash_data(data) -> str:
    """Generates a generic md5 hash string for arbitrary data."""
    return hashlib.md5(
        json.dumps(data, sort_keys=True, cls=FlexibleJSONEncoder).encode('utf-8')
    ).hexdigest()


def is_bad(t: Tensor) -> bool:
    """Checks if any of the elements in the tensor are infinite or nan."""
    if torch.isinf(t).any():
        return True
    if torch.isnan(t).any():
        return True
    return False


def calculate_model_norm(
        model: torch.nn.Module,
        p: int = 2,
        device: torch.device = torch.device('cpu')
) -> Tensor:
    """Calculate the cumulative Lp norms of the model parameters."""
    if isinstance(model, torch.nn.DataParallel):  # Check if the model is a DataParallel object
        model = model.module
    with torch.no_grad():
        norm = torch.tensor(0., dtype=torch.float32, device=device)
        for name, m in model.named_modules():
            if hasattr(m, 'parameters'):
                p_norm = 0
                for i, param in enumerate(m.parameters()):
                    p_norm += param.norm(p)
                norm += p_norm
    return norm


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


# @torch.jit.script
def to_multiscale(img: Tensor, blur: GaussianBlur) -> List[Tensor]:
    """
    Generate downsampled and blurred images.
    """
    assert img.ndim == 3, 'Input image must have 3 dimensions.'
    if img.shape[-1] in [1, 3]:
        img = img.clone().permute(2, 0, 1)
    imgs = [img.clone()[None, ...]]
    while min(imgs[-1].shape[-2:]) > blur.kernel_size[0] + 2:
        imgs.append(blur(F.interpolate(imgs[-1], scale_factor=0.5, mode='bilinear')))
    imgs = [i[0].permute(1, 2, 0) for i in imgs]
    return imgs


@torch.jit.script
def gumbel_sigmoid(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Sample from a Gumbel-Sigmoid distribution.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    return torch.sigmoid(y / temperature)


def is_main_thread():
    """Check if the current thread is the main thread."""
    return threading.current_thread().name == 'MainThread'


def smooth_signal(x: np.ndarray, window_size: int = 11) -> np.ndarray:
    """
    Smooth the input signal using a moving average filter.
    """
    kernel = np.ones(window_size) / window_size
    pad_width = (window_size - 1) // 2
    x = np.pad(x, (pad_width, window_size - pad_width - 1), mode='edge')
    return np.convolve(x, kernel, mode='valid')


def get_crystal_face_groups(manager: 'Manager') -> Dict[Tuple[int, int, int], Dict[Tuple[int, int, int], int]]:
    """
    Get the crystal face groups.
    """
    groups = {}
    for i, group_hkl in enumerate(manager.ds.dataset_args.miller_indices):
        group_idxs = (manager.crystal.symmetry_idx == i).nonzero().squeeze()
        groups[group_hkl] = OrderedDict([
            (tuple(manager.crystal.all_miller_indices[j].tolist()), int(j))
            for j in group_idxs
        ])
    return groups
