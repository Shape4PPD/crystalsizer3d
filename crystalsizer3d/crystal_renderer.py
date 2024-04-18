import json
import os
from abc import ABC, abstractmethod
from multiprocessing import Lock
from pathlib import Path

from crystalsizer3d import N_WORKERS
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs
from crystalsizer3d.args.renderer_args import RendererArgs


class RenderError(RuntimeError):
    def __init__(self, message: str, idx: int = None):
        super().__init__(message)
        self.idx = idx


def append_json(file_path: Path, new_data: dict):
    if not file_path.exists():
        data = {}
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
    if len(data) > 0:
        for k in new_data.keys():
            assert k not in data, f'Key "{k}" already exists in {file_path}'
    data.update(new_data)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def append_json_mp(file_path: Path, new_data: dict, lock: Lock):
    with lock:
        append_json(file_path, new_data)


class CrystalRenderer(ABC):
    def __init__(
            self,
            dataset_args: DatasetSyntheticArgs,
            renderer_args: RendererArgs,
            quiet_render: bool = False
    ):
        self.dataset_args = dataset_args
        self.renderer_args = renderer_args
        self.quiet_render = quiet_render
        if N_WORKERS > 0:
            self.n_workers = N_WORKERS
        else:
            self.n_workers = len(os.sched_getaffinity(0))

    @property
    @abstractmethod
    def images_dir(self) -> Path:
        """
        Directory to save images to.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Render all crystal objects to images.
        """
        pass
