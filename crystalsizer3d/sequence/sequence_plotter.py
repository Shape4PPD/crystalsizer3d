import multiprocessing as mp
import os
import time
from pathlib import Path
from queue import Empty, Full
from typing import Any, Dict

import numpy as np
from PIL import Image
from torch import Tensor

from crystalsizer3d import N_WORKERS, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.sequence.plots import annotate_image
from crystalsizer3d.util.parallelism import start_process, stop_event as global_stop_event
from crystalsizer3d.util.utils import to_numpy


def plotter_worker(queue: mp.Queue, stop_event: mp.Event):
    """
    Worker process for the sequence plotter.
    """
    worker_id = os.getpid()
    logger.info(f'Plotter worker started (id={worker_id})')
    while not stop_event.is_set() and not global_stop_event.is_set():
        try:
            job = queue.get(timeout=1)
        except Empty:
            continue
        if job is None:
            time.sleep(1)
            continue
        method_name = f'_{job["type"]}'
        getattr(SequencePlotter, method_name)(job)
        time.sleep(1)


class SequencePlotter:
    def __init__(self, n_workers: int = N_WORKERS, queue_size: int = 100):
        self.workers = {}
        self.n_workers = n_workers

        if n_workers > 1:
            # Ensure that CUDA will work in subprocesses
            mp.set_start_method('spawn', force=True)

            # Set up the channels
            self.queue = mp.Queue(maxsize=queue_size)

            # Start the worker processes
            self.stop_event = mp.Event()
            self._start_workers()

    def _start_workers(self):
        """
        Start the worker processes.
        """
        for i in range(self.n_workers):
            process = mp.Process(
                target=plotter_worker,
                args=(self.queue, self.stop_event),
            )
            start_process(process)

    def enqueue_job(self, job: Dict[str, Any]):
        """
        Queue a job for asynchronous processing.
        """
        method_name = f'_{job["type"]}'
        if not hasattr(SequencePlotter, method_name):
            raise RuntimeError(f'Invalid job type: "{job["type"]}"')
        if self.n_workers > 1:
            while True:
                try:
                    self.queue.put(job, block=False)
                    break
                except Full:
                    time.sleep(0.2)
                    continue
        else:
            getattr(SequencePlotter, method_name)(job)

    def wait_for_empty_queue(self):
        """
        Wait for the queue to empty.
        """
        if not self.queue.empty():
            logger.info('Waiting for the queue to empty...')
            while not self.queue.empty():
                time.sleep(1)
            logger.info('Queue is empty.')

    def save_image(self, X: Tensor | np.ndarray, save_path: Path):
        """
        Save an image to disk.
        """
        if isinstance(X, Tensor):
            X = to_numpy(X)
        if X.dtype in [np.float32, np.float64]:
            X = (X * 255).astype(np.uint8)
        if X.ndim == 3 and X.shape[0] == 1:
            X = X[0]
        if X.ndim == 3 and X.shape[0] == 3:
            X = X.transpose(1, 2, 0)
        self.enqueue_job({
            'type': 'save_image',
            'X': X.copy(),
            'save_path': save_path,
        })

    @staticmethod
    def _save_image(job: Dict[str, Any]):
        """
        Save an image to disk.
        """
        X = job['X']
        Image.fromarray(X).save(job['save_path'])

    def annotate_image(self, X_path: Path, X_annotated_path: Path, scene: Scene):
        """
        Annotate an image with the wireframe overlay of the crystal.
        """
        self.enqueue_job({
            'type': 'annotate_image',
            'X_path': X_path,
            'X_annotated_path': X_annotated_path,
            'zoom': orthographic_scale_factor(scene),
            'crystal': scene.crystal.to_dict()
        })

    @staticmethod
    def _annotate_image(job: Dict[str, Any]):
        """
        Annotate an image with the wireframe overlay of the crystal.
        """
        annotate_image(
            image_path=job['X_path'],
            crystal=Crystal.from_dict(job['crystal']),
            zoom=job['zoom'],
        ).save(job['X_annotated_path'])

    def render_scene(self, scene: Scene, save_path: Path):
        """
        Render the scene to an image.
        """
        self.enqueue_job({
            'type': 'render_scene',
            'scene': scene.to_dict(),
            'save_path': save_path,
        })

    @staticmethod
    def _render_scene(job: Dict[str, Any]):
        """
        Render the scene to an image.
        """
        scene = Scene.from_dict(job['scene'])
        X = scene.render()
        Image.fromarray(X).save(job['save_path'])

    def generate_video(self, imgs_dir: Path, train_or_eval: str, save_root: Path, step: int):
        """
        Generate a video from the images.
        """
        self.enqueue_job({
            'type': 'generate_video',
            'imgs_dir': imgs_dir,
            'out_dir': save_root / 'videos' / train_or_eval / 'annotations',
            'train_or_eval': train_or_eval,
            'step': step,
        })

    @staticmethod
    def _generate_video(job: Dict[str, Any]):
        """
        Generate a video from the images.
        """
        imgs_dir = job['imgs_dir']
        out_dir = job['out_dir']
        step = job['step']
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f'{step:05d}.mp4'
        logger.info(f'Making growth video from {imgs_dir} to {save_path}.')
        escaped_images_dir = str(imgs_dir.absolute()).replace('[', '\\[').replace(']', '\\]')
        cmd = f'ffmpeg -y -framerate 25 -pattern_type glob -i "{escaped_images_dir}/*.png" -c:v libx264 -pix_fmt yuv420p "{save_path}"'
        logger.info(f'Running command: {cmd}')
        os.system(cmd)
