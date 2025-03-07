import multiprocessing as mp
import os
import time
from pathlib import Path
from queue import Empty, Full
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from torch import Tensor

from crystalsizer3d import N_WORKERS, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.sequence.plots import annotate_image, annotate_image_with_keypoints, plot_distances, \
    plot_light_radiance, plot_material_properties, plot_origin, plot_rotation
from crystalsizer3d.util.parallelism import start_process, stop_event as global_stop_event
from crystalsizer3d.util.utils import to_numpy


def plotter_worker(queue: mp.Queue, stop_event: mp.Event, worker_status: mp.Array, worker_idx: int):
    """
    Worker process for the sequence plotter.
    """
    worker_id = os.getpid()
    logger.info(f'Plotter worker started (id={worker_id})')
    while not stop_event.is_set() and not global_stop_event.is_set():
        try:
            job = queue.get(timeout=1)
        except Empty:
            worker_status[worker_idx] = 0
            continue
        if job is None:
            time.sleep(1)
            continue
        worker_status[worker_idx] = 1
        method_name = f'_{job["type"]}'
        getattr(SequencePlotter, method_name)(job)
        time.sleep(1)


class SequencePlotter:
    def __init__(
            self,
            n_workers: int = N_WORKERS,
            queue_size: int = 100,
            measurements: Dict[str, Tensor] = None
    ):
        self.workers = {}
        self.n_workers = n_workers
        self.measurements = measurements

        if n_workers > 0:
            # Ensure that CUDA will work in subprocesses
            mp.set_start_method('spawn', force=True)

            # Set up the job queue
            self.queue = mp.Queue(maxsize=queue_size)

            # Start the worker processes
            self.stop_event = mp.Event()
            self._start_workers()

    def _start_workers(self):
        """
        Start the worker processes.
        """
        self.worker_status = mp.Array('i', [1] * self.n_workers)
        for i in range(self.n_workers):
            process = mp.Process(
                target=plotter_worker,
                args=(self.queue, self.stop_event, self.worker_status, i),
            )
            start_process(process)

    def enqueue_job(self, job: Dict[str, Any]):
        """
        Queue a job for asynchronous processing.
        """
        method_name = f'_{job["type"]}'
        if not hasattr(SequencePlotter, method_name):
            raise RuntimeError(f'Invalid job type: "{job["type"]}"')
        if self.n_workers > 0:
            while True:
                try:
                    self.queue.put(job, block=False)
                    break
                except Full:
                    time.sleep(0.2)
                    continue
        else:
            getattr(SequencePlotter, method_name)(job)

    def all_workers_idle(self):
        """
        Check if all workers are idle.
        """
        return (self.n_workers == 0
                or all(status == 0 for status in self.worker_status))

    def wait_for_workers(self):
        """
        Wait for all workers to complete.
        """
        if not self.all_workers_idle():
            logger.info('Waiting for plotter workers...')
            while not self.all_workers_idle():
                time.sleep(1)
            logger.info('Plotter workers ready.')

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

    def annotate_image(
            self,
            X_path: Path,
            X_annotated_path: Path,
            scene: Scene | None = None,
            zoom: float | None = None,
            crystal: Crystal | Dict[str, Any] | None = None,
            keypoints: np.ndarray | None = None,
            edge_points: np.ndarray | None = None,
            edge_point_deltas: np.ndarray | None = None
    ):
        """
        Annotate an image with the wireframe overlay of the crystal.
        """
        if scene is None:
            assert zoom is not None and crystal is not None, 'If scene is not provided, zoom and crystal must be.'
        else:
            assert zoom is None and crystal is None, 'If scene is provided, zoom and crystal must not be.'
            crystal = scene.crystal
            zoom = orthographic_scale_factor(scene)
        if isinstance(crystal, Crystal):
            crystal = crystal.to_dict()

        self.enqueue_job({
            'type': 'annotate_image',
            'X_path': X_path,
            'X_annotated_path': X_annotated_path,
            'zoom': zoom,
            'crystal': crystal,
            'keypoints': keypoints,
            'edge_points': edge_points,
            'edge_point_deltas': edge_point_deltas,
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
            keypoints=job['keypoints'],
            edge_points=job['edge_points'],
            edge_point_deltas=job['edge_point_deltas'],
        ).save(job['X_annotated_path'])

    def annotate_image_with_keypoints(
            self,
            X_path: Path,
            keypoints: np.ndarray | Tensor,
            save_path: Path
    ):
        """
        Annotate an image with keypoints.
        """
        if isinstance(keypoints, Tensor):
            keypoints = to_numpy(keypoints)
        self.enqueue_job({
            'type': 'annotate_image_with_keypoints',
            'X_path': X_path,
            'keypoints': keypoints,
            'save_path': save_path
        })

    @staticmethod
    def _annotate_image_with_keypoints(job: Dict[str, Any]):
        """
        Annotate an image with keypoints.
        """
        annotate_image_with_keypoints(
            image_path=job['X_path'],
            keypoints=job['keypoints'],
        ).save(job['save_path'])

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

    def plot_sequence_parameters(
            self,
            plot_dir: Path,
            step: int,
            parameters: Dict[str, Tensor],
            image_paths: List[Tuple[int, Path]],
            face_groups: List[List[int]]
    ):
        """
        Plot the sequence parameters.
        """
        plot_args = dict(
            type='plot_parameter_sequence',
            plot_dir=plot_dir,
            step=step,
            parameters=parameters,
            measurements=self.measurements,
            image_paths=image_paths,
            face_groups=face_groups,
            make_means_plot=False,
        )
        for plot_type in ['distances', 'origin', 'rotation', 'material', 'light']:
            self.enqueue_job({'plot_type': plot_type, **plot_args})

    @staticmethod
    def _plot_parameter_sequence(job: Dict[str, Any]):
        """
        Plot the distances.
        """
        if job['plot_type'] == 'distances':
            fig = plot_distances(**job)
        elif job['plot_type'] == 'origin':
            fig = plot_origin(**job)
        elif job['plot_type'] == 'rotation':
            fig = plot_rotation(**job)
        elif job['plot_type'] == 'material':
            fig = plot_material_properties(**job)
        elif job['plot_type'] == 'light':
            fig = plot_light_radiance(**job)
        else:
            raise ValueError(f'Invalid plot type: {job["plot_type"]}')
        save_dir = job['plot_dir'] / job['plot_type']
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'{job["step"] + 1:06d}.png')
        plt.close(fig)

    def plot_sequence_losses(
            self,
            plot_dir: Path,
            step: int,
            losses: Dict[str, Tensor],
            image_paths: List[Tuple[int, Path]],
    ):
        """
        Plot the sequence parameters.
        """
        self.enqueue_job({
            'type': 'plot_sequence_losses',
            'plot_dir': plot_dir,
            'step': step,
            'losses': losses,
            'image_paths': image_paths,
            'measurements': self.measurements,
        })

    @staticmethod
    def _plot_sequence_losses(job: Dict[str, Any]):
        """
        Plot the sequence losses.
        """
        include_keys = ['total', 'measurement', 'l1', 'l2', 'perceptual', 'latents',
                        'overshoot', 'symmetry', 'z_pos', 'rotation_xy', 'keypoints', 'edge_matching',
                        'negative_growth']
        losses = {k: job['losses'][k] for k in include_keys if k in job['losses'] and job['losses'][k] is not None}
        measurements = job['measurements']
        n = len(losses)
        n_rows = int(np.ceil(np.sqrt(n)))
        n_cols = int(np.ceil(n / n_rows))
        fig = plt.figure(figsize=(15, 15))
        gs = GridSpec(
            n_rows, n_cols,
            top=0.95, bottom=0.05, right=0.98, left=0.05,
            hspace=0.3, wspace=0.2
        )
        for i, (name, loss) in enumerate(losses.items()):
            ax = fig.add_subplot(gs[i])
            ax.set_title(name)
            if name == 'measurement' and measurements is not None:
                m_idxs = measurements['idx'][measurements['idx'] < len(loss)]
                ax.plot(m_idxs, loss[m_idxs])
            else:
                ax.plot(loss)
            ax.set_xlabel('Image index')
            ax.grid()
        save_dir = job['plot_dir'] / 'losses'
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'{job["step"] + 1:06d}.png')
        plt.close(fig)
