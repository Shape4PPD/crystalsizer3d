import multiprocessing as mp
import os
import time
from pathlib import Path
from queue import Empty, Full
from typing import Any, Dict, List, Tuple

import mitsuba as mi
import numpy as np
from torch import Tensor

from crystalsizer3d import N_WORKERS, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.sequence.sequence_plotter import SequencePlotter
from crystalsizer3d.util.parallelism import start_process, stop_event as global_stop_event
from crystalsizer3d.util.utils import init_tensor, set_seed, to_numpy


def refiner_worker(
        refiner_args: RefinerArgs,
        output_dir: Path,
        initial_scene_dict: Dict[str, Any],
        fixed_parameters: Dict[str, Tensor],
        seed: int | None,
        job_queue: mp.Queue,
        response_queue: mp.Queue,
        stop_event: mp.Event,
        worker_status: mp.Array,
        worker_idx: int
):
    """
    Refiner worker.
    """
    from crystalsizer3d.sequence.sequence_fitter import PARAMETER_KEYS
    worker_id = os.getpid()
    logger.info(f'Refiner worker started (id={worker_id})')

    # Initialise the refiner
    refiner = Refiner(
        args=refiner_args,
        output_dir=output_dir,
        do_init=False,
    )
    if seed is not None:
        set_seed(seed)

    # Load the first scene and adapt the crystal to use buffers instead of parameters
    scene = Scene.from_dict(initial_scene_dict)
    crystal = scene.crystal
    for k in list(crystal._parameters.keys()):
        val = crystal.get_parameter(k).data
        del crystal._parameters[k]
        crystal.register_buffer(k, val)

    # Set up the refiner
    refiner.scene = scene
    refiner.scene_params = mi.traverse(scene.mi_scene)
    refiner.crystal = crystal
    refiner.init_projector()

    # Work out the parameter shapes
    parameter_shapes = {}
    for k in PARAMETER_KEYS:
        if hasattr(scene, k):
            v = getattr(scene, k)
        else:
            v = getattr(crystal, k)
        parameter_shapes[k] = v.numel()

    def _parameter_vector_to_dict(parameter_vector: Tensor) -> Dict[str, Tensor]:
        """
        Convert a parameter vector to a dictionary of parameters, with fixed parameter replacements.
        """
        parameters = {}
        idx = 0
        for k, n in parameter_shapes.items():
            val = parameter_vector[idx:idx + n].squeeze()
            if k in fixed_parameters:
                val = fixed_parameters[k].squeeze()
            elif k == 'scale':
                val = val.clamp(0.01, 10)
            elif k == 'distances':
                val = val.clamp(0.01, 10)
            elif k == 'origin':
                val = val.clamp(-10, 10)
            elif k == 'rotation':
                val = val.clamp(-2 * np.pi, 2 * np.pi)
            elif k == 'material_roughness':
                val = val.clamp(0.01, 1)
            elif k == 'material_ior':
                val = val.clamp(1, 3)
            elif k == 'light_radiance':
                val = val.clamp(0, 10)
            parameters[k] = val
            idx += n
        return parameters

    def calculate_gradients(
            refiner_args: RefinerArgs,
            p_vec: Tensor,
            X_target: Tensor,
            X_target_denoised: Tensor,
            X_target_wis: Tensor,
            X_target_denoised_wis: Tensor,
            keypoints: Tensor | None,
    ):
        """
        Calculate gradients for the given parameters and targets
        """
        # Update args
        refiner.args = refiner_args

        # Update targets
        refiner.X_target = X_target
        refiner.X_target_denoised = X_target_denoised
        refiner.X_target_wis = X_target_wis.to(refiner.device)
        refiner.X_target_denoised_wis = X_target_denoised_wis.to(refiner.device)
        if refiner_args.use_keypoints:
            refiner.keypoint_targets = keypoints

        # Update scene and crystal parameters
        p_vec.requires_grad = True
        p_dict = _parameter_vector_to_dict(p_vec)
        for k, v in p_dict.items():
            if k == 'light_radiance':
                refiner.scene.light_radiance = v.to(refiner.device)
            else:
                setattr(refiner.scene.crystal, k, v.to('cpu'))

        # Calculate losses
        loss, stats = refiner.process_step(add_noise=False)
        loss.backward()

        # Accumulate gradients
        p_grads = p_vec.grad.clone()

        return loss, p_grads, refiner.X_pred

    while not stop_event.is_set() and not global_stop_event.is_set():
        try:
            job = job_queue.get(timeout=1)
        except Empty:
            worker_status[worker_idx] = 0
            continue
        if job is None:
            time.sleep(1)
            continue
        worker_status[worker_idx] = 1

        # Update the refiner step
        refiner.step = job['step']

        # Generate the result
        loss, p_grads, X_pred = calculate_gradients(
            refiner_args=RefinerArgs.from_args(job['refiner_args']),
            p_vec=init_tensor(job['p_vec']),
            X_target=init_tensor(job['X_target']),
            X_target_denoised=init_tensor(job['X_target_denoised']),
            X_target_wis=init_tensor(job['X_target_wis']),
            X_target_denoised_wis=init_tensor(job['X_target_denoised_wis']),
            keypoints=init_tensor(job['keypoints']) if job['keypoints'] is not None else None
        )

        # Return the result through the response queue
        response_queue.put({
            'batch_idx': job['batch_idx'],
            'loss': loss.item(),
            'p_grads': to_numpy(p_grads),
            'X_pred': to_numpy(X_pred),
            'crystal': scene.crystal.to_dict(),
            'projector_zoom': orthographic_scale_factor(scene),
        })


class RefinerPool:
    def __init__(
            self,
            refiner_args: RefinerArgs,
            output_dir: Path,
            initial_scene_dict: Dict[str, Any],
            fixed_parameters: Dict[str, Tensor],
            seed: int | None,
            plotter: SequencePlotter,

            n_workers: int = N_WORKERS,
            queue_size: int = 100
    ):
        self.refiner_args = refiner_args
        self.output_dir = output_dir
        self.initial_scene_dict = initial_scene_dict
        self.fixed_parameters = fixed_parameters
        self.seed = seed
        self.plotter = plotter
        self.workers = {}
        self.n_workers = n_workers

        if n_workers > 0:
            # Ensure that CUDA will work in subprocesses
            mp.set_start_method('spawn', force=True)

            # Set up the queues
            self.job_queue = mp.Queue(maxsize=queue_size)
            self.response_queue = mp.Queue()

            # Start the worker processes
            self.stop_event = mp.Event()
            self._start_workers()

    def _start_workers(self):
        """
        Start the worker processes.
        """
        self.worker_status = mp.Array('i', [1] * self.n_workers)

        worker_args = {
            'refiner_args': self.refiner_args,
            'output_dir': self.output_dir,
            'initial_scene_dict': self.initial_scene_dict,
            'fixed_parameters': self.fixed_parameters,
            'seed': self.seed,
            'job_queue': self.job_queue,
            'response_queue': self.response_queue,
            'stop_event': self.stop_event,
            'worker_status': self.worker_status
        }

        for i in range(self.n_workers):
            process = mp.Process(
                target=refiner_worker,
                kwargs={**worker_args, 'worker_idx': i}
            )
            start_process(process)

    def all_workers_idle(self):
        """
        Check if all workers are idle.
        """
        return all(status == 0 for status in self.worker_status)

    def wait_for_workers(self):
        """
        Wait for all workers to complete.
        """
        if not self.all_workers_idle():
            logger.info('Waiting for refiner workers...')
            while not self.all_workers_idle():
                time.sleep(1)
            logger.info('Refiner workers ready.')

    def calculate_grads(
            self,
            step: int,
            refiner_args: RefinerArgs,
            p_vec_batch: Tensor,
            X_target: Tensor,
            X_target_denoised: Tensor,
            X_target_wis: Tensor,
            X_target_denoised_wis: Tensor,
            keypoints: List[Tensor] | None,
            X_preds_paths: List[Path],
            X_targets_paths: List[Path],
            X_targets_annotated_paths: List[Path],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the loss and parameter gradients for a single example.
        """
        if self.n_workers < 0:
            raise NotImplementedError('RefinerPool is only configured for parallel processing.')

        for idx, p_vec in enumerate(p_vec_batch):
            job = {
                'step': step,
                'batch_idx': idx,
                'refiner_args': refiner_args.to_dict(),
                'p_vec': to_numpy(p_vec),
                'X_target': to_numpy(X_target[idx]),
                'X_target_denoised': to_numpy(X_target_denoised[idx]),
                'X_target_wis': to_numpy(X_target_wis[idx]),
                'X_target_denoised_wis': to_numpy(X_target_denoised_wis[idx]),
                'keypoints': keypoints[idx]
            }
            while True:
                try:
                    self.job_queue.put(job, block=False)
                    break
                except Full:
                    time.sleep(0.2)
                    continue

        # Wait for the results
        results = []
        while True:
            try:
                result = self.response_queue.get(timeout=1)
                results.append(result)

                # Send the results to the plotter immediately
                idx = result['batch_idx']
                if result['X_pred'] is not None:
                    self.plotter.save_image(result['X_pred'], X_preds_paths[idx])
                self.plotter.annotate_image(
                    X_targets_paths[idx],
                    X_targets_annotated_paths[idx],
                    zoom=result['projector_zoom'],
                    crystal=result['crystal']
                )

                # Break if we have all the results
                if len(results) == len(p_vec_batch):
                    break
            except Empty:
                continue

        # Sort the results by index
        results = sorted(results, key=lambda x: x['batch_idx'])
        losses = init_tensor([r['loss'] for r in results])
        p_grads = init_tensor(np.stack([r['p_grads'] for r in results]))
        X_preds = init_tensor(np.stack([r['X_pred'] for r in results]))

        return losses, p_grads, X_preds
