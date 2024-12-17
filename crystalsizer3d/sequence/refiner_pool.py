import json
import multiprocessing as mp
import os
import time
from datetime import timedelta
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Tuple

import mitsuba as mi
import numpy as np
import torch
from PIL import Image
from filelock import SoftFileLock as FileLock
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torchvision.transforms.functional import to_tensor

from crystalsizer3d import N_WORKERS, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import orthographic_scale_factor
from crystalsizer3d.sequence.sequence_plotter import SequencePlotter
from crystalsizer3d.util.parallelism import start_process, stop_event as global_stop_event
from crystalsizer3d.util.utils import FlexibleJSONEncoder, init_tensor, set_seed, to_numpy


def append_to_shared_json(file_path: Path, idx: int | str, new_data: dict | float, timeout: int = 60):
    """
    Append data to a shared JSON file.
    """
    if not file_path.exists():
        with open(file_path, 'w') as f:
            json.dump({}, f)
    lock = FileLock(file_path.with_suffix('.lock'), timeout=timeout)
    lock.acquire()
    with open(file_path, 'r') as f:
        data = json.load(f)
    idx = str(idx)
    if len(data) > 0:
        assert idx not in data, f'Index {idx} already exists in {file_path}!'
    data[idx] = new_data
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, cls=FlexibleJSONEncoder)
    lock.release()


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
    refiner.save_dir = output_dir
    if seed is not None:
        set_seed(seed)

    # Load the first scene and adapt the crystal to use buffers instead of parameters
    scene = Scene.from_dict(initial_scene_dict)
    crystal = scene.crystal

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

    def _parameters_to_buffers():
        if len(crystal._parameters) == 0:
            return
        for k in list(crystal._parameters.keys()):
            val = crystal.get_parameter(k).data
            del crystal._parameters[k]
            crystal.register_buffer(k, val)

    def _buffers_to_parameters():
        if len(crystal._parameters) == 0:
            for k in PARAMETER_KEYS:
                val = crystal.get_buffer(k)
                del crystal._buffers[k]
                crystal.register_parameter(k, nn.Parameter(val, requires_grad=True))
        if not isinstance(scene.light_radiance, nn.Parameter):
            scene.light_radiance = nn.Parameter(
                init_tensor(scene.light_radiance, device=refiner.device),
                requires_grad=True
            )

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

    def update_refiner(
            refiner_args: RefinerArgs,
            X_target: Path,
            X_target_denoised: Path,
            X_target_wis: Path | Tensor,
            X_target_denoised_wis: Path | Tensor,
            keypoints: Tensor | None,
            edges: Path | None,
            distances_target: Tensor | None,
            vertices_target: Tensor | None
    ):
        """
        Update the refiner args and targets.
        """
        # Update args
        refiner.args = refiner_args

        # Update targets
        refiner.X_target = X_target
        refiner.X_target_denoised = X_target_denoised
        if isinstance(X_target_wis, Path):
            X_target_wis = to_tensor(Image.open(X_target_wis))
        if isinstance(X_target_wis, np.ndarray):
            X_target_wis = init_tensor(X_target_wis)
        if X_target_wis.shape[0] == 3:
            X_target_wis = X_target_wis.permute(1, 2, 0)
        refiner.X_target_wis = X_target_wis.to(refiner.device)
        if isinstance(X_target_denoised_wis, Path):
            X_target_denoised_wis = to_tensor(Image.open(X_target_denoised_wis))
        if isinstance(X_target_denoised_wis, np.ndarray):
            X_target_denoised_wis = init_tensor(X_target_denoised_wis)
        if X_target_denoised_wis.shape[0] == 3:
            X_target_denoised_wis = X_target_denoised_wis.permute(1, 2, 0)
        refiner.X_target_denoised_wis = X_target_denoised_wis.to(refiner.device)
        if refiner_args.use_keypoints:
            refiner.keypoint_targets = init_tensor(keypoints) if keypoints is not None else None
        if refiner_args.use_edge_matching:
            refiner.edge_map = to_tensor(Image.open(edges)).squeeze()
        refiner.distances_target = init_tensor(distances_target) if distances_target is not None else None
        refiner.vertices_target = init_tensor(vertices_target) if vertices_target is not None else None

    def calculate_losses(
            p_vec: Tensor,
            add_noise: bool
    ):
        """
        Calculate losses for the given parameters and targets
        """
        # Update scene and crystal parameters
        p_dict = _parameter_vector_to_dict(p_vec)
        for k, v in p_dict.items():
            if k == 'light_radiance':
                refiner.scene.light_radiance = v.to(refiner.device)
            else:
                setattr(refiner.scene.crystal, k, v.to('cpu'))

        # Calculate losses
        loss, stats = refiner.process_step(add_noise=add_noise)

        # Return the rendered image if it was generated
        X_pred = refiner.X_pred if refiner.args.use_inverse_rendering else None

        return loss, stats, X_pred

    def refine_parameters(
            p_vec_init: Tensor,
            plots_path: Path
    ):
        """
        Iteratively refine the parameters.
        """
        # Set initial scene and crystal parameters
        p_dict_init = _parameter_vector_to_dict(p_vec_init)
        for k, v in p_dict_init.items():
            if k == 'light_radiance':
                refiner.scene.light_radiance.data = v.to(refiner.device)
            else:
                getattr(refiner.scene.crystal, k).data = v.to('cpu')

        def _make_plot(reprocess=True):
            with torch.no_grad():
                if reprocess:
                    refiner.process_step(add_noise=False, no_crop_render=True)
                fig = refiner._plot_comparison()
            plots_path.mkdir(parents=True, exist_ok=True)
            path = plots_path / f'{refiner.step + 1:08d}.png'
            plt.savefig(path, bbox_inches='tight')
            plt.close(fig)

        n_steps = refiner.args.max_steps
        start_step = refiner.step
        refiner.step = start_step - 1
        end_step = start_step + n_steps
        log_freq = refiner.args.log_every_n_steps
        running_loss = 0.
        running_tps = 0
        use_inverse_rendering = refiner.args.use_inverse_rendering  # Save the inverse rendering setting
        use_edge_matching = refiner.args.use_edge_matching  # Save the edge matching setting

        # Needed for plotting only
        refiner.X_target = to_tensor(Image.open(refiner.X_target)).permute(1, 2, 0).to(refiner.device)
        refiner.X_target_denoised = to_tensor(Image.open(refiner.X_target_denoised)).permute(1, 2, 0).to(refiner.device)

        # (Re-)initialise the optimiser
        refiner._init_optimiser(quiet=True)

        # Plot initial prediction
        if start_step == 0:
            _make_plot()
        for step in range(start_step, end_step):
            start_time = time.time()
            refiner.step = step

            # Conditionally enable the inverse rendering and edge matching
            refiner.args.use_inverse_rendering = use_inverse_rendering and step >= refiner.args.ir_wait_n_steps
            refiner.args.use_edge_matching = use_edge_matching and step >= refiner.args.edge_matching_wait_n_steps

            # Train for a single step
            loss, stats = refiner._train_step(log_to_tb=False)

            # Adjust tracking loss to include the IR loss placeholder
            loss_track = loss.detach().cpu()
            if use_inverse_rendering and not refiner.args.use_inverse_rendering and refiner.args.ir_loss_placeholder > 0:
                loss_track += refiner.args.ir_loss_placeholder

            # Track running loss and time per step
            running_loss += loss
            time_per_step = time.time() - start_time
            running_tps += time_per_step

            # Log statistics every X steps
            if (step + 1) % log_freq == 0:
                average_tps = running_tps / log_freq
                seconds_left = float((end_step - step) * average_tps)
                tps = 'tps: {}, rem: {}'.format(
                    str(timedelta(seconds=average_tps)),
                    str(timedelta(seconds=seconds_left)))
                logger.info(f'{worker_id}: [{step + 1}/{end_step}]\tLoss: {running_loss / log_freq:.4E}\t{tps}')
                running_loss = 0.
                running_tps = 0

            # Plots
            if refiner.args.plot_every_n_steps > -1 and (refiner.step + 1) % refiner.args.plot_every_n_steps == 0:
                _make_plot()

        # Reprocess without noise to get the loss and stats to return
        loss, stats = refiner.process_step(add_noise=False, no_crop_render=True)

        # Final plots
        _make_plot(reprocess=False)

        params = {
            'scale': refiner.crystal.scale,
            'distances': refiner.crystal.distances,
            'origin': refiner.crystal.origin,
            'rotation': refiner.crystal.rotation,
            'material_roughness': refiner.crystal.material_roughness,
            'material_ior': refiner.crystal.material_ior,
            'light_radiance': refiner.scene.light_radiance,
        }

        # Restore the inverse rendering and edge matching settings
        refiner.args.use_inverse_rendering = use_inverse_rendering
        refiner.args.use_edge_matching = use_edge_matching
        logger.info(f'{worker_id}: Training complete.')

        return loss.item(), stats, params

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

        # Update the refiner
        refiner.step = job['step']
        update_refiner(
            refiner_args=RefinerArgs.from_args(job['refiner_args']),
            X_target=job['X_target'],
            X_target_denoised=job['X_target_denoised'],
            X_target_wis=job['X_target_wis'],
            X_target_denoised_wis=job['X_target_denoised_wis'],
            keypoints=job['keypoints'],
            edges=job['edges'],
            distances_target=job['distances_target'],
            vertices_target=job['vertices_target']
        )

        # Calculate losses / gradients for a single step
        if job['task'] == 'calculate_losses':
            # Load the parameters and enable gradient tracking if required
            _parameters_to_buffers()
            p_vec = init_tensor(job['p_vec'])
            if job['calculate_grads']:
                p_vec.requires_grad = True
            else:
                grad_enabled = torch.is_grad_enabled()
                torch.set_grad_enabled(False)

            # Calculate the losses
            loss, stats, X_pred = calculate_losses(
                p_vec=p_vec,
                add_noise=job['calculate_grads']
            )

            # Calculate grads
            if job['calculate_grads']:
                loss.backward()
                p_grads = to_numpy(p_vec.grad.clone())
            else:
                p_grads = None
                torch.set_grad_enabled(grad_enabled)  # Reset the gradient tracking

            if refiner_args.use_edge_matching:
                edge_points = to_numpy(refiner.edge_matcher.edge_points_rel)
                edge_point_deltas = to_numpy(refiner.edge_matcher.deltas_rel)
            else:
                edge_points = None
                edge_point_deltas = None

            # Return the result through the response queue
            response_queue.put({
                'batch_idx': job['batch_idx'],
                'loss': loss.item(),
                'stats': stats,
                'p_grads': p_grads,
                'X_pred': to_numpy(X_pred) if X_pred is not None else None,
                'crystal': scene.crystal.to_dict(),
                'projector_zoom': orthographic_scale_factor(scene),
                'edge_points': edge_points,
                'edge_point_deltas': edge_point_deltas
            })

        # Iteratively refine the parameters
        elif job['task'] == 'refine':
            _buffers_to_parameters()
            loss, stats, params = refine_parameters(
                p_vec_init=init_tensor(job['p_vec']),
                plots_path=job['plots_path']
            )
            append_to_shared_json(job['losses_path'], job['idx'], loss)
            append_to_shared_json(job['stats_path'], job['idx'], stats)
            append_to_shared_json(job['parameters_path'], job['idx'], params)

    # Mark as idle and exit
    worker_status[worker_idx] = 0


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
            'job_queue': self.job_queue,
            'response_queue': self.response_queue,
            'stop_event': self.stop_event,
            'worker_status': self.worker_status
        }

        for i in range(self.n_workers):
            process = mp.Process(
                target=refiner_worker,
                kwargs={**worker_args, 'worker_idx': i, 'seed': self.seed + i}
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

    def close(self):
        """
        Close the worker processes.
        """
        logger.info('Closing refiner pool.')
        self.stop_event.set()
        self.wait_for_workers()

    def calculate_losses(
            self,
            step: int,
            refiner_args: RefinerArgs,
            p_vec_batch: Tensor,
            X_target: List[Path],
            X_target_denoised: List[Path],
            X_target_wis: Tensor,
            X_target_denoised_wis: Tensor,
            keypoints: List[Tensor] | None,
            edges: List[Path] | None,
            distances_target: Tensor | None,
            vertices_target: List[Tensor] | None,
            calculate_grads: bool,
            save_annotations: bool,
            save_edge_annotations: bool,
            save_renders: bool,
            X_preds_paths: List[Path],
            X_targets_paths: List[Path],
            X_targets_annotated_paths: List[Path],
            edges_fullsize_paths: List[Path] | None,
            edges_annotated_paths: List[Path] | None
    ) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        """
        Calculate the loss and parameter gradients for a single example.
        """
        if self.n_workers < 0:
            raise NotImplementedError('RefinerPool is only configured for parallel processing.')

        for idx, p_vec in enumerate(p_vec_batch):
            job = {
                'task': 'calculate_losses',
                'step': step,
                'batch_idx': idx,
                'refiner_args': refiner_args.to_dict(),
                'p_vec': to_numpy(p_vec_batch[idx]),
                'X_target': X_target[idx],
                'X_target_denoised': X_target_denoised[idx],
                'X_target_wis': to_numpy(X_target_wis[idx]),
                'X_target_denoised_wis': to_numpy(X_target_denoised_wis[idx]),
                'keypoints': to_numpy(keypoints[idx]) if keypoints[idx] is not None else None,
                'edges': edges[idx],
                'distances_target': to_numpy(distances_target[idx]) if distances_target is not None else None,
                'vertices_target': to_numpy(vertices_target[idx]) if vertices_target is not None else None,
                'calculate_grads': calculate_grads,
            }
            self.job_queue.put(job, block=True)

        # Wait for the results
        results = []
        while True:
            try:
                result = self.response_queue.get(timeout=1)
                results.append(result)

                # Send the results to the plotter immediately if required
                idx = result['batch_idx']
                if save_renders and result['X_pred'] is not None:
                    self.plotter.save_image(result['X_pred'], X_preds_paths[idx])
                if save_annotations:
                    self.plotter.annotate_image(
                        X_targets_paths[idx],
                        X_targets_annotated_paths[idx],
                        zoom=result['projector_zoom'],
                        crystal=result['crystal']
                    )
                if save_edge_annotations and self.refiner_args.use_edge_matching:
                    self.plotter.annotate_image(
                        edges_fullsize_paths[idx],
                        edges_annotated_paths[idx],
                        zoom=result['projector_zoom'],
                        crystal=result['crystal'],
                        keypoints=to_numpy(keypoints[idx]) if keypoints[idx] is not None else None,
                        edge_points=result['edge_points'],
                        edge_point_deltas=result['edge_point_deltas']
                    )

                # Break if we have all the results
                if len(results) == len(p_vec_batch):
                    break
            except Empty:
                continue

        # Sort the results by index
        results = sorted(results, key=lambda x: x['batch_idx'])
        losses = init_tensor([r['loss'] for r in results])
        stats = {k: [r['stats'][k] for r in results] for k in results[0]['stats']}
        if calculate_grads:
            p_grads = init_tensor(np.stack([r['p_grads'] for r in results]))
        else:
            p_grads = None

        return losses, stats, p_grads

    def refine(
            self,
            refiner_args: RefinerArgs,
            start_step: int,
            idx: int,
            p_vec: Tensor,
            X_target: Path,
            X_target_denoised: Path,
            X_target_wis: Path,
            X_target_denoised_wis: Path,
            keypoints: Tensor | None,
            edges: Path | None,
            distances_target: Tensor | None,
            vertices_target: Tensor | None,
            losses_path: Path,
            stats_path: Path,
            parameters_path: Path,
            plots_path: Path,
    ):
        """
        Iteratively refine some initial parameters and then save the result.
        """
        if self.n_workers < 0:
            raise NotImplementedError('RefinerPool is only configured for parallel processing.')
        job = {
            'task': 'refine',
            'refiner_args': refiner_args.to_dict(),
            'step': start_step,
            'idx': int(idx),
            'p_vec': to_numpy(p_vec),
            'X_target': X_target,
            'X_target_denoised': X_target_denoised,
            'X_target_wis': X_target_wis,
            'X_target_denoised_wis': X_target_denoised_wis,
            'keypoints': to_numpy(keypoints) if keypoints is not None else None,
            'edges': edges,
            'distances_target': to_numpy(distances_target) if distances_target is not None else None,
            'vertices_target': to_numpy(vertices_target) if vertices_target is not None else None,
            'losses_path': losses_path,
            'stats_path': stats_path,
            'parameters_path': parameters_path,
            'plots_path': plots_path
        }
        self.job_queue.put(job, block=True)

        # Wait until the job has been picked up before returning
        time.sleep(0.1)
        while self.all_workers_idle():
            time.sleep(0.1)
