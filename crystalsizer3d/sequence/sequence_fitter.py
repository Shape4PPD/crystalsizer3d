import gc
import json
import math
import os
import shutil
import time
from argparse import Namespace
from datetime import timedelta
from typing import Dict

import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import center_crop, to_tensor

from crystalsizer3d import DATA_PATH, logger
from crystalsizer3d.args.refiner_args import DENOISER_ARG_NAMES, KEYPOINTS_ARG_NAMES, PREDICTOR_ARG_NAMES, RefinerArgs
from crystalsizer3d.args.sequence_fitter_args import SequenceFitterArgs
from crystalsizer3d.refiner.denoising import denoise_image
from crystalsizer3d.refiner.keypoint_detection import find_keypoints
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.sequence.adaptive_sampler import AdaptiveSampler
from crystalsizer3d.sequence.plots import annotate_image, plot_areas, plot_distances
from crystalsizer3d.sequence.sequence_encoder import SequenceEncoder
from crystalsizer3d.sequence.utils import get_image_paths
from crystalsizer3d.util.utils import FlexibleJSONEncoder, calculate_model_norm, hash_data, init_tensor, is_bad, \
    json_to_torch, set_seed, to_dict, to_numpy

PARAMETER_KEYS = ['scale', 'distances', 'origin', 'rotation', 'material_roughness', 'material_ior', 'light_radiance']
SEQUENCE_DATA_PATH = DATA_PATH / 'sequences'

ARG_NAMES = {
    'denoiser': DENOISER_ARG_NAMES,
    'keypoints': KEYPOINTS_ARG_NAMES,
    'predictor': PREDICTOR_ARG_NAMES,
}


class SequenceFitter:
    def __init__(
            self,
            sf_args: SequenceFitterArgs,
            refiner_args: RefinerArgs,
            runtime_args: Namespace,
    ):
        self.sf_args = sf_args
        self.refiner_args = refiner_args
        self.runtime_args = runtime_args
        if self.sf_args.seed is not None:
            set_seed(self.sf_args.seed)
        self.image_paths = get_image_paths(sf_args, load_all=True)
        self.image_idxs = torch.tensor([p[0] for p in self.image_paths])
        self.image_size_full = to_tensor(Image.open(self.image_paths[0][1])).shape[-2:]
        self.image_size = min(self.image_size_full)

        # Initialise the output directories
        self.base_path = SEQUENCE_DATA_PATH / f'{sf_args.images_dir.stem}_{hash_data(sf_args.images_dir.absolute())}'
        self.path = self.base_path / 'sf' / hash_data([sf_args.to_dict(), refiner_args.to_dict()])
        self.path.mkdir(parents=True, exist_ok=True)
        self._init_output_dirs()

        # Initialise the refiner
        self._init_refiner()

        # Initialise the target data
        self._init_X_targets()
        self._init_X_targets_denoised()
        self._init_keypoints()

        # Initialise the outputs
        self._init_initial_predictions()
        self._init_scenes()
        self._init_parameters()
        self._init_losses()
        self._init_frame_counts()
        self._init_X_preds()
        self._init_X_targets_annotated()
        self._init_tb_logger()

        # Initialise the sequence encoder
        self._init_sequence_encoder()

    def _init_output_dirs(self):
        """
        Ensure the output directories exists with the correct arguments.
        """
        self.cache_dirs = {}
        for model_name, arg_names in ARG_NAMES.items():
            model_args = {k: getattr(self.refiner_args, k) for k in arg_names}
            model_id = model_args[f'{model_name}_model_path'].stem[:4]
            model_cache_dir = self.base_path / model_name / f'{model_id}_{hash_data(model_args)}'
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dirs[model_name] = model_cache_dir
            with open(model_cache_dir / 'args.yml', 'w') as f:
                yaml.dump(model_args, f)

        # Create a refiner cache directory unique to the component models' args only
        refiner_id = hash_data({k: getattr(self.refiner_args, k) for k2 in ARG_NAMES.values() for k in k2})
        refiner_cache_dir = self.base_path / 'refiner' / refiner_id
        refiner_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dirs['refiner'] = refiner_cache_dir
        with open(refiner_cache_dir / 'args.yml', 'w') as f:
            yaml.dump(to_dict(self.refiner_args), f)

        # Clear the sequence directory if it already exists and we're not resuming
        if not self.runtime_args.resume and self.path.exists():
            shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)

        # Save the args into the sequence directory
        with open(self.path / 'args_runtime.yml', 'w') as f:
            yaml.dump(to_dict(self.runtime_args), f)
        with open(self.path / 'args_sequence_fitter.yml', 'w') as f:
            yaml.dump(to_dict(self.sf_args), f)
        with open(self.path / 'args_refiner.yml', 'w') as f:
            yaml.dump(to_dict(self.refiner_args), f)

    def _init_refiner(self):
        """
        Instantiate the refiner.
        """
        logger.info('Initialising refiner.')
        idx0, img_path0 = self.image_paths[0]
        self.refiner_args.image_path = img_path0
        self.refiner = Refiner(
            args=self.refiner_args,
            output_dir=self.cache_dirs['refiner'] / f'image_{idx0:04d}',
            do_init=False,
        )
        self.manager = self.refiner.manager
        self.device = self.refiner.device

    def _init_X_targets(self):
        """
        Instantiate the cropped target images.
        """
        X_targets_dir = self.base_path / 'X_targets'
        if X_targets_dir.exists():
            try:
                self.X_targets = []
                for (idx, _) in self.image_paths:
                    X_target_path = X_targets_dir / f'{idx:04d}.png'
                    assert X_target_path.exists()
                    self.X_targets.append(X_target_path)
                logger.info(f'Loaded cropped target images from {X_targets_dir}.')
                return
            except Exception:
                pass

        # Generate the square target images
        logger.info('Generating square target images.')
        X_targets_dir.mkdir(parents=True, exist_ok=True)
        img_size = self.image_size
        for i, (idx, image_path) in enumerate(self.image_paths):
            if (i + 1) % 50 == 0:
                logger.info(f'Generating target image {i + 1}/{len(self)}.')
            X = to_tensor(Image.open(image_path))
            if X.shape[0] == 4:
                assert torch.allclose(X[3], torch.ones_like(X[3])), 'Transparent images not supported.'
                X = X[:3]
            assert min(X.shape[-2:]) == img_size, 'All images must have the same size.'
            X = center_crop(X, [img_size, img_size])
            X_target = to_numpy(X.permute(1, 2, 0))
            Image.fromarray((X_target * 255).astype(np.uint8)).save(X_targets_dir / f'{idx:04d}.png')

        return self._init_X_targets()

    @torch.no_grad()
    def _init_X_targets_denoised(self):
        """
        Instantiate the denoised data.
        """
        X_targets_denoised_dir = self.cache_dirs['denoiser']
        try:
            self.X_targets_denoised = []
            for (idx, _) in self.image_paths:
                X_target_denoised_path = X_targets_denoised_dir / f'{idx:04d}.png'
                assert X_target_denoised_path.exists()
                self.X_targets_denoised.append(X_target_denoised_path)
            logger.info(f'Loaded denoised images from {X_targets_denoised_dir}.')
            return
        except Exception:
            pass

        # Load the denoiser model
        assert self.refiner_args.denoiser_model_path.exists(), f'Denoiser model path does not exist: {self.refiner_args.denoiser_model_path}.'
        self.manager.load_network(self.refiner_args.denoiser_model_path, 'denoiser')

        # Denoise the images
        logger.info('Denoising image sequence.')
        X_targets_denoised_dir.mkdir(parents=True, exist_ok=True)
        for i, X_target_path in enumerate(self.X_targets):
            if (i + 1) % 10 == 0:
                logger.info(f'Denoising image {i + 1}/{len(self)}.')
            X = to_tensor(Image.open(X_target_path))
            X_denoised = denoise_image(
                manager=self.manager,
                X=X,
                n_tiles=self.refiner_args.denoiser_n_tiles,
                overlap=self.refiner_args.denoiser_tile_overlap,
                oversize_input=self.refiner_args.denoiser_oversize_input,
                max_img_size=self.refiner_args.denoiser_max_img_size,
                batch_size=self.refiner_args.denoiser_batch_size,
                return_patches=False
            )
            X_target_denoised = to_numpy(X_denoised.permute(1, 2, 0))
            idx = self.image_paths[i][0]
            Image.fromarray((X_target_denoised * 255).astype(np.uint8)).save(X_targets_denoised_dir / f'{idx:04d}.png')

        # Destroy the denoiser to free up space
        logger.info('Destroying denoiser to free up space.')
        self.manager.denoiser = None
        torch.cuda.empty_cache()
        gc.collect()

        return self._init_X_targets_denoised()

    @torch.no_grad()
    def _init_keypoints(self):
        """
        Instantiate the keypoints.
        """
        ra = self.refiner_args
        if not ra.use_keypoints:
            self.keypoints = None
            return
        keypoints_dir = self.cache_dirs['keypoints']
        if keypoints_dir.exists():
            try:
                self.keypoints = []
                for i, (idx, _) in enumerate(self.image_paths):
                    keypoints_path = keypoints_dir / f'{idx:04d}.json'
                    assert keypoints_path.exists()
                    with open(keypoints_path) as f:
                        keypoints = json_to_torch(json.load(f))
                    self.keypoints.append({
                        'path': keypoints_path,
                        'keypoints': keypoints,
                    })
                logger.info(f'Loaded keypoints from {keypoints_dir}.')
                return
            except Exception:
                pass

        # Load the keypoint detector
        self.manager.load_network(ra.keypoints_model_path, 'keypointdetector')
        assert self.manager.keypoint_detector is not None, 'No keypoints model loaded, so can\'t predict keypoints.'

        # Find the keypoints for each image
        logger.info('Finding keypoints across image sequence.')
        for i, (X_path, X_denoised_path) in enumerate(zip(self.X_targets, self.X_targets_denoised)):
            if (i + 1) % 10 == 0:
                logger.info(f'Finding keypoints for image {i + 1}/{len(self)}.')
            X = to_tensor(Image.open(X_path))
            X_dn = to_tensor(Image.open(X_denoised_path))
            res = find_keypoints(
                X_target=X,
                X_target_denoised=X_dn,
                manager=self.manager,
                oversize_input=ra.keypoints_oversize_input,
                max_img_size=ra.keypoints_max_img_size,
                batch_size=ra.keypoints_batch_size,
                min_distance=ra.keypoints_min_distance,
                threshold=ra.keypoints_threshold,
                exclude_border=ra.keypoints_exclude_border,
                blur_kernel_relative_size=ra.keypoints_blur_kernel_relative_size,
                n_patches=ra.keypoints_n_patches,
                patch_size=ra.keypoints_patch_size,
                patch_search_res=ra.keypoints_patch_search_res,
                attenuation_sigma=ra.keypoints_attenuation_sigma,
                max_attenuation_factor=ra.keypoints_max_attenuation_factor,
                low_res_catchment_distance=ra.keypoints_low_res_catchment_distance,
                return_everything=False,
                quiet=True
            )
            idx = self.image_paths[i][0]
            with open(keypoints_dir / f'{idx:04d}.json', 'w') as f:
                json.dump(res, f, cls=FlexibleJSONEncoder)

        # Destroy the keypoint detector to free up space
        logger.info('Destroying keypoint detector to free up space.')
        self.manager.keypoint_detector = None
        torch.cuda.empty_cache()
        gc.collect()

        return self._init_keypoints()

    def _init_initial_predictions(self):
        """
        Instantiate the initial predictions.
        """
        scenes_initial_dir = self.cache_dirs['predictor'] / 'scenes_initial'
        X_preds_initial_dir = self.cache_dirs['predictor'] / 'X_preds_initial'
        X_annotated_initial_dir = self.cache_dirs['predictor'] / 'X_targets_annotated_initial'
        if scenes_initial_dir.exists() and X_preds_initial_dir.exists() and X_annotated_initial_dir.exists():
            try:
                self.scenes_initial = []
                self.X_preds_initial = []
                self.X_targets_annotated_initial = []
                for i, (idx, _) in enumerate(self.image_paths):
                    scene_path = scenes_initial_dir / f'{idx:04d}.yml'
                    X_pred_path = X_preds_initial_dir / f'{idx:04d}.png'
                    X_annotated_path = X_annotated_initial_dir / f'{idx:04d}.png'
                    assert scene_path.exists() and X_pred_path.exists() and X_annotated_path.exists()
                    self.scenes_initial.append(scene_path)
                    self.X_preds_initial.append(X_pred_path)
                    self.X_targets_annotated_initial.append(X_annotated_path)
                logger.info(f'Loaded initial predictions from {self.cache_dirs["predictor"]}.')
                return
            except Exception:
                pass
        logger.info('Generating initial predictions.')

        # Clean up if anything exists already
        if scenes_initial_dir.exists():
            shutil.rmtree(scenes_initial_dir)
        if X_preds_initial_dir.exists():
            shutil.rmtree(X_preds_initial_dir)
        if X_annotated_initial_dir.exists():
            shutil.rmtree(X_annotated_initial_dir)
        scenes_initial_dir.mkdir(parents=True, exist_ok=True)
        X_preds_initial_dir.mkdir(parents=True, exist_ok=True)
        X_annotated_initial_dir.mkdir(parents=True, exist_ok=True)

        # Sinkhorn loss re-enables grad so disable and re-enable it manually
        # https://github.com/jeanfeydy/geomloss/issues/57
        prev_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        # Ensure the inverse rendering losses are turned on
        use_inverse_rendering = self.refiner_args.use_inverse_rendering
        self.refiner_args.use_inverse_rendering = True
        use_latents_model = self.refiner_args.use_latents_model
        self.refiner_args.use_latents_model = self.refiner_args.w_latent > 0
        use_perceptual_model = self.refiner_args.use_perceptual_model
        self.refiner_args.use_perceptual_model = self.refiner_args.w_perceptual > 0

        # Set up the resizing
        resize_args = dict(mode='bilinear', align_corners=False)
        resize_input_old = self.manager.predictor.resize_input
        self.manager.predictor.resize_input = False
        oversize_input = self.refiner_args.initial_pred_oversize_input
        img_size = self.refiner_args.initial_pred_max_img_size if oversize_input else self.manager.image_shape[-1]
        render_size = self.manager.crystal_renderer.dataset_args.image_size  # Use the rendering size the predictor was trained on

        # Noise batch vars
        bs = self.refiner_args.initial_pred_batch_size // 2
        noise_scale = torch.linspace(
            self.refiner_args.initial_pred_noise_min,
            self.refiner_args.initial_pred_noise_max,
            bs,
            device=self.device
        )

        def update_scene_parameters(scene_: Scene):
            # Update the scene rendering parameters for optimisation
            scene_.res = self.refiner_args.rendering_size
            scene_.spp = self.refiner_args.spp
            scene_.camera_type = 'perspective'  # thinlens doesn't work for inverse rendering
            scene_.integrator_max_depth = self.refiner_args.integrator_max_depth
            scene_.integrator_rr_depth = self.refiner_args.integrator_rr_depth
            scene_.build_mi_scene()
            scene_.crystal.to('cpu')
            return scene_

        for i, (X_path, X_dn_path) in enumerate(zip(self.X_targets, self.X_targets_denoised)):
            if (i + 1) % 10 == 0:
                logger.info(f'Generating initial predictions for frame {i + 1}/{len(self)}.')
            torch.set_grad_enabled(False)
            X = to_tensor(Image.open(X_path)).to(self.device)
            X_dn = to_tensor(Image.open(X_dn_path)).to(self.device)
            X_target = torch.stack([X, X_dn])

            # Resize the input if needed
            if oversize_input and X_target.shape[-1] > img_size \
                    or not oversize_input and X_target.shape[-1] != img_size:
                X_target = F.interpolate(X_target, size=img_size, **resize_args)

            # Set the target image for loss calculations
            self.refiner.X_target_aug = self.refiner._resize(X_dn, render_size).permute(1, 2, 0)

            # Create a noisy batch
            X_target_batch = X_target[None, ...].repeat(bs, 1, 1, 1, 1)
            X_target_batch += torch.randn_like(X_target_batch) * noise_scale[:, None, None, None, None]
            X_target_batch = torch.cat([X_target_batch[:, 0], X_target_batch[:, 1]])

            # Predict the parameters
            Y_pred_batch = self.manager.predict(X_target_batch)

            # Render the predicted parameters
            scene_i_batch = []
            losses_i = []
            for j in range(bs):
                torch.set_grad_enabled(False)

                # Render the image
                r_params = self.manager.ds.denormalise_rendering_params(Y_pred_batch, idx=j)
                X_pred_ij, scene_ij = self.manager.crystal_renderer.render_from_parameters(r_params, return_scene=True)
                scene_i_batch.append(scene_ij)

                # Set the variables temporarily
                X_pred_ij = X_pred_ij.astype(np.float32) / 255.
                X_pred_ij = torch.from_numpy(X_pred_ij).to(self.device)
                self.refiner.X_pred = X_pred_ij
                self.refiner.scene = scene_ij
                self.refiner.crystal = scene_ij.crystal
                self.refiner.crystal.to('cpu')

                # Project the crystal mesh - reinitialise the projector for the new scene
                if self.refiner_args.use_keypoints:
                    self.refiner._init_projector()
                    self.refiner.projector.project(generate_image=False)
                    self.refiner.keypoint_targets = self.keypoints[i]['keypoints']

                # Calculate losses
                loss_ij, _ = self.refiner._calculate_losses()
                losses_i.append(loss_ij.item())

            # Update the scene and render the best initial prediction at the correct resolution
            best_idx = np.argmin(losses_i)
            scene = scene_i_batch[best_idx]
            scene = update_scene_parameters(scene)
            X_pred_initial = scene.render()

            # Save the parameters and images
            idx = self.image_paths[i][0]
            scene.to_yml(scenes_initial_dir / f'{idx:04d}.yml')
            X_pred_initial_path = X_preds_initial_dir / f'{idx:04d}.png'
            Image.fromarray(X_pred_initial).save(X_pred_initial_path)
            annotate_image(X_path, scene).save(X_annotated_initial_dir / f'{idx:04d}.png')

        # Restore the settings
        del self.refiner.projector
        self.manager.predictor.resize_input = resize_input_old
        self.refiner_args.use_inverse_rendering = use_inverse_rendering
        self.refiner_args.use_latents_model = use_latents_model
        self.refiner_args.use_perceptual_model = use_perceptual_model
        torch.set_grad_enabled(prev_grad_enabled)

        # Destroy the predictor to free up space
        logger.info('Destroying predictor to free up space.')
        self.manager.predictor = None
        torch.cuda.empty_cache()
        gc.collect()

        return self._init_initial_predictions()

    def _init_scenes(self):
        """
        Instantiate the scenes.
        """
        scenes_dir = self.path / 'scenes'
        if scenes_dir.exists():
            try:
                self.scenes = {'train': [], 'eval': []}
                for i, (idx, _) in enumerate(self.image_paths):
                    for train_or_eval in ['train', 'eval']:
                        scene_path = scenes_dir / train_or_eval / f'{idx:04d}.yml'
                        assert scene_path.exists()
                        with open(scene_path) as f:
                            scene_dict = yaml.load(f, Loader=yaml.FullLoader)
                        self.scenes[train_or_eval].append({
                            'path': scene_path,
                            'scene_dict': scene_dict,
                        })
                logger.info(f'Loaded scenes from {scenes_dir}.')
                return
            except Exception:
                pass

        # Initialise the scene parameters with the initial predictions
        logger.info(f'Initialising scenes at {scenes_dir}.')
        for train_or_eval in ['train', 'eval']:
            (scenes_dir / train_or_eval).mkdir(parents=True, exist_ok=True)
            for scene_initial_path in self.scenes_initial:
                shutil.copy(scene_initial_path, scenes_dir / train_or_eval)

        return self._init_scenes()

    def _init_parameters(self):
        """
        Instantiate the parameters.
        """
        parameters_path = self.path / 'parameters.json'
        if parameters_path.exists():
            with open(parameters_path, 'r') as f:
                self.parameters = json_to_torch(json.load(f))
            self.n_parameters = sum([v.shape[1] if v.ndim > 1 else 1 for v in self.parameters['train'].values()])
            logger.info(f'Loaded parameters from {parameters_path}.')
            return

        # Initialise the parameters from the scenes
        logger.info(f'Initialising parameters at {parameters_path}.')
        parameters = {'train': {}, 'eval': {}}
        for train_or_eval in ['train', 'eval']:
            for scene in self.scenes[train_or_eval]:
                scene_dict = scene['scene_dict']
                for k in PARAMETER_KEYS:
                    if k not in parameters[train_or_eval]:
                        parameters[train_or_eval][k] = []
                    if k in scene_dict:
                        v = scene_dict[k]
                    elif k in scene_dict['crystal']:
                        v = scene_dict['crystal'][k]
                    else:
                        raise RuntimeError(f'Parameter {k} not found in scene or crystal.')
                    parameters[train_or_eval][k].append(v)
        with open(parameters_path, 'w') as f:
            json.dump(parameters, f, cls=FlexibleJSONEncoder)

        return self._init_parameters()

    def _init_losses(self):
        """
        Instantiate the losses.
        """
        losses_path = self.path / 'losses.json'
        if losses_path.exists():
            with open(losses_path, 'r') as f:
                self.losses = json_to_torch(json.load(f))
            logger.info(f'Loaded losses from {losses_path}.')
            return

        logger.info(f'Initialising losses at {losses_path}.')
        losses = {
            'total/train': np.zeros(len(self), dtype=np.float32),
            'total/eval': np.zeros(len(self), dtype=np.float32),
            'training': []
        }
        with open(losses_path, 'w') as f:
            json.dump(losses, f, cls=FlexibleJSONEncoder)
        return self._init_losses()

    def _init_frame_counts(self):
        """
        Instantiate the frame counts.
        """
        frame_counts_path = self.path / 'frame_counts.json'
        if frame_counts_path.exists():
            with open(frame_counts_path, 'r') as f:
                self.frame_counts = json_to_torch(json.load(f))
            logger.info(f'Loaded frame counts from {frame_counts_path}.')
            return

        # Initialise the frame counts to zero
        logger.info(f'Initialising frame counts at {frame_counts_path}.')
        frame_counts = np.zeros(len(self), dtype=np.int32)
        with open(frame_counts_path, 'w') as f:
            json.dump(frame_counts, f, cls=FlexibleJSONEncoder)
        return self._init_frame_counts()

    def _init_X_preds(self):
        """
        Instantiate the predicted (rendered) images.
        """
        X_preds_dir = self.path / 'X_preds'
        if X_preds_dir.exists():
            try:
                self.X_preds = {'train': [], 'eval': []}
                for (idx, _) in self.image_paths:
                    for train_or_eval in ['train', 'eval']:
                        X_pred_path = X_preds_dir / train_or_eval / f'{idx:04d}.png'
                        assert X_pred_path.exists()
                        self.X_preds[train_or_eval].append(X_pred_path)
                logger.info(f'Loaded predicted images from {X_preds_dir}.')
                return
            except Exception:
                pass

        # Initialise the predicted images with the initial predictions
        logger.info(f'Initialising predicted images at {X_preds_dir}.')
        for train_or_eval in ['train', 'eval']:
            (X_preds_dir / train_or_eval).mkdir(parents=True, exist_ok=True)
            for X_pred_path in self.X_preds_initial:
                shutil.copy(X_pred_path, X_preds_dir / train_or_eval)

        return self._init_X_preds()

    def _init_X_targets_annotated(self):
        """
        Instantiate the target images with wireframe overlaid.
        """
        X_annotated_dir = self.path / 'X_targets_annotated'
        if X_annotated_dir.exists():
            try:
                self.X_targets_annotated = {'train': [], 'eval': []}
                for (idx, _) in self.image_paths:
                    for train_or_eval in ['train', 'eval']:
                        X_annotated_path = X_annotated_dir / train_or_eval / f'{idx:04d}.png'
                        assert X_annotated_path.exists()
                        self.X_targets_annotated[train_or_eval].append(X_annotated_path)
                logger.info(f'Loaded annotated images from {X_annotated_dir}.')
                return
            except Exception:
                pass

        # Initialise the annotated images with the initial predictions
        logger.info(f'Initialising annotated images at {X_annotated_dir}.')
        for train_or_eval in ['train', 'eval']:
            (X_annotated_dir / train_or_eval).mkdir(parents=True, exist_ok=True)
            for X_annotated_path in self.X_targets_annotated_initial:
                shutil.copy(X_annotated_path, X_annotated_dir / train_or_eval)

        return self._init_X_targets_annotated()

    def _init_tb_logger(self):
        """
        Instantiate the tensorboard logger.
        """
        tb_dir = self.path / 'tensorboard'
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.tb_logger = SummaryWriter(tb_dir, flush_secs=5)

    def _init_sequence_encoder(self):
        """
        Instantiate the sequence encoder, optimiser and learning rate scheduler.
        """
        sa = self.sf_args
        self.sequence_encoder = SequenceEncoder(
            param_dim=self.n_parameters,
            hidden_dim=sa.hidden_dim,
            n_layers=sa.n_layers,
            n_heads=sa.n_heads,
            max_freq=sa.max_freq,
            dropout=sa.dropout,
            activation=sa.activation,
        )
        logger.info(f'Instantiated sequence encoder with {self.sequence_encoder.get_n_params() / 1e6:.4f}M parameters.')
        logger.debug(f'----------- Sequence Encoder Network --------------\n\n{self.sequence_encoder}\n\n')
        self.sequence_encoder = self.sequence_encoder.to(self.device)

        # Instantiate the optimiser
        self.optimiser = create_optimizer_v2(
            opt=sa.opt_algorithm,
            lr=sa.lr_init,
            weight_decay=sa.weight_decay,
            model_or_params=self.sequence_encoder,
        )

        # Instantiate the learning rate scheduler
        self.lr_scheduler, _ = create_scheduler_v2(
            optimizer=self.optimiser,
            sched='plateau',
            num_epochs=sa.max_steps,
            patience_epochs=sa.lr_patience_steps,
            decay_rate=sa.lr_decay_rate,
            min_lr=sa.lr_min,
            plateau_mode='min'
        )

        # Load the checkpoint if it exists
        loaded = False
        state_path = self.path / 'sequence_encoder.pt'
        if state_path.exists():
            logger.info(f'Loading sequence encoder state from {state_path}.')
            state = torch.load(state_path, weights_only=True)
            self.sequence_encoder.load_state_dict(state['model'])
            self.optimiser.load_state_dict(state['optimiser'])
            loaded = True
        self.sequence_encoder.eval()
        if loaded:
            return

        # Pretrain the sequence encoder if required, to predict the initial parameters
        if sa.n_pretraining_steps == 0:
            return
        logger.info('Pretraining the sequence encoder.')
        self.sequence_encoder.train()
        N = len(self)

        # Instantiate a new pretraining optimiser
        pretrain_optimiser = create_optimizer_v2(
            opt=sa.opt_algorithm,
            lr=sa.lr_pretrain,
            weight_decay=sa.weight_decay,
            model_or_params=self.sequence_encoder,
        )

        # Pretrain the sequence encoder
        running_loss = 0.
        for i in range(sa.n_pretraining_steps):
            # Sample random frame indices and normalise
            idx_batch = self.image_idxs[torch.randperm(N)[:sa.pretrain_batch_size]]
            ts = ((idx_batch - self.image_idxs[0]) / (N - 1)).to(self.device)

            # Generate target parameter batch
            parameters_target = torch.stack([
                self.get_parameter_vector(idx) for idx in idx_batch
            ]).to(self.device)

            # Train to predict the initial parameters predictions
            parameters_pred = self.sequence_encoder(ts)
            loss = torch.mean((parameters_pred - parameters_target)**2)
            loss.backward()
            pretrain_optimiser.step()
            pretrain_optimiser.zero_grad()

            # Log statistics
            running_loss += loss.item()
            if (i + 1) % self.runtime_args.log_freq_pretrain == 0:
                loss_avg = running_loss / self.runtime_args.log_freq_pretrain
                logger.info(f'[{i + 1}/{sa.n_pretraining_steps}]\tLoss: {loss_avg:.4E}')
                running_loss = 0.
            self.tb_logger.add_scalar('pretraining/loss', loss.item(), i)

        logger.info('Pretraining complete.')
        self.sequence_encoder.eval()
        self._save_encoder_state()
        self.update_parameters_from_encoder()
        self.update_X_preds_from_parameters()

    def fit(self):
        """
        Train the sequence encoder to fit the sequence.
        """
        logger.info('Training the sequence encoder.')
        N = len(self)
        bs = self.sf_args.batch_size

        # Set up the adaptive sampler
        sampler = AdaptiveSampler(
            sequence_length=N,
            ema_decay=self.sf_args.ema_decay,
            ema_init=self.sf_args.ema_val_init,
        )
        for i in range(N):
            if self.frame_counts[i] > 0:
                sampler.emas[i].val = self.losses['total/train'][i].item()

        # Load the first scene and adapt the crystal to use buffers instead of parameters
        scene = Scene.from_yml(self.scenes['train'][0]['path'])
        crystal = scene.crystal
        for k in list(crystal._parameters.keys()):
            val = crystal.get_parameter(k).data
            del crystal._parameters[k]
            crystal.register_buffer(k, val)

        # Set up the refiner
        self.refiner.scene = scene
        self.refiner.scene_params = mi.traverse(scene.mi_scene)
        self.refiner.crystal = crystal
        self.refiner._init_projector()
        use_inverse_rendering = self.refiner_args.use_inverse_rendering  # Save the inverse rendering setting

        # Determine how many steps have been completed from the frame counts
        start_step = int(self.frame_counts.sum() / bs)
        assert start_step == len(self.losses['training']), 'Frame counts and training losses do not match.'
        if start_step > 0:
            logger.info(f'Resuming training from step {start_step}.')
        max_steps = self.sf_args.max_steps

        # Train the sequence encoder
        running_loss = 0.
        running_tps = 0
        for step in range(start_step, max_steps):
            start_time = time.time()
            self.sequence_encoder.train()
            self.refiner.step = step

            # Sample frame indices and normalise
            idxs = sampler.sample_frames(bs)
            ts = (idxs / (N - 1)).to(self.device)

            # Generate the target images batches
            X_targets = {'og': [], 'dn': [], 'og_wis': [], 'dn_wis': []}
            for k, batch in X_targets.items():
                for idx in idxs:
                    if k[:2] == 'dn':
                        X_path = self.X_targets_denoised[idx]
                    else:
                        X_path = self.X_targets[idx]
                    X = to_tensor(Image.open(X_path))
                    if k[-3:] == 'wis':
                        X = self.refiner._resize(X)
                    batch.append(X)
                X_targets[k] = torch.stack(batch).permute(0, 2, 3, 1).to(self.device)

            # Generate the parameters from the encoder
            parameters_pred = self._parameter_vectors_to_dict(
                self.sequence_encoder(ts)
            )

            # Conditionally enable the inverse rendering
            self.refiner_args.use_inverse_rendering = use_inverse_rendering and step >= self.refiner_args.ir_wait_n_steps

            # Loop over the batch and use the refiner to calculate the loss for each frame
            losses_batch = []
            for i in range(bs):
                idx = idxs[i]

                # Update targets
                self.refiner.X_target = X_targets['og'][i]
                self.refiner.X_target_wis = X_targets['og_wis'][i]
                self.refiner.X_target_denoised = X_targets['dn'][i]
                self.refiner.X_target_denoised_wis = X_targets['dn_wis'][i]
                if self.refiner_args.use_keypoints:
                    self.refiner.keypoint_targets = self.keypoints[idx]['keypoints']

                # Update scene and crystal parameters
                for k, v in parameters_pred.items():
                    if k == 'light_radiance':
                        scene.light_radiance = v[i].to(self.device)
                    else:
                        setattr(crystal, k, v[i].to('cpu'))
                    self.parameters['train'][k][idx] = v[i].clone().detach().cpu()

                # Calculate losses
                loss_j, stats_j = self.refiner._process_step(add_noise=False)
                (loss_j / bs).backward(retain_graph=True)
                losses_batch.append(loss_j)

                # Update sequence state
                self.losses['total/train'][idx] = loss_j.item()
                self.frame_counts[idx] += 1
                if self.refiner_args.use_inverse_rendering:
                    X_pred = (to_numpy(self.refiner.X_pred) * 255).astype(np.uint8)
                    Image.fromarray(X_pred).save(self.X_preds['train'][idx])
                annotate_image(self.X_targets[idx], scene).save(self.X_targets_annotated['train'][idx])

            # Calculate the mean loss and gradients
            losses_batch = torch.tensor(losses_batch)
            loss = losses_batch.mean()

            # Clip gradients
            if self.sf_args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.sequence_encoder.parameters(), max_norm=self.sf_args.clip_grad_norm)

            # Check for bad gradients
            for group in self.optimiser.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        if is_bad(param.grad):
                            logger.warning('Bad gradients detected!')
                            param.grad.zero_()

            # Optimisation step
            self.optimiser.step()
            self.optimiser.zero_grad()

            # Update the adaptive sampler and log the total sequence loss estimate
            sampler.update_errors(idxs, losses_batch)
            seq_loss = sampler.errors.sum().item()
            self.losses['training'] = torch.concatenate([self.losses['training'], torch.tensor(seq_loss)[None, ...]])
            self.tb_logger.add_scalar('losses/seq_train', self.losses['total/train'].sum().item(), step)
            self.tb_logger.add_scalar('losses/seq_train_ema', seq_loss, step)
            self.tb_logger.add_scalar('losses/step', loss.item(), step)

            # Log learning rate and update
            self.tb_logger.add_scalar('lr', self.optimiser.param_groups[0]['lr'], step)
            self.lr_scheduler.step(step + 1, seq_loss)

            # Log network norm
            weights_cumulative_norm = calculate_model_norm(self.sequence_encoder, device=self.device)
            assert not is_bad(weights_cumulative_norm), 'Bad parameters!'
            self.tb_logger.add_scalar('w_norm', weights_cumulative_norm.item(), step)

            # Log statistics every X steps
            time_per_step = time.time() - start_time
            running_tps += time_per_step
            running_loss += loss.item()
            if (step + 1) % self.runtime_args.log_freq_train == 0:
                loss_avg = running_loss / self.runtime_args.log_freq_train
                average_tps = running_tps / self.runtime_args.log_freq_train
                seconds_left = float((max_steps - step) * average_tps)
                logger.info(f'[{step + 1}/{max_steps}]\tLoss: {loss_avg:.4E}'
                            + '\tTime per step: {}\tEst. complete in: {}'.format(
                    str(timedelta(seconds=average_tps)),
                    str(timedelta(seconds=seconds_left))))
                running_loss = 0.
                running_tps = 0

            # Checkpoint every X steps
            if (step + 1) % self.runtime_args.checkpoint_freq == 0:
                self.save()
                self._save_encoder_state()

            # Plot every X steps
            if (step + 1) % self.runtime_args.plot_freq == 0:
                logger.info('Making training plots.')
                image_idx = self.image_idxs[idx].item()
                fig = self.refiner._plot_comparison()
                self._save_plot(fig, 'train', 'comparison', image_idx)
                plt.close(fig)
                if self.refiner.patch_centres is not None:
                    fig = self.refiner._plot_patches()
                    self._save_plot(fig, 'train', 'patches', image_idx)
                    plt.close(fig)
                self._plot_parameters('train')

            # Evaluate the sequence every X steps
            if (step + 1) % self.runtime_args.eval_freq == 0:
                logger.info('Evaluating the sequence.')
                self.update_parameters_from_encoder()
                self._plot_parameters('eval')
                generate_annotations = (step + 1) % self.runtime_args.eval_annotate_freq == 0
                generate_renders = (step + 1) % self.runtime_args.eval_render_freq == 0
                if generate_annotations or generate_renders:
                    self.update_X_preds_from_parameters(
                        train_or_eval='eval',
                        generate_renders=generate_renders,
                        generate_annotations=generate_annotations
                    )
                if (step + 1) % self.runtime_args.eval_video_freq == 0:
                    logger.info('Generating evaluation video.')
                    self._generate_video(train_or_eval='eval')

        logger.info('Training complete.')

    @torch.no_grad()
    def update_parameters_from_encoder(self, train_or_eval: str = 'eval'):
        """
        Generate the parameters for the sequence using the encoder.
        """
        logger.info('Generating parameters from the sequence encoder.')
        if train_or_eval == 'train':
            self.sequence_encoder.train()
        else:
            self.sequence_encoder.eval()
        ts = ((self.image_idxs - self.image_idxs[0]) / (len(self) - 1)).to(self.device)
        bs = min(len(self), self.sf_args.eval_batch_size)
        n_batches = math.ceil(len(self) / bs)
        for i in range(n_batches):
            if (i + 1) % 10 == 0:
                logger.info(f'Generating {train_or_eval} parameters for frame batch {i + 1}/{n_batches}.')
            ts_i = ts[i * bs:(i + 1) * bs]
            p_vec_i = self.sequence_encoder(ts_i)
            parameters_i = self._parameter_vectors_to_dict(p_vec_i)
            for k, v in parameters_i.items():
                self.parameters[train_or_eval][k][i * bs:(i + 1) * bs] = v.detach().cpu().clone().squeeze()
        self.sequence_encoder.eval()  # Ensure the encoder is returned to eval mode
        self.save()

    @torch.no_grad()
    def update_X_preds_from_parameters(
            self,
            train_or_eval: str = 'eval',
            generate_renders: bool = True,
            generate_annotations: bool = True
    ):
        """
        Render predicted images from the parameters.
        """
        logger.info('Generating predicted images from the parameters.')
        scene = Scene.from_yml(self.scenes[train_or_eval][0]['path'])
        crystal = scene.crystal
        crystal.to('cpu')
        for i in range(len(self)):
            if (i + 1) % 10 == 0:
                logger.info(f'Generating image {i + 1}/{len(self)}.')
            for k in PARAMETER_KEYS:
                v = init_tensor(self.parameters[train_or_eval][k][i])
                if k == 'light_radiance':
                    scene.light_radiance.data = v.to(scene.device)
                else:
                    getattr(crystal, k).data = v
            if generate_renders:
                scene.build_mi_scene()
                X_pred = scene.render()
                Image.fromarray(X_pred).save(self.X_preds[train_or_eval][i])
            if generate_annotations:
                annotate_image(self.X_targets[i], scene).save(self.X_targets_annotated[train_or_eval][i])

    def _plot_parameters(self, train_or_eval: str):
        """
        Plot the parameters.
        """
        plot_args = dict(
            parameters=self.parameters[train_or_eval],
            # measurements=measurements,
            image_paths=self.image_paths,
        )
        plot_dir = self.path / 'plots' / train_or_eval
        for plot_type in ['distances', 'areas']:
            figs = {}
            if plot_type == 'distances':
                figs['grouped'], figs['mean'] = plot_distances(manager=self.refiner.manager, **plot_args)
            else:
                figs['grouped'], figs['mean'] = plot_areas(manager=self.refiner.manager, **plot_args)
            for group_type in ['grouped', 'mean']:
                save_dir = plot_dir / f'{plot_type}_{group_type}'
                save_dir.mkdir(parents=True, exist_ok=True)
                figs[group_type].savefig(save_dir / f'{self.refiner.step + 1:06d}.png')
                plt.close(figs[group_type])

    def _generate_video(self, train_or_eval: str = 'eval'):
        """
        Generate a video from the images.
        """
        imgs_dir = self.X_targets_annotated[train_or_eval][0].parent
        out_dir = self.path / 'videos' / train_or_eval / 'annotations'
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f'{self.refiner.step:05d}.mp4'
        logger.info(f'Making growth video from {imgs_dir} to {save_path}.')
        escaped_images_dir = str(imgs_dir.absolute()).replace('[', '\\[').replace(']', '\\]')
        cmd = f'ffmpeg -y -framerate 25 -pattern_type glob -i "{escaped_images_dir}/*.png" -c:v libx264 -pix_fmt yuv420p "{save_path}"'
        logger.info(f'Running command: {cmd}')
        os.system(cmd)

    def get_parameter_vector(self, idx: int, train_or_eval: str = 'eval') -> Tensor:
        """
        Return a vector of the parameters for the frame at the given index.
        """
        i = self.image_idxs.tolist().index(idx)
        scene_dict = self.scenes[train_or_eval][i]['scene_dict']

        params = torch.concatenate([
            torch.tensor(v) for v in [
                [scene_dict['crystal']['scale'], ],
                scene_dict['crystal']['distances'],
                scene_dict['crystal']['origin'],
                scene_dict['crystal']['rotation'],
                [scene_dict['crystal']['material_roughness'], ],
                [scene_dict['crystal']['material_ior'], ],
                scene_dict['light_radiance'],
            ]
        ])

        return params

    def _parameter_vectors_to_dict(self, parameter_vectors: Tensor) -> Dict[str, Tensor]:
        """
        Convert a parameter vector to a dictionary of parameters.
        """
        parameters = {}
        idx = 0
        for k, v in self.parameters['eval'].items():
            n = v.shape[1] if v.ndim > 1 else 1
            val = parameter_vectors[:, idx:idx + n].squeeze(1)
            if k == 'scale':
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

    def save(self):
        """
        Save the sequence data.
        """
        logger.info('Saving parameters, losses, frame counts and scenes.')
        for name, data in zip(['losses', 'parameters', 'frame_counts'],
                              [self.losses, self.parameters, self.frame_counts]):
            with open(self.path / f'{name}.json', 'w') as f:
                json.dump(data, f, cls=FlexibleJSONEncoder)

        # Update the scene files with the parameters
        for train_or_eval in ['train', 'eval']:
            for i, scene in enumerate(self.scenes[train_or_eval]):
                scene_dict = scene['scene_dict']
                for k in PARAMETER_KEYS:
                    v = self.parameters[train_or_eval][k][i].tolist()
                    if k in scene_dict:
                        scene_dict[k] = v
                    elif k in scene_dict['crystal']:
                        scene_dict['crystal'][k] = v
                    else:
                        raise RuntimeError(f'Parameter {k} not found in scene or crystal.')
                with open(scene['path'], 'w') as f:
                    yaml.dump(scene_dict, f)

    def _save_encoder_state(self):
        """
        Save the sequence encoder state.
        """
        state_path = self.path / 'sequence_encoder.pt'
        logger.info(f'Saving sequence encoder state to {state_path}.')
        torch.save({
            'model': self.sequence_encoder.state_dict(),
            'optimiser': self.optimiser.state_dict(),
        }, state_path)

    def _save_plot(self, fig: Figure, train_or_eval: str, plot_type: str, idx: int):
        """
        Save the current plot.
        """
        fig.suptitle(f'Frame #{idx} Step {self.refiner.step + 1} Loss: {self.refiner.loss:.4E}')
        save_dir = self.path / 'plots' / train_or_eval / plot_type
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f'{idx:04d}_{self.refiner.step + 1:06d}.png'
        plt.savefig(path, bbox_inches='tight')

    def __len__(self) -> int:
        return len(self.image_paths)
