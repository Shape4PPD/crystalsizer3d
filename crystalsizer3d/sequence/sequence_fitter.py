import gc
import json
import math
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import center_crop, to_tensor

from crystalsizer3d import DATA_PATH, logger
from crystalsizer3d.args.refiner_args import DENOISER_ARG_NAMES, EDGES_ARG_NAMES, KEYPOINTS_ARG_NAMES, \
    PREDICTOR_ARG_NAMES, PREDICTOR_ARG_NAMES_BS1, RefinerArgs
from crystalsizer3d.args.sequence_fitter_args import SequenceFitterArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.refiner.denoising import denoise_image
from crystalsizer3d.refiner.edge_detection import find_edges
from crystalsizer3d.refiner.keypoint_detection import find_keypoints
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.sequence.adaptive_sampler import AdaptiveSampler
from crystalsizer3d.sequence.data_loader import get_data_loader
from crystalsizer3d.sequence.refiner_pool import RefinerPool
from crystalsizer3d.sequence.sequence_encoder import SequenceEncoder
from crystalsizer3d.sequence.sequence_encoder_ffn import SequenceEncoderFFN
from crystalsizer3d.sequence.sequence_plotter import SequencePlotter
from crystalsizer3d.sequence.utils import get_image_paths, load_manual_measurements
from crystalsizer3d.util.utils import FlexibleJSONEncoder, calculate_model_norm, get_crystal_face_groups, hash_data, \
    init_tensor, is_bad, json_to_torch, set_seed, to_dict, to_numpy

# Suppress mitsuba warning messages
mi.set_log_level(mi.LogLevel.Error)

PARAMETER_KEYS = ['scale', 'distances', 'origin', 'rotation', 'material_roughness', 'material_ior', 'light_radiance']
SEQUENCE_DATA_PATH = DATA_PATH / 'sequences'

ARG_NAMES = {
    'denoiser': DENOISER_ARG_NAMES,
    'keypoints': KEYPOINTS_ARG_NAMES,
    'edges': EDGES_ARG_NAMES,
    'predictor': PREDICTOR_ARG_NAMES,
}

resize_args = dict(mode='bilinear', align_corners=False)


class SequenceFitter:
    dataloader: DataLoader

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
        self._init_output_dirs()

        # Initialise the refiner and the asynchronous plotter
        self._init_refiner()
        self.refiner_pool = None  # Instantiated on demand
        self._load_manual_measurements()
        self._init_plotter()

        # Initialise the target data
        self._init_X_targets()
        self._init_X_targets_denoised()
        self._init_X_wis()
        self._init_keypoints()
        self._init_edges()
        self._init_edges_annotated()

        # Initialise the outputs
        self._init_initial_predictions()
        self._init_initial_prediction_losses()
        self._init_fixed_parameters()
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
            if model_name == 'predictor' and self.refiner_args.initial_pred_batch_size == 1:
                model_args = {k: getattr(self.refiner_args, k) for k in PREDICTOR_ARG_NAMES_BS1}
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
        if self.path.exists() and not self.runtime_args.resume:
            shutil.rmtree(self.path)

        # If the sequence directory exists, but we're asked to resume from a different checkpoint, abort
        elif self.path.exists() and self.runtime_args.resume_from is not None:
            raise RuntimeError('Resuming from a different run will remove the current run\'s data. '
                               f'Please remove the current run\'s data manually from {self.path} and try again.')

        # Create the sequence directory
        self.path.mkdir(parents=True, exist_ok=True)

        # Initialise from a previous checkpoint
        if self.runtime_args.resume_from is not None:
            assert self.runtime_args.resume_from.exists(), f'Resume from path does not exist: {self.runtime_args.resume_from}'
            logger.info(f'Copying files from previous run at {self.runtime_args.resume_from}.')
            shutil.copytree(self.runtime_args.resume_from, self.path, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('parent_*'))

            # Copy all yml, json and pt files and any parent directory from the previous run to preserve a record
            parent_dir = self.path / f'parent_{self.runtime_args.resume_from.name}'
            parent_dir.mkdir(exist_ok=True)
            for file_path in self.runtime_args.resume_from.iterdir():
                if file_path.suffix in ['.yml', '.json', '.pt']:
                    shutil.copy2(file_path, parent_dir)
                elif file_path.name[:6] == 'parent':
                    shutil.copytree(file_path, parent_dir / file_path.name, dirs_exist_ok=True)

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

    def _load_manual_measurements(self):
        """
        Load manual measurements from the measurements directory.
        """
        measurements_dir = self.runtime_args.measurements_dir
        if measurements_dir is None:
            return
        self.measurements = load_manual_measurements(
            measurements_dir=measurements_dir,
            manager=self.manager
        )

    def _init_plotter(self):
        """
        Instantiate the asynchronous plotter.
        """
        self.plotter = SequencePlotter(
            n_workers=self.runtime_args.n_plotting_workers,
            queue_size=self.runtime_args.plot_queue_size,
            measurements=self.measurements
        )
        self.plotter.wait_for_workers()

    def _init_X_targets(self):
        """
        Instantiate the cropped target images.
        """
        X_targets_dir = self.base_path / 'X_targets' / 'fullsize'
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
            self.plotter.save_image(X, X_targets_dir / f'{idx:04d}.png')

        # Wait for the plotter workers
        self.plotter.wait_for_workers()

        return self._init_X_targets()

    @torch.no_grad()
    def _init_X_targets_denoised(self):
        """
        Instantiate the denoised data.
        """
        X_targets_denoised_dir = self.cache_dirs['denoiser'] / 'fullsize'
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
            self.plotter.save_image(X_target_denoised, X_targets_denoised_dir / f'{idx:04d}.png')

        # Destroy the denoiser to free up space
        logger.info('Destroying denoiser to free up space.')
        self.manager.denoiser = None
        torch.cuda.empty_cache()
        gc.collect()

        # Wait for the plotter workers
        self.plotter.wait_for_workers()

        return self._init_X_targets_denoised()

    def _init_X_wis(self):
        """
        Instantiate the dataset of resized images.
        """
        wis = self.refiner_args.rendering_size
        X_dirs = {
            'og': self.base_path / 'X_targets' / str(wis),
            'dn': self.cache_dirs['denoiser'] / str(wis)
        }
        try:
            self.X_wis = {'og': [], 'dn': []}
            for X_type in ['og', 'dn']:
                for (idx, _) in self.image_paths:
                    X_wis_path = X_dirs[X_type] / f'{idx:04d}.png'
                    self.X_wis[X_type].append(to_tensor(Image.open(X_wis_path)))
            logger.info(f'Loaded resized images from {X_dirs["og"]} and {X_dirs["dn"]}.')
            return
        except Exception:
            pass

        # Resize the images to the rendering sizes
        logger.info('Initialising the resized images.')
        for X_dir in X_dirs.values():
            X_dir.mkdir(parents=True, exist_ok=True)
        for i, (idx, _) in enumerate(self.image_paths):
            if (i + 1) % 50 == 0:
                logger.info(f'Resizing target images {i + 1}/{len(self)}.')
            X = to_tensor(Image.open(self.X_targets[i]))
            X_dn = to_tensor(Image.open(self.X_targets_denoised[i]))
            X_wis = self.refiner._resize(X).permute(1, 2, 0)
            X_dn_wis = self.refiner._resize(X_dn).permute(1, 2, 0)
            self.plotter.save_image(X_wis, X_dirs['og'] / f'{idx:04d}.png')
            self.plotter.save_image(X_dn_wis, X_dirs['dn'] / f'{idx:04d}.png')

        # Wait for the plotter workers
        self.plotter.wait_for_workers()

        return self._init_X_wis()

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

    @torch.no_grad()
    def _init_edges(self):
        """
        Instantiate the edges.
        """
        ra = self.refiner_args
        if not ra.use_edge_matching:
            self.edges = None
            return

        def load_edges(edges_dir):
            try:
                edges = []
                for (idx, _) in self.image_paths:
                    edges_path = edges_dir / f'{idx:04d}.png'
                    assert edges_path.exists()
                    edges.append(edges_path)
                return edges
            except Exception:
                pass

        # First, load the fullsize edge images
        edges_dir_fullsize = self.cache_dirs['edges'] / 'fullsize'
        edges_fullsize = load_edges(edges_dir_fullsize)

        # If these were loaded, try to load the target size images
        if edges_fullsize is not None:
            self.edges_fullsize = edges_fullsize
            size = str(ra.edge_matching_image_size)
            edges_dir = self.cache_dirs['edges'] / size
            edges = load_edges(edges_dir)

            # If these were also loaded, we're done
            if edges is not None:
                logger.info(f'Loaded edge images from {edges_dir}.')
                self.edges = edges
                return

            # Otherwise, resize the fullsize images
            edges_dir.mkdir(parents=True, exist_ok=True)
            for i, (idx, _) in enumerate(self.image_paths):
                if (i + 1) % 50 == 0:
                    logger.info(f'Resizing edge images {i + 1}/{len(self)}.')
                X = to_tensor(Image.open(self.edges_fullsize[i]))
                X_rs = self.refiner._resize(X, size=int(size)).squeeze()
                self.plotter.save_image(X_rs, edges_dir / f'{idx:04d}.png')

            # Reload the resized images
            return self._init_edges()

        # Otherwise, generate the fullsize edge images
        edges_dir_fullsize.mkdir(parents=True, exist_ok=True)
        self.manager.load_network(ra.edges_model_path, 'keypointdetector')
        assert self.manager.keypoint_detector is not None, 'No edge detector model loaded, so can\'t detect edges.'

        # Find the edges for each image
        logger.info('Finding edges across image sequence.')
        for i, (X_path, X_denoised_path) in enumerate(zip(self.X_targets, self.X_targets_denoised)):
            if (i + 1) % 10 == 0:
                logger.info(f'Finding edges for image {i + 1}/{len(self)}.')
            idx = self.image_paths[i][0]
            if (edges_dir_fullsize / f'{idx:04d}.png').exists():
                continue
            X = to_tensor(Image.open(X_path))
            X_dn = to_tensor(Image.open(X_denoised_path))
            X_wf = find_edges(
                X_target=X,
                X_target_denoised=X_dn,
                manager=self.manager,
                oversize_input=ra.edges_oversize_input,
                max_img_size=ra.edges_max_img_size,
                batch_size=ra.edges_batch_size,
                threshold=ra.edges_threshold,
                exclude_border=ra.edges_exclude_border,
                n_patches=ra.edges_n_patches,
                patch_size=ra.edges_patch_size,
                patch_search_res=ra.edges_patch_search_res,
                attenuation_sigma=ra.edges_attenuation_sigma,
                max_attenuation_factor=ra.edges_max_attenuation_factor,
                return_everything=False,
                quiet=True
            )
            self.plotter.save_image(X_wf, edges_dir_fullsize / f'{idx:04d}.png')

        # Destroy the edge detector to free up space
        logger.info('Destroying edge detector to free up space.')
        self.manager.keypoint_detector = None
        torch.cuda.empty_cache()
        gc.collect()

        return self._init_edges()

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
        ds = self.refiner.manager.ds
        ra = self.refiner_args

        # Make directories
        scenes_initial_dir.mkdir(parents=True, exist_ok=True)
        X_preds_initial_dir.mkdir(parents=True, exist_ok=True)
        X_annotated_initial_dir.mkdir(parents=True, exist_ok=True)

        # Set up the resizing
        resize_input_old = self.manager.predictor.resize_input
        self.manager.predictor.resize_input = False
        oversize_input = ra.initial_pred_oversize_input
        img_size = ra.initial_pred_max_img_size if oversize_input else self.manager.image_shape[-1]
        render_size = ds.dataset_args.image_size  # Use the rendering size the predictor was trained on

        # If the batch size is 1 then we can simplify things significantly
        if ra.initial_pred_batch_size == 1:
            scene_args = ds.dataset_args.to_dict()
            scene_args['spp'] = ra.spp
            scene_args['res'] = ra.rendering_size

            cs = CSDProxy().load(ds.dataset_args.crystal_id)
            crystal_args = dict(
                lattice_unit_cell=cs.lattice_unit_cell,
                lattice_angles=cs.lattice_angles,
                miller_indices=ds.miller_indices,
                point_group_symbol=cs.point_group_symbol,
                rotation_mode=ds.dataset_args.rotation_mode,
            )

            for i, X_path in enumerate(self.X_targets):
                idx = self.image_paths[i][0]
                scene_path = scenes_initial_dir / f'{idx:04d}.yml'
                X_pred_path = X_preds_initial_dir / f'{idx:04d}.png'
                X_annotated_path = X_annotated_initial_dir / f'{idx:04d}.png'
                if scene_path.exists() and X_pred_path.exists() and X_annotated_path.exists():
                    continue
                if (i + 1) % 10 == 0:
                    logger.info(f'Generating initial predictions for frame {i + 1}/{len(self)}.')

                # Resize the input if needed
                X = to_tensor(Image.open(X_path)).to(self.device)[None, ...]
                if oversize_input and X.shape[-1] > img_size \
                        or not oversize_input and X.shape[-1] != img_size:
                    X = F.interpolate(X, size=img_size, **resize_args)

                # Predict the parameters
                Y_pred = self.manager.predict(X)
                params = ds.denormalise_rendering_params(Y_pred, idx=0)

                # Build the crystal
                crystal = Crystal(
                    **crystal_args,
                    distances=params['crystal']['distances'],
                    scale=params['crystal']['scale'],
                    origin=params['crystal']['origin'],
                    rotation=params['crystal']['rotation'],
                    material_roughness=params['crystal']['material_roughness'],
                    material_ior=params['crystal']['material_ior'],
                )

                # Build the scene
                scene = Scene(
                    crystal=crystal,
                    light_radiance=params['light_radiance'],
                    **scene_args,
                )

                # Save the parameters and images
                scene.to_yml(scene_path, overwrite=True)
                self.plotter.annotate_image(X_path, X_annotated_path, scene)
                self.plotter.render_scene(scene, X_pred_path)

        else:
            # Sinkhorn loss re-enables grad so disable and re-enable it manually
            # https://github.com/jeanfeydy/geomloss/issues/57
            prev_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

            # Ensure the inverse rendering losses are turned on
            use_inverse_rendering = ra.use_inverse_rendering
            ra.use_inverse_rendering = True
            use_latents_model = ra.use_latents_model
            ra.use_latents_model = ra.w_latent > 0
            use_perceptual_model = ra.use_perceptual_model
            ra.use_perceptual_model = ra.w_perceptual > 0

            # Noise batch vars
            bs = ra.initial_pred_batch_size // 2
            noise_scale = torch.linspace(
                ra.initial_pred_noise_min,
                ra.initial_pred_noise_max,
                bs,
                device=self.device
            )

            def update_scene_parameters(scene_: Scene):
                # Update the scene rendering parameters for optimisation
                scene_.res = ra.rendering_size
                scene_.spp = ra.spp
                scene_.camera_type = 'perspective'  # thinlens doesn't work for inverse rendering
                scene_.integrator_max_depth = ra.integrator_max_depth
                scene_.integrator_rr_depth = ra.integrator_rr_depth
                scene_.build_mi_scene()
                scene_.crystal.to('cpu')
                return scene_

            for i, (X_path, X_dn_path) in enumerate(zip(self.X_targets, self.X_targets_denoised)):
                idx = self.image_paths[i][0]
                scene_path = scenes_initial_dir / f'{idx:04d}.yml'
                X_pred_path = X_preds_initial_dir / f'{idx:04d}.png'
                X_annotated_path = X_annotated_initial_dir / f'{idx:04d}.png'
                if scene_path.exists() and X_pred_path.exists() and X_annotated_path.exists():
                    continue
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

                # Set the target image, keypoints and edge map for loss calculations
                self.refiner.X_target_aug = self.refiner._resize(X_dn, render_size).permute(1, 2, 0)
                if ra.use_keypoints:
                    self.refiner.keypoint_targets = self.keypoints[i]['keypoints']
                if ra.use_edge_matching:
                    self.refiner.edge_map = to_tensor(Image.open(self.edges[i]))

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
                    r_params = ds.denormalise_rendering_params(Y_pred_batch, idx=j)
                    X_pred_ij, scene_ij = self.manager.crystal_renderer.render_from_parameters(
                        r_params, return_scene=True)
                    scene_i_batch.append(scene_ij)

                    # Set the variables temporarily
                    X_pred_ij = X_pred_ij.astype(np.float32) / 255.
                    X_pred_ij = torch.from_numpy(X_pred_ij).to(self.device)
                    self.refiner.X_pred = X_pred_ij
                    self.refiner.scene = scene_ij
                    self.refiner.crystal = scene_ij.crystal
                    self.refiner.crystal.to('cpu')

                    # Project the crystal mesh - reinitialise the projector for the new scene
                    if ra.use_keypoints or ra.use_edge_matching:
                        self.refiner.init_projector()
                        self.refiner.projector.project(generate_image=False)

                    # Calculate losses
                    loss_ij, _ = self.refiner._calculate_losses()
                    losses_i.append(loss_ij.item())

                # Update the scene and render the best initial prediction at the correct resolution
                best_idx = np.argmin(losses_i)
                scene = scene_i_batch[best_idx]
                scene = update_scene_parameters(scene)
                X_pred_initial = scene.render()

                # Save the parameters and images
                scene.to_yml(scene_path, overwrite=True)
                self.plotter.save_image(X_pred_initial, X_pred_path)
                self.plotter.annotate_image(X_path, X_annotated_path, scene)

            # Restore the settings
            del self.refiner.projector
            self.manager.predictor.resize_input = resize_input_old
            ra.use_inverse_rendering = use_inverse_rendering
            ra.use_latents_model = use_latents_model
            ra.use_perceptual_model = use_perceptual_model
            torch.set_grad_enabled(prev_grad_enabled)

        # Destroy the predictor to free up space
        logger.info('Destroying predictor to free up space.')
        self.manager.predictor = None
        torch.cuda.empty_cache()
        gc.collect()

        # Wait for the plotter workers
        self.plotter.wait_for_workers()

        return self._init_initial_predictions()

    def _init_initial_prediction_losses(self):
        """
        Instantiate the initial prediction losses.
        """
        # Use the full predictor args to build the filename in order to calculate the
        # correct losses even if the predictor batch size was 1
        model_args = {k: getattr(self.refiner_args, k) for k in PREDICTOR_ARG_NAMES}
        losses_path = self.cache_dirs['predictor'] / f'losses_{hash_data(model_args)}.json'
        try:
            with open(losses_path) as f:
                self.losses_init = json_to_torch(json.load(f))
            logger.info(f'Loaded initial prediction losses from {losses_path}.')
            return
        except Exception:
            pass
        logger.info('Calculating initial prediction losses.')

        # Set up the data loader (adaptive sampler isn't used, just to get the dataset)
        sampler = AdaptiveSampler(sequence_length=len(self))
        self.dataloader = get_data_loader(
            sequence_fitter=self,
            adaptive_sampler=sampler,
            batch_size=1,
            n_workers=0,
        )

        # Load the initial scenes and parameters into the 'eval' space
        scenes = []
        parameters = {}
        for i, scene_initial_path in enumerate(self.scenes_initial):
            with open(scene_initial_path) as f:
                scene_dict = yaml.load(f, Loader=yaml.FullLoader)
            scenes.append({
                'path': scene_initial_path,
                'scene_dict': scene_dict,
            })
            for k in PARAMETER_KEYS:
                if k not in parameters:
                    parameters[k] = []
                if k in scene_dict:
                    v = scene_dict[k]
                elif k in scene_dict['crystal']:
                    v = scene_dict['crystal'][k]
                else:
                    raise RuntimeError(f'Parameter {k} not found in scene or crystal.')
                parameters[k].append(v)
        self.scenes = {'eval': scenes}
        self.parameters = {'eval': {k: torch.tensor(v) for k, v in parameters.items()}}
        self.fixed_parameters = {}

        # Initialise the refiner pool
        logger.info('Initialising refiner pool.')
        self.refiner_pool = RefinerPool(
            refiner_args=self.refiner_args,
            output_dir=self.refiner.output_dir,
            initial_scene_dict=scenes[0]['scene_dict'],
            fixed_parameters={},
            seed=self.sf_args.seed,
            plotter=self.plotter,
            n_workers=self.runtime_args.n_refiner_workers,
            queue_size=self.runtime_args.refiner_queue_size,
        )
        self.refiner_pool.wait_for_workers()

        # Calculate the losses
        self.losses_init = {
            'total': np.zeros(len(self), dtype=np.float32),
            'measurement': np.zeros(len(self), dtype=np.float32),
        }
        self.calculate_evaluation_losses(
            save_annotations=False,
            save_renders=False,
            initial_or_eval='initial'
        )
        with open(losses_path, 'w') as f:
            json.dump(self.losses_init, f, cls=FlexibleJSONEncoder)

        # Remove the variables
        self.refiner_pool.close()
        logger.info('Removing temporary variables.')
        del self.scenes
        del self.parameters
        del self.fixed_parameters
        del self.losses_init
        del self.refiner_pool
        del self.dataloader
        torch.cuda.empty_cache()

        return self._init_initial_prediction_losses()

    def _init_fixed_parameters(self):
        """
        Instantiate the fixed parameters.
        """
        self.fixed_parameters = {}
        if self.sf_args.initial_scene is None:
            self.initial_scene = None
            return
        assert self.sf_args.initial_scene.exists(), f'Initial scene file {self.sf_args.initial_scene} does not exist.'
        logger.info(f'Loading initial scene data from {self.sf_args.initial_scene}.')

        # Update the scene rendering parameters for optimisation
        self.initial_scene = Scene.from_yml(self.sf_args.initial_scene)
        self.initial_scene.res = self.refiner_args.rendering_size
        self.initial_scene.spp = self.refiner_args.spp
        self.initial_scene.camera_type = 'perspective'  # thinlens doesn't work for inverse rendering
        self.initial_scene.integrator_max_depth = self.refiner_args.integrator_max_depth
        self.initial_scene.integrator_rr_depth = self.refiner_args.integrator_rr_depth
        self.initial_scene.build_mi_scene()

        # Set the fixed parameters
        for k in self.sf_args.fix_parameters:
            if k == 'light_radiance':
                v = self.initial_scene.light_radiance
            else:
                v = getattr(self.initial_scene.crystal, k)
            self.fixed_parameters[k] = init_tensor(v)

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

        # If the origin is going to be fixed then we need to adjust the initial predictions to use it
        crystal = None
        if 'origin' in self.fixed_parameters:
            origin_new = init_tensor(self.fixed_parameters['origin'])
            ds = self.refiner.manager.ds
            cs = CSDProxy().load(ds.dataset_args.crystal_id)
            crystal = Crystal(
                lattice_unit_cell=cs.lattice_unit_cell,
                lattice_angles=cs.lattice_angles,
                miller_indices=ds.miller_indices,
                point_group_symbol=cs.point_group_symbol,
            )

        # Initialise the scene parameters with the initial predictions
        logger.info(f'Initialising scenes at {scenes_dir}.')
        for train_or_eval in ['train', 'eval']:
            (scenes_dir / train_or_eval).mkdir(parents=True, exist_ok=True)
        for i, scene_initial_path in enumerate(self.scenes_initial):
            if (i + 1) % 50 == 0:
                logger.info(f'Initialising scene {i + 1}/{len(self)}.')
            with open(scene_initial_path) as f:
                scene_initial_dict = yaml.load(f, Loader=yaml.FullLoader)
            c_dict = scene_initial_dict['crystal']

            # Fix the parameters if required
            for k, v in self.fixed_parameters.items():
                if k == 'origin':
                    crystal.distances.data = init_tensor(c_dict['distances'])
                    crystal.origin.data = init_tensor(c_dict['origin'])
                    crystal.adjust_origin(origin_new, verify=False)
                    c_dict['distances'] = crystal.distances.tolist()
                    c_dict['origin'] = crystal.origin.tolist()
                elif k == 'scale':
                    new_scale = self.fixed_parameters['scale'].item()
                    distances = init_tensor(c_dict['distances']) * c_dict['scale'] / new_scale
                    c_dict['scale'] = new_scale
                    c_dict['distances'] = distances.tolist()
                elif k == 'light_radiance':
                    scene_initial_dict['light_radiance'] = v.tolist()
                else:
                    if v.numel() == 1:
                        c_dict[k] = v.item()
                    else:
                        c_dict[k] = v.tolist()

            for train_or_eval in ['train', 'eval']:
                scene_path = scenes_dir / train_or_eval / scene_initial_path.name
                with open(scene_path, 'w') as f:
                    yaml.dump(scene_initial_dict, f)

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
        losses_init = {
            'train': {
                'total': np.zeros(len(self), dtype=np.float32),
                'measurement': np.zeros(len(self), dtype=np.float32),
            },
            'eval': {
                'total': np.zeros(len(self), dtype=np.float32),
                'measurement': np.zeros(len(self), dtype=np.float32),
            },
            'log': {
                'train': [],
                'eval': [],
            },
        }
        if losses_path.exists():
            with open(losses_path, 'r') as f:
                self.losses = json_to_torch(json.load(f))

            # Ensure the losses is in correct format
            if 'train' not in self.losses or 'eval' not in self.losses:
                losses = losses_init.copy()
                for k, v in self.losses.items():
                    if k == 'training':
                        losses['log']['train'] = v
                        continue
                    k_parts = k.split('/')
                    if k_parts[0] == 'total':
                        continue
                    if k_parts[0] == 'losses':
                        k_parts = k_parts[1:]
                    losses['train']['/'.join(k_parts)] = v
                self.losses = losses

            logger.info(f'Loaded losses from {losses_path}.')
            return

        logger.info(f'Initialising losses at {losses_path}.')
        with open(losses_path, 'w') as f:
            json.dump(losses_init, f, cls=FlexibleJSONEncoder)
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

    def _init_edges_annotated(self):
        """
        Instantiate the edge images with wireframe overlaid (paths only).
        """
        if not self.refiner_args.use_edge_matching:
            self.edges_annotated = None
            return
        edges_annotated_dir = self.path / 'X_edges_annotated'
        self.edges_annotated = {'train': [], 'eval': []}
        for train_or_eval in ['train', 'eval']:
            (edges_annotated_dir / train_or_eval).mkdir(parents=True, exist_ok=True)
            for (idx, _) in self.image_paths:
                edges_annotated_path = edges_annotated_dir / train_or_eval / f'{idx:04d}.png'
                self.edges_annotated[train_or_eval].append(edges_annotated_path)
        logger.info(f'Loaded annotated edge images from {edges_annotated_dir}.')

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
        if sa.seq_encoder_model == 'transformer':
            self.sequence_encoder = SequenceEncoder(
                scene_dict=self.scenes['train'][0]['scene_dict'],
                fixed_parameters=self.fixed_parameters,
                stationary_parameters=sa.stationary_parameters,
                hidden_dim=sa.hidden_dim,
                n_layers=sa.n_layers,
                n_heads=sa.n_heads,
                dropout=sa.dropout,
                activation=sa.activation,
            )
        elif sa.seq_encoder_model == 'ffn':
            self.sequence_encoder = SequenceEncoderFFN(
                param_dim=self.n_parameters,
                hidden_dim=sa.hidden_dim,
                n_layers=sa.n_layers,
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
            if not self.runtime_args.reset_lrs:
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
        ra = self.runtime_args
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
                sampler.emas[i].val = self.losses['train']['total'][i].item()

        # Dataset
        self.dataloader = get_data_loader(
            sequence_fitter=self,
            adaptive_sampler=sampler,
            batch_size=bs,
            n_workers=ra.n_dataloader_workers,
            prefetch_factor=ra.prefetch_factor,
        )

        # Load the first scene and adapt the crystal to use buffers instead of parameters
        if self.initial_scene is not None:
            scene = self.initial_scene
        else:
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
        self.refiner.init_projector()
        use_inverse_rendering = self.refiner_args.use_inverse_rendering  # Save the inverse rendering setting

        # Set up the refiner pool
        if ra.n_refiner_workers > 0:
            logger.info('Initialising refiner pool.')
            self.refiner_pool = RefinerPool(
                refiner_args=self.refiner_args,
                output_dir=self.refiner.output_dir,
                initial_scene_dict=scene.to_dict(),
                fixed_parameters=self.fixed_parameters,
                seed=self.sf_args.seed,
                plotter=self.plotter,
                n_workers=ra.n_refiner_workers,
                queue_size=ra.refiner_queue_size,
            )
            self.refiner_pool.wait_for_workers()

        # Determine how many steps have been completed from the frame counts
        start_step = len(self.losses['log']['train'])
        # assert start_step == int(self.frame_counts.sum() / bs), 'Frame counts and training losses do not match.'
        if start_step > 0:
            logger.info(f'Resuming training from step {start_step}.')
        max_steps = self.sf_args.max_steps

        # Train the sequence encoder
        running_loss = 0.
        running_tps = 0
        step = start_step - 1
        for batch_idxs, batch in self.dataloader:
            step += 1
            start_time = time.time()
            self.sequence_encoder.train()
            self.refiner.step = step

            # Normalise frame indices to time points
            ts = (batch_idxs / (N - 1)).to(self.device)

            # Generate the parameters from the encoder
            parameters_pred = self.sequence_encoder(ts)
            p_grads = torch.zeros_like(parameters_pred)

            # Conditionally enable the inverse rendering
            self.refiner_args.use_inverse_rendering = use_inverse_rendering and step >= self.refiner_args.ir_wait_n_steps

            # Calculate the gradients in parallel using the refiner pool
            if self.refiner_pool is not None:
                losses_batch, stats_batch, p_grads = self.refiner_pool.calculate_losses(
                    step=step,
                    refiner_args=self.refiner_args,
                    p_vec_batch=parameters_pred,
                    X_target=batch[0],
                    X_target_denoised=batch[1],
                    X_target_wis=batch[2],
                    X_target_denoised_wis=batch[3],
                    keypoints=batch[4],
                    edges=batch[5],
                    calculate_grads=True,
                    save_annotations=(ra.save_annotations_freq > 0
                                      and (step + 1) % ra.save_annotations_freq == 0),
                    save_edge_annotations=(ra.save_edge_annotations_freq > 0
                                           and (step + 1) % ra.save_edge_annotations_freq == 0),
                    save_renders=ra.save_renders_freq > 0 and (step + 1) % ra.save_renders_freq == 0,
                    X_preds_paths=[self.X_preds['train'][idx] for idx in batch_idxs],
                    X_targets_paths=[self.X_targets[idx] for idx in batch_idxs],
                    X_targets_annotated_paths=[self.X_targets_annotated['train'][idx] for idx in batch_idxs],
                    edges_fullsize_paths=[self.edges_fullsize[idx]
                                          for idx in batch_idxs] if self.edges_fullsize is not None else None,
                    edges_annotated_paths=[self.edges_annotated['train'][idx]
                                           for idx in batch_idxs] if self.edges_annotated is not None else None,
                )
                p_grads = p_grads.to(self.device)
                p_dict = self._parameter_vectors_to_dict(parameters_pred)

                # Compute the negative growth penalty loss
                if self.sf_args.w_negative_growth > 0:
                    d = p_dict['distances']
                    t_order = ts.argsort()
                    d_sorted = d[t_order]
                    neg_growth = (d_sorted[1:] - d_sorted[:-1]).clamp(max=0)**2
                    l_neg = neg_growth.sum()
                    reg_grads = torch.autograd.grad(l_neg * self.sf_args.w_negative_growth, parameters_pred)[0]
                    p_grads = p_grads + reg_grads

                    # Spread out the negative growth loss between neighbouring frames for logging
                    neg_growth_batch = torch.zeros_like(d)
                    neg_growth_batch[1:] += neg_growth.detach() / 2
                    neg_growth_batch[:-1] += neg_growth.detach() / 2
                    neg_growth_batch = neg_growth_batch[t_order.argsort()]  # reorder back to the batch ordering
                    stats_batch['losses/negative_growth'] = neg_growth_batch.sum(dim=1).tolist()

                # Update scene and crystal parameters
                with torch.no_grad():
                    for i, idx in enumerate(batch_idxs):
                        self.frame_counts[idx] += 1
                        self.losses['train']['total'][idx] = losses_batch[i].item()
                        for k, v in stats_batch.items():
                            k = k if k[:7] != 'losses/' else k[7:]
                            if k not in self.losses['train']:
                                self.losses['train'][k] = torch.zeros(N)
                            self.losses['train'][k][idx] = float(v[i])
                        for k, v in p_dict.items():
                            self.parameters['train'][k][idx] = v[i].clone().detach().cpu()

                        # Calculate the measurement loss
                        if self.measurements is not None and idx in self.measurements['idx']:
                            idx_m = np.where(self.measurements['idx'] == idx)[0][0]
                            dists_m = self.measurements['distances'][idx_m] * self.measurements['scale'][idx_m]
                            dists_p = to_numpy(p_dict['distances'][i] * p_dict['scale'][i])
                            loss_m = np.sum((dists_m - dists_p)**2)
                            self.losses['train']['measurement'][idx] = loss_m.item()

            # Loop over the batch and use the refiner to calculate the loss for each frame
            else:
                losses_batch = torch.zeros(bs)
                for i in range(bs):
                    idx = batch_idxs[i]
                    p_vec = parameters_pred[i].clone().detach().requires_grad_(True)
                    p_dict = self._parameter_vectors_to_dict(p_vec[None, ...])

                    # Update targets
                    self.refiner.X_target = batch[0][i]
                    self.refiner.X_target_denoised = batch[1][i]
                    self.refiner.X_target_wis = batch[2][i].to(self.device)
                    self.refiner.X_target_denoised_wis = batch[3][i].to(self.device)
                    if self.refiner_args.use_keypoints:
                        self.refiner.keypoint_targets = self.keypoints[idx]['keypoints']

                    # Update scene and crystal parameters
                    for k, v in p_dict.items():
                        if k == 'light_radiance':
                            scene.light_radiance = v[0].to(self.device)
                        else:
                            setattr(crystal, k, v[0].to('cpu'))
                        self.parameters['train'][k][idx] = v[0].clone().detach().cpu()

                    # Calculate losses
                    loss_i, stats_i = self.refiner.process_step(add_noise=False)
                    loss_i.backward()
                    losses_batch[i] = loss_i.item()

                    # Accumulate gradients
                    p_grads[i] = p_vec.grad.clone()

                    # Update sequence state
                    self.losses['train']['total'][idx] = loss_i.item()
                    self.frame_counts[idx] += 1
                    if self.refiner_args.use_inverse_rendering:
                        self.plotter.save_image(self.refiner.X_pred, self.X_preds['train'][idx])
                    self.plotter.annotate_image(self.X_targets[idx], self.X_targets_annotated['train'][idx], scene)

            # Propagate the parameter gradients back to the encoder
            torch.autograd.backward(parameters_pred, grad_tensors=p_grads)

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
            loss = losses_batch.mean().item()
            sampler.update_errors(batch_idxs, losses_batch)
            seq_loss_ema = sampler.errors.sum().item()
            seq_loss = self.losses['train']['total'].sum().item()
            mes_loss = self.losses['train']['measurement'].sum().item()
            self.losses['log']['train'] = torch.concatenate([self.losses['log']['train'],
                                                             torch.tensor(seq_loss)[None, ...]])
            self.tb_logger.add_scalar('losses/seq_train', seq_loss, step)
            self.tb_logger.add_scalar('losses/seq_train_ema', seq_loss_ema, step)
            self.tb_logger.add_scalar('losses/batch', loss, step)
            self.tb_logger.add_scalar('losses/measurement_train', mes_loss, step)
            for k in stats_batch.keys():
                if 'losses' not in k:
                    continue
                kk = k.split('/')[1]
                seq_loss_k = self.losses['train'][kk]
                seq_loss_k = (seq_loss_k[seq_loss_k > 0]).mean().item()
                self.tb_logger.add_scalar(f'stats/{kk}', seq_loss_k, step)

            # Log learning rate and update
            self.tb_logger.add_scalar('lr', self.optimiser.param_groups[0]['lr'], step)
            self.lr_scheduler.step(step + 1, seq_loss_ema + self.refiner_args.ir_loss_placeholder)

            # Log network norm
            weights_cumulative_norm = calculate_model_norm(self.sequence_encoder, device=self.device)
            assert not is_bad(weights_cumulative_norm), 'Bad parameters!'
            self.tb_logger.add_scalar('w_norm', weights_cumulative_norm.item(), step)

            # Log statistics every X steps
            time_per_step = time.time() - start_time
            running_tps += time_per_step
            running_loss += loss
            if (step + 1) % ra.log_freq_train == 0:
                loss_avg = running_loss / ra.log_freq_train
                average_tps = running_tps / ra.log_freq_train
                seconds_left = float((max_steps - step) * average_tps)
                logger.info(f'[{step + 1}/{max_steps}]\tLoss: {loss_avg:.4E}'
                            + '\tTime per step: {}\tEst. complete in: {}'.format(
                    str(timedelta(seconds=average_tps)),
                    str(timedelta(seconds=seconds_left))))
                running_loss = 0.
                running_tps = 0

            # Checkpoint every X steps
            if (step + 1) % ra.checkpoint_freq == 0:
                self.save()
                self._save_encoder_state()

            # Plot every X steps
            if (step + 1) % ra.plot_freq == 0:
                logger.info('Making training plots.')
                if self.refiner_pool is None:  # Needs fixing for refiner pool
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
            if (step + 1) % ra.eval_freq == 0:
                logger.info('Evaluating the sequence.')
                self.update_parameters_from_encoder(save=False)
                save_annotations = ra.eval_annotate_freq > 0 and (step + 1) % ra.eval_annotate_freq == 0
                save_edge_annotations = ra.eval_edge_annotate_freq > 0 and (step + 1) % ra.eval_edge_annotate_freq == 0
                save_renders = ra.eval_render_freq > 0 and (step + 1) % ra.eval_render_freq == 0
                self.calculate_evaluation_losses(
                    save_annotations=save_annotations,
                    save_edge_annotations=save_edge_annotations,
                    save_renders=save_renders,
                )
                self.save()
                self._plot_parameters('eval')
                if ra.eval_video_freq > 0 and (step + 1) % ra.eval_video_freq == 0:
                    logger.info('Generating evaluation video.')
                    self._generate_video(train_or_eval='eval')

        logger.info('Training complete.')

    @torch.no_grad()
    def update_parameters_from_encoder(self, train_or_eval: str = 'eval', save: bool = True):
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
        if save:
            self.save()

    @torch.no_grad()
    def calculate_evaluation_losses(
            self,
            save_annotations: bool = True,
            save_edge_annotations: bool = True,
            save_renders: bool = True,
            initial_or_eval: str = 'eval'
    ):
        """
        Calculate the evaluation losses.
        """
        if initial_or_eval == 'eval':
            logger.info(f'Calculating evaluation losses.')
            step = self.refiner.step
            losses = self.losses['eval']
            X_preds = self.X_preds['eval']
            X_targets_annotated = self.X_targets_annotated['eval']

        else:
            logger.info(f'Calculating initial prediction losses.')
            step = 0
            losses = self.losses_init
            X_preds = self.X_preds_initial
            X_targets_annotated = self.X_targets_annotated_initial
            save_edge_annotations = False

        ds = self.dataloader.dataset

        # Disable the cropping so the renders are generated at full size
        crop_renders = self.refiner_args.crop_render
        self.refiner_args.crop_render = False

        # Process the entire sequence in batches
        bs = min(len(self), self.runtime_args.n_refiner_workers * 4)
        n_batches = math.ceil(len(self) / bs)
        log_freq = int(n_batches / 4)
        for i in range(n_batches):
            if (i + 1) % log_freq == 0:
                logger.info(f'Calculating evaluation losses for frame batch {i + 1}/{n_batches}.')

            # Load the batch
            batch_idxs = list(range(i * bs, min((i + 1) * bs, len(self))))
            p_vec_batch = torch.stack([
                self.get_parameter_vector(idx) for idx in batch_idxs
            ]).to(self.device)
            p_dict_batch = self._parameter_vectors_to_dict(p_vec_batch)
            batch = [ds[idx] for idx in batch_idxs]

            # Calculate losses and stats
            losses_batch, stats_batch, _ = self.refiner_pool.calculate_losses(
                step=step,
                refiner_args=self.refiner_args,
                p_vec_batch=p_vec_batch,
                X_target=[X[0] for X in batch],
                X_target_denoised=[X[1] for X in batch],
                X_target_wis=torch.stack([X[2] for X in batch]).permute(0, 2, 3, 1),
                X_target_denoised_wis=torch.stack([X[3] for X in batch]).permute(0, 2, 3, 1),
                keypoints=[X[4] for X in batch] if batch[0][4] is not None else [None for _ in batch],
                edges=[X[5] for X in batch] if batch[0][5] is not None else [None for _ in batch],
                calculate_grads=False,
                save_annotations=save_annotations,
                save_edge_annotations=save_edge_annotations,
                save_renders=save_renders,
                X_preds_paths=[X_preds[idx] for idx in batch_idxs],
                X_targets_paths=[self.X_targets[idx] for idx in batch_idxs],
                X_targets_annotated_paths=[X_targets_annotated[idx] for idx in batch_idxs],
                edges_fullsize_paths=[self.edges_fullsize[idx] for idx in
                                      batch_idxs] if self.edges_fullsize is not None else None,
                edges_annotated_paths=[self.edges_annotated['eval'][idx] for idx in
                                       batch_idxs] if self.edges_annotated is not None else None,
            )

            # Compute the negative growth penalty loss
            if self.sf_args.w_negative_growth > 0:
                d = p_dict_batch['distances']
                neg_growth = (d[1:] - d[:-1]).clamp(max=0)**2
                neg_growth_batch = torch.zeros_like(d)
                neg_growth_batch[1:] += neg_growth.detach() / 2
                neg_growth_batch[:-1] += neg_growth.detach() / 2
                stats_batch['losses/negative_growth'] = neg_growth_batch.sum(dim=1).tolist()

            # Update scene and crystal parameters
            for j, idx in enumerate(batch_idxs):
                losses['total'][idx] = losses_batch[j].item()
                for k, v in stats_batch.items():
                    k = k if k[:7] != 'losses/' else k[7:]
                    if k not in losses:
                        losses[k] = torch.zeros(len(self))
                    losses[k][idx] = float(v[j])

                # Calculate the measurement loss
                if self.measurements is not None and idx in self.measurements['idx']:
                    idx_m = np.where(self.measurements['idx'] == idx)[0][0]
                    dists_m = self.measurements['distances'][idx_m] * self.measurements['scale'][idx_m]
                    dists_p = to_numpy(p_dict_batch['distances'][j] * p_dict_batch['scale'][j])
                    loss_m = np.sum((dists_m - dists_p)**2)
                    losses['measurement'][idx] = loss_m.item()

        # Restore the cropping setting
        self.refiner_args.crop_render = crop_renders

        # Log the total sequence loss
        if initial_or_eval == 'eval':
            seq_loss = losses['total'].sum().item()
            mes_loss = losses['measurement'].sum().item()
            self.losses['log']['eval'] = torch.concatenate([self.losses['log']['eval'],
                                                            torch.tensor(seq_loss)[None, ...]])
            self.tb_logger.add_scalar('losses/seq_eval', seq_loss, step)
            self.tb_logger.add_scalar('losses/measurement_eval', mes_loss, step)
            for k in stats_batch.keys():
                if 'losses' not in k:
                    continue
                kk = k.split('/')[1]
                seq_loss_k = self.losses['eval'][kk]
                seq_loss_k = (seq_loss_k[seq_loss_k > 0]).mean().item()
                self.tb_logger.add_scalar(f'stats_eval/{kk}', seq_loss_k, step)

    @torch.no_grad()
    def update_X_preds_from_parameters(
            self,
            train_or_eval: str = 'eval',
            generate_renders: bool = True,
            generate_annotations: bool = True,
            generate_edge_annotations: bool = True,
    ):
        """
        Generate rendered and annotated images from the parameters.
        """
        logger.info('Generating predicted images from the parameters.')
        generate_edge_annotations = generate_edge_annotations and self.edges_annotated is not None
        scene = Scene.from_yml(self.scenes[train_or_eval][0]['path'])
        crystal = scene.crystal
        crystal.to('cpu')
        for i in range(len(self)):
            if (i + 1) % 50 == 0:
                logger.info(f'Generating image {i + 1}/{len(self)}.')
            for k in PARAMETER_KEYS:
                v = init_tensor(self.parameters[train_or_eval][k][i])
                if k == 'light_radiance':
                    scene.light_radiance.data = v.to(scene.device)
                else:
                    getattr(crystal, k).data = v
            if generate_renders:
                self.plotter.render_scene(scene, self.X_preds[train_or_eval][i])
            if generate_annotations or generate_edge_annotations:
                crystal.build_mesh()  # Required to update the buffers
            if generate_annotations:
                self.plotter.annotate_image(self.X_targets[i], self.X_targets_annotated[train_or_eval][i], scene)
            if generate_edge_annotations:
                self.plotter.annotate_image(self.edges_fullsize[i], self.edges_annotated[train_or_eval][i], scene)

    def _plot_parameters(self, train_or_eval: str):
        """
        Plot the parameters.
        """
        self.plotter.plot_sequence_parameters(
            plot_dir=self.path / 'plots' / train_or_eval,
            step=self.refiner.step,
            parameters=self.parameters[train_or_eval],
            image_paths=self.image_paths,
            face_groups=get_crystal_face_groups(self.refiner.manager)
        )
        self.plotter.plot_sequence_losses(
            plot_dir=self.path / 'plots' / train_or_eval,
            step=self.refiner.step,
            losses=self.losses[train_or_eval],
            image_paths=self.image_paths,
        )

    def _generate_video(self, train_or_eval: str = 'eval'):
        """
        Generate a video from the images.
        """
        self.plotter.generate_video(
            imgs_dir=self.X_targets_annotated[train_or_eval][0].parent,
            train_or_eval=train_or_eval,
            save_root=self.path,
            step=self.refiner.step + 1,
        )

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
        Convert a batch of parameter vectors to a dictionary of parameters, with fixed parameter replacements.
        """
        parameters = {}
        bs = parameter_vectors.shape[0]
        idx = 0
        for k, v in self.parameters['eval'].items():
            n = v.shape[1] if v.ndim > 1 else 1
            val = parameter_vectors[:, idx:idx + n].squeeze(1)
            if k in self.fixed_parameters:
                val = self.fixed_parameters[k][None, ...].repeat(bs, 1).squeeze(1)
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
