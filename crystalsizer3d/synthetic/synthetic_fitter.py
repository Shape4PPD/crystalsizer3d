import gc
import json
import math
import shutil
from argparse import Namespace
from typing import Dict

import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from crystalsizer3d import DATA_PATH, LOGS_PATH, logger
from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs
from crystalsizer3d.args.refiner_args import DENOISER_ARG_NAMES, KEYPOINTS_ARG_NAMES, PREDICTOR_ARG_NAMES, \
    PREDICTOR_ARG_NAMES_BS1, RefinerArgs
from crystalsizer3d.args.synthetic_fitter_args import SyntheticFitterArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.nn.dataset import Dataset
from crystalsizer3d.refiner.denoising import denoise_image
from crystalsizer3d.refiner.keypoint_detection import find_keypoints
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.sequence.refiner_pool import RefinerPool
from crystalsizer3d.sequence.sequence_plotter import SequencePlotter
from crystalsizer3d.util.utils import FlexibleJSONEncoder, hash_data, init_tensor, json_to_torch, set_seed, to_dict, \
    to_numpy

# Suppress mitsuba warning messages
mi.set_log_level(mi.LogLevel.Error)

PARAMETER_KEYS = ['scale', 'distances', 'origin', 'rotation', 'material_roughness', 'material_ior', 'light_radiance']

ARG_NAMES = {
    'denoiser': DENOISER_ARG_NAMES,
    'keypoints': KEYPOINTS_ARG_NAMES,
    'predictor': PREDICTOR_ARG_NAMES,
}

resize_args = dict(mode='bilinear', align_corners=False)


class SyntheticFitter:
    dataloader: DataLoader

    def __init__(
            self,
            sf_args: SyntheticFitterArgs,
            refiner_args: RefinerArgs,
            runtime_args: Namespace,
    ):
        self.sf_args = sf_args
        self.refiner_args = refiner_args
        assert self.refiner_args.initial_pred_batch_size == 1, 'Initial predictions only supported for batch size 1.'
        self.runtime_args = runtime_args
        if self.sf_args.seed is not None:
            set_seed(self.sf_args.seed)

        # Initialise the dataset
        self.dataset_args = DatasetTrainingArgs(
            dataset_path=sf_args.dataset_path,
        )
        self.ds = Dataset(self.dataset_args)
        ds_idxs = self.ds.train_idxs.copy() if self.sf_args.train_or_test == 'train' else self.ds.test_idxs.copy()
        np.random.shuffle(ds_idxs)
        self.ds_idxs = ds_idxs[:self.sf_args.n_samples]
        self.image_size = self.ds.dataset_args.image_size
        assert self.refiner_args.rendering_size == self.image_size, 'Rendering size must match dataset image size.'

        # Initialise the output directories
        self.base_path = LOGS_PATH / f'ds={self.dataset_args.dataset_path.name}_{self.sf_args.train_or_test}'
        self.path = self.base_path / 'sf' / hash_data([sf_args.to_dict(), refiner_args.to_dict()])
        self._init_output_dirs()

        # Initialise the refiner and the asynchronous plotter
        self._init_refiner()
        self.refiner_pool = None  # Instantiated on demand
        self._init_plotter()

        # Initialise the target data
        self._init_X_targets()
        self._init_X_targets_denoised()
        self._init_keypoints()

        # Initialise the outputs
        self._init_initial_predictions()
        self._init_initial_prediction_losses()
        self._init_fixed_parameters()
        self._init_scenes()
        self._init_parameters()
        self._init_losses()
        self._init_X_preds()
        self._init_X_targets_annotated()
        self._init_tb_logger()

    def _init_output_dirs(self):
        """
        Ensure the output directories exists with the correct arguments.
        """
        self.cache_dirs = {}
        for model_name, arg_names in ARG_NAMES.items():
            model_args = {k: getattr(self.refiner_args, k) for k in arg_names}
            if model_name == 'predictor':
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

        # Clear the output directory if it already exists and we're not resuming
        if self.path.exists() and not self.runtime_args.resume:
            shutil.rmtree(self.path)

        # If the output directory exists, but we're asked to resume from a different checkpoint, abort
        elif self.path.exists() and self.runtime_args.resume_from is not None:
            raise RuntimeError('Resuming from a different run will remove the current run\'s data. '
                               f'Please remove the current run\'s data manually from {self.path} and try again.')

        # Create the output directory
        self.path.mkdir(parents=True, exist_ok=True)

        # Save the ds idxs
        with open(self.path / 'ds_idxs.json', 'w') as f:
            json.dump(self.ds_idxs, f, cls=FlexibleJSONEncoder)

        # Initialise from a previous checkpoint
        if self.runtime_args.resume_from is not None:
            assert self.runtime_args.resume_from.exists(), f'Resume from path does not exist: {self.runtime_args.resume_from}'
            with open(self.runtime_args.resume_from / 'ds_idxs.json', 'r') as f:
                assert self.ds_idxs == json.load(f), 'The dataset indices must match the previous run.'
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

        # Save the args into the output directory
        with open(self.path / 'args_runtime.yml', 'w') as f:
            yaml.dump(to_dict(self.runtime_args), f)
        with open(self.path / 'args_synthetic_fitter.yml', 'w') as f:
            yaml.dump(to_dict(self.sf_args), f)
        with open(self.path / 'args_refiner.yml', 'w') as f:
            yaml.dump(to_dict(self.refiner_args), f)

    def _init_refiner(self):
        """
        Instantiate the refiner.
        """
        logger.info('Initialising refiner.')
        self.refiner_args.image_path = self.ds.data[0]['image']
        self.refiner = Refiner(
            args=self.refiner_args,
            output_dir=self.path / 'refiner',
            do_init=False,
        )
        self.refiner.save_dir = self.path / 'refiner'
        (self.path / 'refiner').mkdir(parents=True, exist_ok=True)
        self.manager = self.refiner.manager
        self.device = self.refiner.device

    def _init_plotter(self):
        """
        Instantiate the asynchronous plotter.
        """
        self.plotter = SequencePlotter(
            n_workers=self.runtime_args.n_plotting_workers,
            queue_size=self.runtime_args.plot_queue_size,
        )
        self.plotter.wait_for_workers()

    def _init_X_targets(self):
        """
        Load the synthetic image paths.
        """
        X_targets_dir = self.ds.path / 'images'
        assert X_targets_dir.exists(), 'Image directory does not exist!'
        self.X_targets = []
        for idx in self.ds_idxs:
            X_target_path = self.ds.data[idx]['image']
            assert X_target_path.exists()
            self.X_targets.append(X_target_path)
        logger.info(f'Loaded target images from {X_targets_dir}.')

    @torch.no_grad()
    def _init_X_targets_denoised(self):
        """
        Instantiate the denoised data.
        """
        X_targets_denoised_dir = self.cache_dirs['denoiser'] / 'fullsize'
        try:
            self.X_targets_denoised = []
            for X_target_path in self.X_targets:
                X_target_denoised_path = X_targets_denoised_dir / X_target_path.name
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
        logger.info('Denoising synthetic images.')
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
            self.plotter.save_image(X_target_denoised, X_targets_denoised_dir / X_target_path.name)

        # Destroy the denoiser to free up space
        logger.info('Destroying denoiser to free up space.')
        self.manager.denoiser = None
        torch.cuda.empty_cache()
        gc.collect()

        # Wait for the plotter workers
        self.plotter.wait_for_workers()

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
        keypoints_dir = self.cache_dirs['keypoints'] / 'keypoints'
        annotations_dir = self.cache_dirs['keypoints'] / 'annotations'
        if keypoints_dir.exists() and annotations_dir.exists():
            try:
                self.keypoints = []
                for i, idx in enumerate(self.ds_idxs):
                    keypoints_path = keypoints_dir / f'{idx:010d}.json'
                    assert keypoints_path.exists()
                    annotation_path = annotations_dir / f'{idx:010d}.png'
                    assert annotation_path.exists()
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
        keypoints_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Load the keypoint detector
        self.manager.load_network(ra.keypoints_model_path, 'keypointdetector')
        assert self.manager.keypoint_detector is not None, 'No keypoints model loaded, so can\'t predict keypoints.'

        # Find the keypoints for each image
        logger.info('Finding keypoints in synthetic images.')
        for i, (idx, X_path, X_denoised_path) in enumerate(zip(self.ds_idxs, self.X_targets, self.X_targets_denoised)):
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
                n_patches=ra.keypoints_n_patches,
                patch_size=ra.keypoints_patch_size,
                patch_search_res=ra.keypoints_patch_search_res,
                attenuation_sigma=ra.keypoints_attenuation_sigma,
                max_attenuation_factor=ra.keypoints_max_attenuation_factor,
                low_res_catchment_distance=ra.keypoints_low_res_catchment_distance,
                return_everything=False,
                quiet=True,
                n_workers=1
            )
            if len(res) == 0:
                res = []
            with open(keypoints_dir / f'{idx:010d}.json', 'w') as f:
                json.dump(res, f, cls=FlexibleJSONEncoder)
            self.plotter.annotate_image_with_keypoints(X_path, res, annotations_dir / f'{idx:010d}.png')

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
                for i, idx in enumerate(self.ds_idxs):
                    scene_path = scenes_initial_dir / f'{idx:010d}.yml'
                    X_pred_path = X_preds_initial_dir / f'{idx:010d}.png'
                    X_annotated_path = X_annotated_initial_dir / f'{idx:010d}.png'
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
        assert ra.initial_pred_batch_size == 1, 'Initial predictions only supported for batch size 1.'

        # Make directories
        scenes_initial_dir.mkdir(parents=True, exist_ok=True)
        X_preds_initial_dir.mkdir(parents=True, exist_ok=True)
        X_annotated_initial_dir.mkdir(parents=True, exist_ok=True)

        # Set up the resizing
        self.manager.predictor.resize_input = False
        oversize_input = ra.initial_pred_oversize_input
        img_size = ra.initial_pred_max_img_size if oversize_input else self.manager.image_shape[-1]

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
            idx = self.ds_idxs[i]
            scene_path = scenes_initial_dir / f'{idx:010d}.yml'
            X_pred_path = X_preds_initial_dir / f'{idx:010d}.png'
            X_annotated_path = X_annotated_initial_dir / f'{idx:010d}.png'
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
        # correct losses even though the predictor batch size was 1
        model_args = {k: getattr(self.refiner_args, k) for k in PREDICTOR_ARG_NAMES}
        losses_path = self.cache_dirs['predictor'] / f'losses_{hash_data(model_args)}_{hash_data(self.ds_idxs)}.json'
        try:
            with open(losses_path) as f:
                self.losses_init = json_to_torch(json.load(f))
            assert len(self.losses_init['total']) == len(self.ds_idxs)
            logger.info(f'Loaded initial prediction losses from {losses_path}.')
            return
        except Exception:
            pass
        logger.info('Calculating initial prediction losses.')

        # Load the initial scenes and parameters into the 'current' space
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
        self.scenes = scenes
        self.parameters = {'current': {k: torch.tensor(v) for k, v in parameters.items()}}
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
        }
        self.calculate_current_losses(
            save_annotations=False,
            save_renders=False,
            initial_or_current='initial'
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

    def _init_scenes(self):
        """
        Instantiate the scenes.
        """
        scenes_dir = self.path / 'scenes'
        if scenes_dir.exists():
            try:
                self.scenes = []
                for i, idx in enumerate(self.ds_idxs):
                    scene_path = scenes_dir / f'{idx:010d}.yml'
                    with open(scene_path) as f:
                        scene_dict = yaml.load(f, Loader=yaml.FullLoader)
                    self.scenes.append({
                        'path': scene_path,
                        'scene_dict': scene_dict,
                    })
                logger.info(f'Loaded scenes from {scenes_dir}.')
                return
            except Exception:
                pass

        # Initialise the scene parameters with the initial predictions
        logger.info(f'Initialising scenes at {scenes_dir}.')
        scenes_dir.mkdir(parents=True, exist_ok=True)
        for i, scene_initial_path in enumerate(self.scenes_initial):
            if (i + 1) % 50 == 0:
                logger.info(f'Initialising scene {i + 1}/{len(self)}.')
            with open(scene_initial_path) as f:
                scene_initial_dict = yaml.load(f, Loader=yaml.FullLoader)
            scene_path = scenes_dir / scene_initial_path.name
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
            self.n_parameters = sum([v.shape[1] if v.ndim > 1 else 1 for v in self.parameters['current'].values()])
            logger.info(f'Loaded parameters from {parameters_path}.')
            return

        # Initialise the parameters from the scenes
        logger.info(f'Initialising parameters at {parameters_path}.')
        parameters = {'current': {}, 'checkpoints': {}}
        for scene in self.scenes:
            scene_dict = scene['scene_dict']
            for k in PARAMETER_KEYS:
                if k not in parameters['current']:
                    parameters['current'][k] = []
                if k in scene_dict:
                    v = scene_dict[k]
                elif k in scene_dict['crystal']:
                    v = scene_dict['crystal'][k]
                else:
                    raise RuntimeError(f'Parameter {k} not found in scene or crystal.')
                parameters['current'][k].append(v)
        with open(parameters_path, 'w') as f:
            json.dump(parameters, f, cls=FlexibleJSONEncoder)

        return self._init_parameters()

    def _init_losses(self):
        """
        Instantiate the losses.
        """
        losses_path = self.path / 'losses.json'
        losses_init = {
            'init': self.losses_init,
            'current': {
                'total': np.zeros(len(self), dtype=np.float32),
            },
            'checkpoints': {},
        }
        if losses_path.exists():
            try:
                with open(losses_path, 'r') as f:
                    self.losses = json_to_torch(json.load(f))
                assert len(self.losses['init']['total']) == len(self.ds_idxs)
                assert len(self.losses['current']['total']) == len(self.ds_idxs)
                logger.info(f'Loaded losses from {losses_path}.')
                return
            except Exception:
                pass
        logger.info(f'Initialising losses at {losses_path}.')
        with open(losses_path, 'w') as f:
            json.dump(losses_init, f, cls=FlexibleJSONEncoder)
        return self._init_losses()

    def _init_X_preds(self):
        """
        Instantiate the predicted (rendered) images.
        """
        X_preds_dir = self.path / 'X_preds'
        if X_preds_dir.exists():
            try:
                self.X_preds = {'train': [], 'eval': []}
                for idx in self.ds_idxs:
                    for train_or_eval in ['train', 'eval']:
                        X_pred_path = X_preds_dir / train_or_eval / f'{idx:010d}.png'
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
                self.X_targets_annotated = []
                for idx in self.ds_idxs:
                    X_annotated_path = X_annotated_dir / f'{idx:010d}.png'
                    assert X_annotated_path.exists()
                    self.X_targets_annotated.append(X_annotated_path)
                logger.info(f'Loaded annotated images from {X_annotated_dir}.')
                return
            except Exception:
                pass

        # Initialise the annotated images with the initial predictions
        logger.info(f'Initialising annotated images at {X_annotated_dir}.')
        X_annotated_dir.mkdir(parents=True, exist_ok=True)
        for X_annotated_path in self.X_targets_annotated_initial:
            shutil.copy(X_annotated_path, X_annotated_dir)

        return self._init_X_targets_annotated()

    def _init_tb_logger(self):
        """
        Instantiate the tensorboard logger.
        """
        tb_dir = self.path / 'tensorboard'
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.tb_logger = SummaryWriter(tb_dir, flush_secs=5)

    def fit(self):
        """
        Fit the synthetic crystal parameters.
        """
        logger.info('Fitting synthetic crystal parameters.')
        ra = self.runtime_args
        N = len(self)
        assert ra.n_refiner_workers > 0, 'Refiner workers must be greater than 0.'

        # Load the first scene and adapt the crystal to use buffers instead of parameters
        if self.initial_scene is not None:
            scene = self.initial_scene
        else:
            scene = Scene.from_yml(self.scenes[0]['path'])

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

        # Break up the total number of steps into checkpoints and process the whole dataset in chunks
        total_steps = self.refiner_args.max_steps
        checkpoint_freq = self.sf_args.checkpoint_freq
        self.refiner_args.max_steps = checkpoint_freq
        checkpoint_steps = list(range(checkpoint_freq, total_steps + 1, checkpoint_freq))
        for i, checkpoint_step in enumerate(checkpoint_steps):
            start_step = 0 if i == 0 else checkpoint_steps[i - 1]
            end_step = checkpoint_step
            ck = str(checkpoint_step)

            # Create the output paths for this checkpoint
            checkpoint_path = self.path / 'checkpoints' / f'{checkpoint_step:06d}'
            losses_path = checkpoint_path / 'losses.json'
            stats_path = checkpoint_path / 'stats.json'
            parameters_path = checkpoint_path / 'parameters.json'

            # Check if this checkpoint has already been completed
            if losses_path.exists() and stats_path.exists() and parameters_path.exists():
                try:
                    with open(losses_path) as f:
                        losses = json.load(f)
                    assert all([str(idx) in losses for idx in self.ds_idxs])
                    assert str(ck) in self.losses['checkpoints']
                    with open(stats_path) as f:
                        stats = json.load(f)
                    assert all([str(idx) in stats for idx in self.ds_idxs])
                    with open(parameters_path) as f:
                        parameters = json.load(f)
                    assert all([str(idx) in parameters for idx in self.ds_idxs])
                    assert str(ck) in self.parameters['checkpoints']
                    logger.info(f'Checkpoint {ck} already completed.')
                    continue
                except Exception:
                    pass

            # Remove any half-completed checkpoint data
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)

            # Process the next lot of refinement steps for the whole dataset
            logger.info(f'Processing steps {start_step}-{end_step} ({i + 1}/{len(checkpoint_steps)}).')
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            for j, idx in enumerate(self.ds_idxs):
                logger.info(f'Queuing refiner job for image idx {idx:010d} ({j + 1}/{len(self)}).')
                distances_target = init_tensor(self.ds.data[idx]['rendering_parameters']['crystal']['distances']) \
                                   * init_tensor(self.ds.data[idx]['rendering_parameters']['crystal']['scale'])
                self.refiner_pool.refine(
                    refiner_args=self.refiner_args,
                    start_step=start_step,
                    idx=idx,
                    p_vec=self.get_parameter_vector(j),
                    X_target=self.X_targets[j],
                    X_target_denoised=self.X_targets_denoised[j],
                    X_target_wis=self.X_targets[j],
                    X_target_denoised_wis=self.X_targets_denoised[j],
                    keypoints=self.keypoints[j]['keypoints'] if self.keypoints is not None else None,
                    edges=None,
                    distances_target=distances_target,
                    vertices_target=init_tensor(self.ds.data[idx]['vertices']),
                    losses_path=losses_path,
                    stats_path=stats_path,
                    parameters_path=parameters_path,
                    plots_path=self.path / 'refiner' / f'{idx:010d}',
                )
            self.refiner_pool.wait_for_workers()

            # Update scene and crystal parameters
            logger.info(f'Copying checkpoint data to the current training set.')

            with open(losses_path) as f:
                losses = json.load(f)
            with open(stats_path) as f:
                stats = json.load(f)
            with open(parameters_path) as f:
                parameters = json.load(f)

            for j, idx in enumerate(self.ds_idxs):
                idx = str(idx)
                self.losses['current']['total'][j] = losses[idx]
                if ck not in self.losses['checkpoints']:
                    self.losses['checkpoints'][ck] = {}
                for k, v in stats[idx].items():
                    k = k if k[:7] != 'losses/' else k[7:]
                    if k not in self.losses['current']:
                        self.losses['current'][k] = torch.zeros(N)
                    self.losses['current'][k][j] = v
                    if k not in self.losses['checkpoints'][ck]:
                        self.losses['checkpoints'][ck][k] = torch.zeros(N)
                    self.losses['checkpoints'][ck][k][j] = v

                if ck not in self.parameters['checkpoints']:
                    self.parameters['checkpoints'][ck] = {}
                for k, v in parameters[idx].items():
                    v = init_tensor(v)
                    if k not in self.parameters['current']:
                        if v.ndim == 0:
                            self.parameters['current'][k] = torch.zeros(N)
                        else:
                            self.parameters['current'][k] = torch.zeros(N, len(v))
                    self.parameters['current'][k][j] = v
                    if k not in self.parameters['checkpoints'][ck]:
                        if v.ndim == 0:
                            self.parameters['checkpoints'][ck][k] = torch.zeros(N)
                        else:
                            self.parameters['checkpoints'][ck][k] = torch.zeros(N, len(v))
                    self.parameters['checkpoints'][ck][k][j] = v
            self.save()

        logger.info('Training complete.')

    @torch.no_grad()
    def calculate_current_losses(
            self,
            save_annotations: bool = True,
            save_renders: bool = True,
            initial_or_current: str = 'current'
    ):
        """
        Calculate the current losses.
        """
        if initial_or_current == 'current':
            logger.info(f'Calculating current losses.')
            step = self.refiner.step
            losses = self.losses['current']
            X_preds = self.X_preds['current']
            X_targets_annotated = self.X_targets_annotated

        else:
            logger.info(f'Calculating initial prediction losses.')
            step = 0
            losses = self.losses_init
            X_preds = self.X_preds_initial
            X_targets_annotated = self.X_targets_annotated_initial

        # Disable the cropping so the renders are generated at full size
        crop_renders = self.refiner_args.crop_render
        self.refiner_args.crop_render = False

        # Process the entire dataset in batches
        bs = min(len(self), self.runtime_args.n_refiner_workers * 4)
        n_batches = math.ceil(len(self) / bs)
        log_freq = max(1, int(n_batches / 4))
        for i in range(n_batches):
            if (i + 1) % log_freq == 0:
                logger.info(f'Calculating {initial_or_current} losses for image batch {i + 1}/{n_batches}.')

            # Load the batch
            batch_idxs = list(range(i * bs, min((i + 1) * bs, len(self))))
            p_vec_batch = torch.stack([
                self.get_parameter_vector(idx) for idx in batch_idxs
            ]).to(self.device)

            # Load the target images and keypoints
            X_target = [self.X_targets[idx] for idx in batch_idxs]
            X_target_denoised = [self.X_targets_denoised[idx] for idx in batch_idxs]
            X_target_wis = torch.stack([to_tensor(Image.open(X)) for X in X_target]).permute(0, 2, 3, 1)
            X_target_denoised_wis = torch.stack([to_tensor(Image.open(X)) for X in X_target_denoised]) \
                .permute(0, 2, 3, 1)
            keypoints = [self.keypoints[idx]['keypoints'] if self.keypoints is not None else None for idx in batch_idxs]

            # Load the target distances and 3D vertices
            distances_target = torch.stack([
                init_tensor(self.ds.data[idx]['rendering_parameters']['crystal']['distances'])
                * init_tensor(self.ds.data[idx]['rendering_parameters']['crystal']['scale']) for idx in batch_idxs
            ])
            vertices_target = [init_tensor(self.ds.data[idx]['vertices']) for idx in batch_idxs]

            # Calculate losses and stats
            losses_batch, stats_batch, _ = self.refiner_pool.calculate_losses(
                step=step,
                refiner_args=self.refiner_args,
                p_vec_batch=p_vec_batch,
                X_target=X_target,
                X_target_denoised=X_target_denoised,
                X_target_wis=X_target_wis,
                X_target_denoised_wis=X_target_denoised_wis,
                keypoints=keypoints,
                edges=[None for _ in batch_idxs],
                distances_target=distances_target,
                vertices_target=vertices_target,
                calculate_grads=False,
                save_annotations=save_annotations,
                save_edge_annotations=False,
                save_renders=save_renders,
                X_preds_paths=[X_preds[idx] for idx in batch_idxs],
                X_targets_paths=X_target,
                X_targets_annotated_paths=[X_targets_annotated[idx] for idx in batch_idxs],
                edges_fullsize_paths=None,
                edges_annotated_paths=None,
            )

            # Update scene and crystal parameters
            for j, idx in enumerate(batch_idxs):
                losses['total'][idx] = losses_batch[j].item()
                for k, v in stats_batch.items():
                    k = k if k[:7] != 'losses/' else k[7:]
                    if k not in losses:
                        losses[k] = torch.zeros(len(self))
                    losses[k][idx] = float(v[j])

        # Restore the cropping setting
        self.refiner_args.crop_render = crop_renders

        # Log the total dataset loss
        if initial_or_current == 'current':
            ds_loss = losses['current'].sum().item()
            self.losses['log']['eval'] = torch.concatenate([self.losses['log']['eval'],
                                                            torch.tensor(ds_loss)[None, ...]])
            self.tb_logger.add_scalar('losses/ds_loss', ds_loss, step)
            for k in stats_batch.keys():
                if 'losses' not in k:
                    continue
                kk = k.split('/')[1]
                ds_loss_k = self.losses['eval'][kk]
                ds_loss_k = (ds_loss_k[ds_loss_k > 0]).mean().item()
                self.tb_logger.add_scalar(f'stats_eval/{kk}', ds_loss_k, step)

    def get_parameter_vector(self, idx: int) -> Tensor:
        """
        Return a vector of the parameters for the frame at the given index.
        """
        scene_dict = self.scenes[idx]['scene_dict']
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
        for k, v in self.parameters['current'].items():
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
        Save the data.
        """
        logger.info('Saving parameters, losses, frame counts and scenes.')
        for name, data in zip(['losses', 'parameters'],
                              [self.losses, self.parameters]):
            with open(self.path / f'{name}.json', 'w') as f:
                json.dump(data, f, cls=FlexibleJSONEncoder)

        # Update the scene files with the parameters
        for i, (idx, scene) in enumerate(zip(self.ds_idxs, self.scenes)):
            scene_dict = scene['scene_dict']
            for k in PARAMETER_KEYS:
                v = self.parameters['current'][k][i].tolist()
                if k in scene_dict:
                    scene_dict[k] = v
                elif k in scene_dict['crystal']:
                    scene_dict['crystal'][k] = v
                else:
                    raise RuntimeError(f'Parameter {k} not found in scene or crystal.')
            with open(scene['path'], 'w') as f:
                yaml.dump(scene_dict, f)

    def __len__(self) -> int:
        return len(self.ds_idxs)
