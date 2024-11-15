from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import List, Optional

from crystalsizer3d import DATA_PATH
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.util.utils import str2bool

DENOISER_ARG_NAMES = [
    'denoiser_model_path', 'denoiser_n_tiles', 'denoiser_tile_overlap',
    'denoiser_oversize_input', 'denoiser_max_img_size', 'denoiser_batch_size',
]

KEYPOINTS_ARG_NAMES = [
    'keypoints_model_path', 'keypoints_oversize_input', 'keypoints_max_img_size', 'keypoints_batch_size',
    'keypoints_min_distance', 'keypoints_threshold', 'keypoints_exclude_border', 'keypoints_blur_kernel_relative_size',
    'keypoints_n_patches', 'keypoints_patch_size', 'keypoints_patch_search_res', 'keypoints_attenuation_sigma',
    'keypoints_max_attenuation_factor', 'keypoints_low_res_catchment_distance', 'keypoints_loss_type'
]

PREDICTOR_ARG_NAMES = [
    'predictor_model_path', 'initial_pred_noise_min', 'initial_pred_noise_max', 'initial_pred_oversize_input',
    'initial_pred_max_img_size', 'multiscale', 'use_keypoints', 'n_patches', 'w_img_l1', 'w_img_l2', 'w_perceptual',
    'w_latent', 'w_rcf', 'w_overshoot', 'w_symmetry', 'w_z_pos', 'w_rotation_xy', 'w_patches', 'w_fullsize',
    'w_switch_probs', 'w_keypoints', 'w_anchors', 'l_decay_l1', 'l_decay_l2', 'l_decay_perceptual', 'l_decay_latent',
    'l_decay_rcf', 'perceptual_model', 'latents_model', 'mv2_config_path', 'mv2_checkpoint_path', 'rcf_model_path',
    'rcf_loss_type', 'keypoints_loss_type'
]


class RefinerArgs(BaseArgs):
    def __init__(
            self,
            predictor_model_path: Optional[Path] = None,
            image_path: Optional[Path] = None,
            ds_idx: int = 0,

            # Denoising settings
            denoiser_model_path: Optional[Path] = None,
            denoiser_n_tiles: int = 4,
            denoiser_tile_overlap: float = 0.1,
            denoiser_oversize_input: bool = True,
            denoiser_max_img_size: int = 512,
            denoiser_batch_size: int = 4,

            # Initial prediction settings
            initial_pred_batch_size: int = 16,
            initial_pred_noise_min: float = 0.0,
            initial_pred_noise_max: float = 0.2,
            initial_pred_oversize_input: bool = True,
            initial_pred_max_img_size: int = 512,

            # Keypoint detection settings
            keypoints_model_path: Optional[Path] = None,
            keypoints_oversize_input: bool = False,
            keypoints_max_img_size: int = 1024,
            keypoints_batch_size: int = 4,
            keypoints_min_distance: int = 5,
            keypoints_threshold: float = 0.5,
            keypoints_exclude_border: float = 0.05,
            keypoints_blur_kernel_relative_size: float = 0.01,
            keypoints_n_patches: int = 16,
            keypoints_patch_size: int = 700,
            keypoints_patch_search_res: int = 256,
            keypoints_attenuation_sigma: float = 0.5,
            keypoints_max_attenuation_factor: float = 1.5,
            keypoints_low_res_catchment_distance: int = 100,
            keypoints_loss_type: str = 'mindists',

            # Refining settings
            use_inverse_rendering: bool = True,
            use_perceptual_model: bool = True,
            use_latents_model: bool = True,
            use_rcf_model: bool = True,
            use_keypoints: bool = False,

            # Rendering settings
            ir_wait_n_steps: int = 0,
            ir_loss_placeholder: float = 0,
            rendering_size: int = 200,
            spp: int = 64,
            integrator_max_depth: int = 16,
            integrator_rr_depth: int = 5,

            # Patches settings
            n_patches: int = 0,
            patch_size: int = 64,

            # Optimisation settings
            seed: Optional[int] = None,
            max_steps: int = 1000,
            multiscale: bool = True,
            acc_grad_steps: int = 1,
            clip_grad_norm: float = 0.0,
            opt_algorithm: str = 'sgd',

            # Convergence detector settings
            convergence_tau_fast: int = 20,
            convergence_tau_slow: int = 100,
            convergence_threshold: float = 0.05,
            convergence_patience: int = 100,

            # Noise
            image_noise_std: float = 0.0,
            distances_noise: float = 0.01,
            rotation_noise: float = 0.02,
            material_roughness_noise: float = 0.01,
            material_ior_noise: float = 0.1,
            radiance_noise: float = 0.01,

            # Conjugate face switching
            use_conj_switching: bool = False,
            conj_switch_prob_init: float = 0.1,
            conj_switch_prob_min: float = 1e-2,
            conj_switch_prob_max: float = 1 - 1e-2,

            # Learning rates
            lr_scale: float = 0.,
            lr_distances: float = 1e-4,
            lr_origin: float = 1e-4,
            lr_rotation: float = 1e-4,
            lr_material: float = 1e-4,
            lr_light: float = 1e-4,
            lr_switches: float = 1e-4,

            # Learning rate scheduler
            lr_scheduler: str = 'cosine',
            lr_min: float = 1e-6,
            lr_warmup: float = 1e-5,
            lr_warmup_steps: int = 0,
            lr_decay_steps: int = 30,
            lr_decay_milestones: List[int] = [],
            lr_cooldown_steps: int = 0,
            lr_patience_steps: int = 10,
            lr_decay_rate: float = 0.1,
            lr_cycle_mul: float = 1.0,
            lr_cycle_decay: float = 0.1,
            lr_cycle_limit: int = 1,
            lr_k_decay: float = 1.0,

            # Loss weightings
            w_img_l1: float = 1.0,
            w_img_l2: float = 1.0,
            w_perceptual: float = 1.0,
            w_latent: float = 1.0,
            w_rcf: float = 1.0,
            w_overshoot: float = 1.0,
            w_symmetry: float = 1.0,
            w_z_pos: float = 1.0,
            w_rotation_xy: float = 1.0,
            w_patches: float = 1.0,
            w_fullsize: float = 1.0,
            w_switch_probs: float = 1.0,
            w_temporal: float = 1.0,
            w_keypoints: float = 1.0,
            w_anchors: float = 1.0,
            w_edge_matching: float = 1.0,

            # Loss decay factors
            l_decay_l1: float = 1.0,
            l_decay_l2: float = 1.0,
            l_decay_perceptual: float = 1.0,
            l_decay_latent: float = 1.0,
            l_decay_rcf: float = 1.0,

            # Helper models
            perceptual_model: Optional[str] = None,
            latents_model: Optional[str] = None,
            latents_input_size: int = 0,
            mv2_config_path: Optional[Path] = DATA_PATH / 'MAGVIT2' / 'imagenet_lfqgan_256_B.yaml',
            mv2_checkpoint_path: Optional[Path] = DATA_PATH / 'MAGVIT2' / 'imagenet_256_B.ckpt',
            rcf_model_path: Optional[Path] = DATA_PATH / 'bsds500_pascal_model.pth',
            rcf_loss_type: str = 'l2',

            # Runtime args
            log_every_n_steps: int = 1,

            # Plotting args
            plot_every_n_steps: int = 10,
            plot_to_tensorboard: bool = False,
            plot_n_samples: int = 2,
            plot_n_patches: int = -1,
            plot_rcf_feats: List[int] = [0, 5],

            img_size_3d: int = 400,
            wireframe_r_factor: float = 0.2,
            surface_colour_target: str = 'orange',
            wireframe_colour_target: str = 'darkorange',
            surface_colour_pred: str = 'skyblue',
            wireframe_colour_pred: str = 'cornflowerblue',
            azim: float = 0,
            elev: float = 0,
            roll: float = 0,
            distance: float = 7,

            plot_colour_target: str = 'red',
            plot_colour_pred: str = 'darkblue',

            **kwargs
    ):
        # Ensure all paths are Paths
        if isinstance(predictor_model_path, str):
            predictor_model_path = Path(predictor_model_path)
        if isinstance(image_path, str):
            image_path = Path(image_path)
        if isinstance(denoiser_model_path, str):
            denoiser_model_path = Path(denoiser_model_path)
        if isinstance(keypoints_model_path, str):
            keypoints_model_path = Path(keypoints_model_path)
        if isinstance(mv2_config_path, str):
            mv2_config_path = Path(mv2_config_path)
        if isinstance(mv2_checkpoint_path, str):
            mv2_checkpoint_path = Path(mv2_checkpoint_path)
        if isinstance(rcf_model_path, str):
            rcf_model_path = Path(rcf_model_path)

        # Predictor model and target image
        if predictor_model_path is not None:
            assert predictor_model_path.exists(), f'Predictor model path does not exist: {predictor_model_path}'
            assert predictor_model_path.suffix == '.json', f'Predictor model path must be a json file: {predictor_model_path}'
        self.predictor_model_path = predictor_model_path
        if image_path is not None:
            assert image_path.exists(), f'Image path does not exist: {image_path}'
        self.image_path = image_path
        self.ds_idx = ds_idx

        # Denoising settings
        if denoiser_model_path is not None:
            assert denoiser_model_path.exists(), f'DN model path does not exist: {denoiser_model_path}'
        self.denoiser_model_path = denoiser_model_path
        self.denoiser_n_tiles = denoiser_n_tiles
        self.denoiser_tile_overlap = denoiser_tile_overlap
        self.denoiser_oversize_input = denoiser_oversize_input
        self.denoiser_max_img_size = denoiser_max_img_size
        self.denoiser_batch_size = denoiser_batch_size

        # Initial prediction settings
        self.initial_pred_batch_size = initial_pred_batch_size
        self.initial_pred_noise_min = initial_pred_noise_min
        self.initial_pred_noise_max = initial_pred_noise_max
        self.initial_pred_oversize_input = initial_pred_oversize_input
        self.initial_pred_max_img_size = initial_pred_max_img_size

        # Keypoint detection settings
        if keypoints_model_path is not None:
            assert keypoints_model_path.exists(), f'Keypoints model path does not exist: {keypoints_model_path}'
        self.keypoints_model_path = keypoints_model_path
        self.keypoints_oversize_input = keypoints_oversize_input
        self.keypoints_max_img_size = keypoints_max_img_size
        self.keypoints_batch_size = keypoints_batch_size
        self.keypoints_min_distance = keypoints_min_distance
        self.keypoints_threshold = keypoints_threshold
        self.keypoints_exclude_border = keypoints_exclude_border
        self.keypoints_blur_kernel_relative_size = keypoints_blur_kernel_relative_size
        self.keypoints_n_patches = keypoints_n_patches
        self.keypoints_patch_size = keypoints_patch_size
        self.keypoints_patch_search_res = keypoints_patch_search_res
        self.keypoints_attenuation_sigma = keypoints_attenuation_sigma
        self.keypoints_max_attenuation_factor = keypoints_max_attenuation_factor
        self.keypoints_low_res_catchment_distance = keypoints_low_res_catchment_distance
        self.keypoints_loss_type = keypoints_loss_type

        # Refining settings
        self.use_inverse_rendering = use_inverse_rendering
        self.use_perceptual_model = use_inverse_rendering & use_perceptual_model
        self.use_latents_model = use_inverse_rendering & use_latents_model
        self.use_rcf_model = use_inverse_rendering & use_rcf_model
        self.use_keypoints = use_keypoints

        # Rendering settings
        self.ir_wait_n_steps = ir_wait_n_steps
        self.ir_loss_placeholder = ir_loss_placeholder
        self.rendering_size = rendering_size
        self.spp = spp
        self.integrator_max_depth = integrator_max_depth
        self.integrator_rr_depth = integrator_rr_depth

        # Patches settings
        self.n_patches = n_patches
        self.patch_size = patch_size

        # Optimisation settings
        self.seed = seed
        self.max_steps = max_steps
        self.multiscale = multiscale
        self.acc_grad_steps = acc_grad_steps
        self.clip_grad_norm = clip_grad_norm
        self.opt_algorithm = opt_algorithm

        # Convergence detector settings
        self.convergence_tau_fast = convergence_tau_fast
        self.convergence_tau_slow = convergence_tau_slow
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience

        # Noise
        self.image_noise_std = image_noise_std
        self.distances_noise = distances_noise
        self.rotation_noise = rotation_noise
        self.material_roughness_noise = material_roughness_noise
        self.material_ior_noise = material_ior_noise
        self.radiance_noise = radiance_noise

        # Conjugate face switching
        self.use_conj_switching = use_conj_switching
        assert 0 <= conj_switch_prob_init <= 1, 'Initial probability of switching conjugate faces must be in [0, 1].'
        self.conj_switch_prob_init = conj_switch_prob_init
        assert 0 <= conj_switch_prob_min <= 1, 'Minimum probability of switching conjugate faces must be in [0, 1].'
        self.conj_switch_prob_min = conj_switch_prob_min
        assert 0 <= conj_switch_prob_max <= 1, 'Maximum probability of switching conjugate faces must be in [0, 1].'
        assert conj_switch_prob_min < conj_switch_prob_max, 'Minimum probability must be less than maximum probability.'
        self.conj_switch_prob_max = conj_switch_prob_max

        # Learning rates
        self.lr_scale = lr_scale
        self.lr_distances = lr_distances
        self.lr_origin = lr_origin
        self.lr_rotation = lr_rotation
        self.lr_material = lr_material
        self.lr_light = lr_light
        self.lr_switches = lr_switches

        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.lr_warmup = lr_warmup
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_cooldown_steps = lr_cooldown_steps
        self.lr_patience_steps = lr_patience_steps
        self.lr_decay_rate = lr_decay_rate
        self.lr_cycle_mul = lr_cycle_mul
        self.lr_cycle_decay = lr_cycle_decay
        self.lr_cycle_limit = lr_cycle_limit
        self.lr_k_decay = lr_k_decay

        # Loss weightings
        self.w_img_l1 = w_img_l1
        self.w_img_l2 = w_img_l2
        self.w_perceptual = w_perceptual
        self.w_latent = w_latent
        self.w_rcf = w_rcf
        self.w_overshoot = w_overshoot
        self.w_symmetry = w_symmetry
        self.w_z_pos = w_z_pos
        self.w_rotation_xy = w_rotation_xy
        self.w_patches = w_patches
        self.w_fullsize = w_fullsize
        self.w_switch_probs = w_switch_probs
        self.w_temporal = w_temporal
        self.w_keypoints = w_keypoints
        self.w_anchors = w_anchors
        self.w_edge_matching = w_edge_matching
        
        # Loss decay factors
        self.l_decay_l1 = l_decay_l1
        self.l_decay_l2 = l_decay_l2
        self.l_decay_perceptual = l_decay_perceptual
        self.l_decay_latent = l_decay_latent
        self.l_decay_rcf = l_decay_rcf

        # Helper models
        self.perceptual_model = perceptual_model
        self.latents_model = latents_model
        self.latents_input_size = latents_input_size
        self.mv2_config_path = mv2_config_path
        self.mv2_checkpoint_path = mv2_checkpoint_path
        self.rcf_model_path = rcf_model_path
        self.rcf_loss_type = rcf_loss_type

        # Runtime args
        self.log_every_n_steps = log_every_n_steps

        # Plotting args
        self.plot_every_n_steps = plot_every_n_steps
        self.plot_to_tensorboard = plot_to_tensorboard
        self.plot_n_samples = plot_n_samples
        self.plot_n_patches = plot_n_patches
        self.plot_rcf_feats = plot_rcf_feats

        # Digital crystal image
        self.img_size_3d = img_size_3d
        self.wireframe_r_factor = wireframe_r_factor
        self.surface_colour_target = surface_colour_target
        self.wireframe_colour_target = wireframe_colour_target
        self.surface_colour_pred = surface_colour_pred
        self.wireframe_colour_pred = wireframe_colour_pred
        self.azim = azim
        self.elev = elev
        self.roll = roll
        self.distance = distance

        # Parameter plots
        self.plot_colour_target = plot_colour_target
        self.plot_colour_pred = plot_colour_pred

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Refiner Args')
        group.add_argument('--predictor-model-path', type=Path,
                           help='Path to the model\'s json file.')
        group.add_argument('--image-path', type=Path,
                           help='Path to the image to process. If set, will override the dataset entry.')
        group.add_argument('--ds-idx', type=int, default=0,
                           help='Index of the dataset entry to use.')

        # Denoising settings
        group.add_argument('--denoiser-model-path', type=Path,
                           help='Path to the denoising model\'s json file.')
        group.add_argument('--denoiser-n-tiles', type=int, default=9,
                           help='Number of tiles to split the image into for denoising.')
        group.add_argument('--denoiser-tile-overlap', type=float, default=0.05,
                           help='Ratio of overlap between tiles for denoising.')
        group.add_argument('--denoiser-oversize-input', type=str2bool, default=True,
                           help='Whether to resize the input images to the training dataset image size.')
        group.add_argument('--denoiser-max-img-size', type=int, default=512,
                           help='Maximum image size for denoising.')
        group.add_argument('--denoiser-batch-size', type=int, default=3,
                           help='Number of tiles to denoise at a time.')

        # Initial prediction settings
        group.add_argument('--initial-pred-batch-size', type=int, default=8,
                           help='Batch size for the initial prediction.')
        group.add_argument('--initial-pred-noise-min', type=float, default=0.0,
                           help='Minimum amount of noise to add to the batch of images used to generate the initial prediction.')
        group.add_argument('--initial-pred-noise-max', type=float, default=0.1,
                           help='Minimum amount of noise to add to the batch of images used to generate the initial prediction.')
        group.add_argument('--initial-pred-oversize-input', type=str2bool, default=True,
                           help='Whether to resize the input images to the training dataset image size.')
        group.add_argument('--initial-pred-max-img-size', type=int, default=512,
                           help='Maximum image size for initial prediction.')

        # Keypoint detection settings
        group.add_argument('--keypoints-model-path', type=Path,
                           help='Path to the keypoints model checkpoint.')
        group.add_argument('--keypoints-oversize-input', type=str2bool, default=False,
                           help='Whether to resize the input images to the training dataset image size.')
        group.add_argument('--keypoints-max-img-size', type=int, default=1024,
                           help='Maximum image size for keypoint detection.')
        group.add_argument('--keypoints-batch-size', type=int, default=3,
                           help='Number of tiles to detect keypoints at a time.')
        group.add_argument('--keypoints-min-distance', type=int, default=5,
                           help='Minimum pixel distance between keypoints.')
        group.add_argument('--keypoints-threshold', type=float, default=0.5,
                           help='Threshold for keypoints detection.')
        group.add_argument('--keypoints-exclude-border', type=float, default=0.05,
                           help='Exclude keypoints within this ratio of the border.')
        group.add_argument('--keypoints-blur-kernel-relative-size', type=float, default=0.01,
                           help='Relative size of the blur kernel for initial keypoints detection from low res, denoised image.')
        group.add_argument('--keypoints-n-patches', type=int, default=16,
                           help='Number of patches to crop from the image for high res keypoints detection.')
        group.add_argument('--keypoints-patch-size', type=int, default=700,
                           help='Size of the crop patches.')
        group.add_argument('--keypoints-patch-search-res', type=int, default=256,
                           help='Resolution of the low-res keypoints heatmap to use for determining where to crop the patches.')
        group.add_argument('--keypoints-attenuation-sigma', type=float, default=0.5,
                           help='Sigma parameter for the Gaussian blob used to iteratively attenuate the keypoints heatmap.')
        group.add_argument('--keypoints-max-attenuation-factor', type=float, default=1.5,
                           help='Maximum Gaussian peak height used for the attenuation function.')
        group.add_argument('--keypoints-low-res-catchment-distance', type=int, default=100,
                           help='Catchment distance (in pixels) for high res keypoints from the original low res keypoints.')
        group.add_argument('--keypoints-loss-type', type=str, default='mindists',
                           choices=['mindists', 'sinkhorn', 'hausdorff'],
                           help='Type of loss to use for keypoints refinement.')

        # Refining settings
        group.add_argument('--use-inverse-rendering', type=str2bool, default=True,
                           help='Use inverse rendering.')
        group.add_argument('--use-perceptual-model', type=str2bool, default=True,
                           help='Use a perceptual model. Requires inverse rendering.')
        group.add_argument('--use-latents-model', type=str2bool, default=True,
                           help='Use a latents model. Requires inverse rendering.')
        group.add_argument('--use-rcf-model', type=str2bool, default=True,
                           help='Use the RCF model. Requires inverse rendering.')
        group.add_argument('--use-keypoints', type=str2bool, default=False,
                           help='Use the keypoints detection method.')

        # Rendering settings
        group.add_argument('--ir-wait-n-steps', type=int, default=0,
                           help='Number of optimisation steps to wait before turning inverse rendering on '
                                '(if --use-inverse-rendering=True).')
        group.add_argument('--ir-loss-placeholder', type=float, default=0,
                           help='Placeholder loss value for inverse rendering.')
        group.add_argument('--rendering-size', type=int, default=300,
                           help='Resolution of the (square) rendered image in pixels. '
                                'Larger images are slower to render and use more resources, but may be more accurate.')
        group.add_argument('--spp', type=int, default=32,
                           help='Ray-tracing samples per pixel. Affects the sharpness and clarity of the rendered image.')
        group.add_argument('--integrator-max-depth', type=int, default=16,
                           help='Maximum depth of the ray-tracing integrator.')
        group.add_argument('--integrator-rr-depth', type=int, default=4,
                           help='Path depth at which rays will begin to use the russian roulette termination criterion.')

        # Patch settings
        group.add_argument('--n-patches', type=int, default=0,
                           help='Number of zooming-in patches to use.')
        group.add_argument('--patch-size', type=int, default=40,
                           help='Size of the patches to use for zooming-in.')

        # Optimisation settings
        group.add_argument('--seed', type=int,
                           help='Seed for the random number generator.')
        group.add_argument('--max-steps', type=int, default=5000,
                           help='Maximum number of refinement steps.')
        group.add_argument('--multiscale', type=str2bool, default=False,
                           help='Use multiscale rendering for the optimisation.')
        group.add_argument('--acc-grad-steps', type=int, default=10,
                           help='Number of gradient accumulation steps.')
        group.add_argument('--clip-grad-norm', type=float, default=10,
                           help='Clip the gradient norm of the distances to this value.')
        group.add_argument('--opt-algorithm', type=str, default='adabelief',
                           help='Optimisation algorithm to use.')

        # Convergence detector settings
        group.add_argument('--convergence-tau-fast', type=int, default=20,
                           help='Fast convergence detector time constant.')
        group.add_argument('--convergence-tau-slow', type=int, default=100,
                           help='Slow convergence detector time constant.')
        group.add_argument('--convergence-threshold', type=float, default=0.05,
                           help='Convergence threshold.')
        group.add_argument('--convergence-patience', type=int, default=100,
                           help='Convergence patience.')

        # Noise
        group.add_argument('--image-noise-std', type=float, default=0.0,
                           help='Standard deviation of the zero-mean Gaussian noise to add to the image.')
        group.add_argument('--distances-noise', type=float, default=0.0,
                           help='Standard deviation of the zero-mean Gaussian noise to add to the distances.')
        group.add_argument('--rotation-noise', type=float, default=0.0,
                           help='Standard deviation of the zero-mean Gaussian noise to add to the rotation.')
        group.add_argument('--material-roughness-noise', type=float, default=0.0,
                           help='Standard deviation of the zero-mean Gaussian noise to add to the material roughness.')
        group.add_argument('--material-ior-noise', type=float, default=0.0,
                           help='Standard deviation of the zero-mean Gaussian noise to add to the material IOR.')
        group.add_argument('--radiance-noise', type=float, default=0.,
                           help='Standard deviation of the zero-mean Gaussian noise to add to the light radiance.')

        # Conjugate face switching
        group.add_argument('--use-conj-switching', type=str2bool, default=True,
                           help='Stochastically switch the distances of two conjugate faces during the refinement process. '
                                'Can sometimes help to get out of local minima.')
        group.add_argument('--conj-switch-prob-init', type=float, default=0.4,
                           help='Initial probability of switching a pair of conjugate faces.')
        group.add_argument('--conj-switch-prob-min', type=float, default=1e-2,
                           help='Minimum probability of switching a pair of conjugate faces.')
        group.add_argument('--conj-switch-prob-max', type=float, default=1 - 1e-2,
                           help='Maximum probability of switching a pair of conjugate faces.')

        # Learning rates
        group.add_argument('--lr-scale', type=float, default=0.,
                           help='Learning rate for the scale parameter.')
        group.add_argument('--lr-distances', type=float, default=5e-3,
                           help='Learning rate for the crystal face distances.')
        group.add_argument('--lr-origin', type=float, default=5e-3,
                           help='Learning rate for the origin position.')
        group.add_argument('--lr-rotation', type=float, default=5e-3,
                           help='Learning rate for the rotation.')
        group.add_argument('--lr-material', type=float, default=1e-3,
                           help='Learning rate for the material properties (IOR/roughness).')
        group.add_argument('--lr-light', type=float, default=5e-3,
                           help='Learning rate for the light.')
        group.add_argument('--lr-switches', type=float, default=1e-1,
                           help='Learning rate for the conjugate switching probabilities.')

        # Learning rate scheduler
        group.add_argument('--lr-scheduler', type=str, default='none',
                           help='Learning rate scheduler.')
        group.add_argument('--lr-min', type=float, default=1e-3,
                           help='Minimum learning rate.')
        group.add_argument('--lr-warmup', type=float, default=1e-3,
                           help='Learning rate used during the warmup steps.')
        group.add_argument('--lr-warmup-steps', type=int, default=1,
                           help='Number of warmup steps.')
        group.add_argument('--lr-decay-steps', type=int, default=30,
                           help='Number of steps before learning rate decay.')
        group.add_argument('--lr-decay-milestones', type=lambda s: [int(item) for item in s.split(',')], default=[],
                           help='Steps at which to decay learning rate.')
        group.add_argument('--lr-cooldown-steps', type=int, default=0,
                           help='Number of steps before learning rate cooldown.')
        group.add_argument('--lr-patience-steps', type=int, default=10,
                           help='Number of steps before learning rate patience.')
        group.add_argument('--lr-decay-rate', type=float, default=0.1,
                           help='Learning rate decay rate.')
        group.add_argument('--lr-cycle-mul', type=float, default=2.,
                           help='Learning rate cycle multiplier.')
        group.add_argument('--lr-cycle-decay', type=float, default=0.9,
                           help='Learning rate cycle decay.')
        group.add_argument('--lr-cycle-limit', type=int, default=1,
                           help='Learning rate cycle limit.')
        group.add_argument('--lr-k-decay', type=float, default=1.0,
                           help='Learning rate k decay.')

        # Loss weightings
        group.add_argument('--w-img-l1', type=float, default=10.,
                           help='Weight of the L1 image loss. Requires inverse rendering.')
        group.add_argument('--w-img-l2', type=float, default=0.,
                           help='Weight of the L2 image loss. Requires inverse rendering.')
        group.add_argument('--w-perceptual', type=float, default=1e-2,
                           help='Weight of the perceptual loss. Uses a pre-trained neural network to extract '
                                'perceptual features of the target and rendered images and minimises the loss between the two. '
                                'Requires inverse rendering. ')
        group.add_argument('--w-latent', type=float, default=1e-1,
                           help='Weight of the latent encoding loss. Uses a pre-trained neural network to extract '
                                'latent encodings of the target and rendered images and minimises the loss between the two. '
                                'Requires inverse rendering.')
        group.add_argument('--w-rcf', type=float, default=0.,
                           help='Weight of the RCF loss. Uses a pre-trained neural network to extract '
                                'edge features of the target and rendered images and minimises the loss between the two. '
                                'Requires inverse rendering.')
        group.add_argument('--w-overshoot', type=float, default=10.0,
                           help='Weight of the overshoot loss.'
                                'Minimises the face distances that correspond to grown-out/zero-area faces.')
        group.add_argument('--w-symmetry', type=float, default=0.1,
                           help='Weight of the symmetry loss. '
                                'Encourages pairs of faces to have similar distances from the origin.')
        group.add_argument('--w-z-pos', type=float, default=10.0,
                           help='Weight of the z position loss.'
                                'Encourages the lowest z-coordinate of the crystal to be at z=0.')
        group.add_argument('--w-rotation-xy', type=float, default=1.0,
                           help='Weight of the rotation xy loss.'
                                'Encourages the crystal to be aligned with the xy plane.')
        group.add_argument('--w-patches', type=float, default=0.,
                           help='Weight of the patch loss. Only used when using patches.')
        group.add_argument('--w-fullsize', type=float, default=1.,
                           help='Weight of the combined losses on the full sized image. Only used when using patches.')
        group.add_argument('--w-switch-probs', type=float, default=0.1,
                           help='Weight of the conjugate face switching probabilities loss regulariser term.')
        group.add_argument('--w-temporal', type=float, default=1.,
                           help='Weight of the temporal regularisation term, used to penalise changes from a previous solution.')
        group.add_argument('--w-keypoints', type=float, default=1.,
                           help='Weight of the keypoints loss. Only used when using keypoints model.')
        group.add_argument('--w-anchors', type=float, default=1.,
                           help='Weight of the manually-defined anchors loss.')
        group.add_argument('--w-edge-matching', type=float, default=1.,
                           help='Weight of the manually-defined edge matching loss.')

        # Loss decay factors
        group.add_argument('--l-decay-l1', type=float, default=0.5,
                           help='Decay factor for the multiscale L1 image loss.')
        group.add_argument('--l-decay-l2', type=float, default=1.0,
                           help='Decay factor for the multiscale L2 image loss.')
        group.add_argument('--l-decay-perceptual', type=float, default=1.,
                           help='Decay factor for the perceptual losses from each layer.')
        group.add_argument('--l-decay-latent', type=float, default=1.,
                           help='Decay factor for the latent losses from each layer.')
        group.add_argument('--l-decay-rcf', type=float, default=0.2,
                           help='Decay factor for the RCF losses from each layer.')

        # Helper models
        group.add_argument('--perceptual-model', type=str, default='timm/regnetz_d8.ra3_in1k',
                           help='Perceptual model to use. Must start with "timm/".')
        group.add_argument('--latents-model', type=str, default='MAGVIT2',
                           help='Latent encoder model to use. Only MAGVIT2 supported.')
        group.add_argument('--latents-input-size', type=int, default=0,
                           help='Size of the input images to the latents model. '
                                '0: resize to training image size (256). -1: use rendered image size.')
        group.add_argument('--mv2-config-path', type=Path,
                           default=DATA_PATH / 'MAGVIT2' / 'imagenet_lfqgan_256_B.yaml',
                           help='Path to the MAGVIT2 config file to use.')
        group.add_argument('--mv2-checkpoint-path', type=Path, default=DATA_PATH / 'MAGVIT2' / 'imagenet_256_B.ckpt',
                           help='Path to the MAGVIT2 model checkpoint to use.')
        group.add_argument('--rcf-model-path', type=Path, default=DATA_PATH / 'bsds500_pascal_model.pth',
                           help='Path to the RCF model checkpoint.')
        group.add_argument('--rcf-loss-type', type=str, default='l2', choices=['l1', 'l2'],
                           help='Loss function to use for the RCF features comparison.')

        # Runtime args
        group.add_argument('--log-every-n-steps', type=int, default=1,
                           help='Log every n batches.')

        # Plotting args
        group.add_argument('--plot-every-n-steps', type=int, default=10,
                           help='Plot every n batches.')
        group.add_argument('--plot-to-tensorboard', type=str2bool, default=False,
                           help='Save plots to tensorboard (as well as to disk).')
        group.add_argument('--plot-n-samples', type=int, default=2,
                           help='Number of multi-scale or probabilistic samples to plot.')
        group.add_argument('--plot-n-patches', type=int, default=-1,
                           help='Number of patches to plot. Set to -1 to plot all patches.')
        group.add_argument('--plot-rcf-feats', type=lambda s: [int(item) for item in s.split(',')], default=[0, 5],
                           help='RCF features to plot.')

        # Digital crystal image
        group.add_argument('--img-size-3d', type=int, default=400,
                           help='Size of the 3D digital crystal image.')
        group.add_argument('--wireframe-r-factor', type=float, default=0.2,
                           help='Wireframe radius factor, multiplied by the maximum dimension of the bounding box to calculate the final edge tube radius.')
        group.add_argument('--surface-colour-target', type=str, default='orange',
                           help='Target mesh surface colour.')
        group.add_argument('--wireframe-colour-target', type=str, default='darkorange',
                           help='Target mesh wireframe colour.')
        group.add_argument('--surface-colour-pred', type=str, default='skyblue',
                           help='Predicted mesh surface colour.')
        group.add_argument('--wireframe-colour-pred', type=str, default='cornflowerblue',
                           help='Predicted mesh wireframe colour.')
        group.add_argument('--azim', type=float, default=50,
                           help='Azimuthal angle of the camera.')
        group.add_argument('--elev', type=float, default=50,
                           help='Elevation angle of the camera.')
        group.add_argument('--roll', type=float, default=-120,
                           help='Roll angle of the camera.')
        group.add_argument('--distance', type=float, default=7,
                           help='Camera distance.')

        # Parameter plots
        group.add_argument('--plot-colour-target', type=str, default='darkorange',
                           help='Target parameters plot colour.')
        group.add_argument('--plot-colour-pred', type=str, default='cornflowerblue',
                           help='Predicted parameters plot colour.')

        return group
