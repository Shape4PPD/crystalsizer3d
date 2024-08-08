from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import List, Optional

from crystalsizer3d import DATA_PATH
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.util.utils import str2bool


class RefinerArgs(BaseArgs):
    def __init__(
            self,
            predictor_model_path: Path,
            image_path: Optional[Path] = None,
            ds_idx: int = 0,

            # Denoising settings
            denoiser_model_path: Optional[Path] = None,
            denoiser_n_tiles: int = 4,
            denoiser_tile_overlap: float = 0.1,
            denoiser_batch_size: int = 4,

            # Initial prediction settings
            initial_pred_from: str = 'denoised',
            initial_pred_batch_size: int = 16,
            initial_pred_noise_min: float = 0.0,
            initial_pred_noise_max: float = 0.2,

            # Rendering settings
            working_image_size: int = 200,
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
            w_anchors: float = 1.0,

            # Loss decay factors
            l_decay_l1: float = 1.0,
            l_decay_l2: float = 1.0,
            l_decay_perceptual: float = 1.0,
            l_decay_latent: float = 1.0,
            l_decay_rcf: float = 1.0,

            # Helper models
            perceptual_model: Optional[str] = None,
            latents_model: Optional[str] = None,
            mv2_config_path: Optional[Path] = DATA_PATH / 'MAGVIT2' / 'imagenet_lfqgan_256_B.yaml',
            mv2_checkpoint_path: Optional[Path] = DATA_PATH / 'MAGVIT2' / 'imagenet_256_B.ckpt',
            rcf_model_path: Optional[Path] = DATA_PATH / 'bsds500_pascal_model.pth',
            rcf_loss_type: str = 'l2',

            # Runtime args
            log_every_n_steps: int = 1,
            plot_every_n_steps: int = 10,

            # Plotting args
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
        if isinstance(mv2_config_path, str):
            mv2_config_path = Path(mv2_config_path)
        if isinstance(mv2_checkpoint_path, str):
            mv2_checkpoint_path = Path(mv2_checkpoint_path)
        if isinstance(rcf_model_path, str):
            rcf_model_path = Path(rcf_model_path)

        # Predictor model and target image
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
        self.denoiser_batch_size = denoiser_batch_size

        # Initial prediction settings
        self.initial_pred_from = initial_pred_from
        self.initial_pred_batch_size = initial_pred_batch_size
        self.initial_pred_noise_min = initial_pred_noise_min
        self.initial_pred_noise_max = initial_pred_noise_max

        # Rendering settings
        self.working_image_size = working_image_size
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
        self.w_anchors = w_anchors

        # Loss decay factors
        self.l_decay_l1 = l_decay_l1
        self.l_decay_l2 = l_decay_l2
        self.l_decay_perceptual = l_decay_perceptual
        self.l_decay_latent = l_decay_latent
        self.l_decay_rcf = l_decay_rcf

        # Helper models
        self.perceptual_model = perceptual_model
        # self.perceptual_model = None
        self.latents_model = latents_model
        # self.latents_model = None
        self.mv2_config_path = mv2_config_path
        self.mv2_checkpoint_path = mv2_checkpoint_path
        # self.rcf_model_path = rcf_model_path
        self.rcf_model_path = None
        self.rcf_loss_type = rcf_loss_type

        # Runtime args
        self.log_every_n_steps = log_every_n_steps
        self.plot_every_n_steps = plot_every_n_steps

        # Plotting args
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
        group.add_argument('--predictor-model-path', type=Path, required=True,
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
                           help='Overlap between tiles for denoising.')
        group.add_argument('--denoiser-batch-size', type=int, default=3,
                           help='Batch size for denoising.')

        # Initial prediction settings
        group.add_argument('--initial-pred-from', type=str, default='denoised', choices=['denoised', 'original'],
                           help='Calculate the initial prediction from either the original or denoised image.')
        group.add_argument('--initial-pred-batch-size', type=int, default=8,
                           help='Batch size for the initial prediction.')
        group.add_argument('--initial-pred-noise-min', type=float, default=0.0,
                           help='Minimum noise to add to the initial prediction.')
        group.add_argument('--initial-pred-noise-max', type=float, default=0.1,
                           help='Maximum noise to add to the initial prediction.')

        # Rendering settings
        group.add_argument('--working-image-size', type=int, default=300,
                           help='Size of the working image.')
        group.add_argument('--spp', type=int, default=32,
                           help='Samples per pixel.')
        group.add_argument('--integrator-max-depth', type=int, default=16,
                           help='Maximum depth of the integrator.')
        group.add_argument('--integrator-rr-depth', type=int, default=4,
                           help='Russian roulette depth.')

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

        # Noise
        group.add_argument('--image-noise-std', type=float, default=0.0,
                           help='Standard deviation of the noise to add to the image.')
        group.add_argument('--distances-noise', type=float, default=0.0,
                           help='Standard deviation of the noise to add to the distances.')
        group.add_argument('--rotation-noise', type=float, default=0.0,
                           help='Standard deviation of the noise to add to the rotation.')
        group.add_argument('--material-roughness-noise', type=float, default=0.0,
                           help='Standard deviation of the noise to add to the material roughness.')
        group.add_argument('--material-ior-noise', type=float, default=0.0,
                           help='Standard deviation of the noise to add to the material IOR.')
        group.add_argument('--radiance-noise', type=float, default=0.,
                            help='Standard deviation of the noise to add to the light radiance.')

        # Conjugate face switching
        group.add_argument('--use-conj-switching', type=str2bool, default=True,
                            help='Use conjugate face switching.')
        group.add_argument('--conj-switch-prob-init', type=float, default=0.4,
                            help='Initial probability of switching a face.')
        group.add_argument('--conj-switch-prob-min', type=float, default=1e-2,
                            help='Minimum probability of switching a face.')
        group.add_argument('--conj-switch-prob-max', type=float, default=1 - 1e-2,
                            help='Maximum probability of switching a face.')

        # Learning rates
        group.add_argument('--lr-scale', type=float, default=0.,
                           help='Learning rate for the scale parameter.')
        group.add_argument('--lr-distances', type=float, default=5e-3,
                           help='Learning rate for the distances.')
        group.add_argument('--lr-origin', type=float, default=5e-3,
                           help='Learning rate for the origin.')
        group.add_argument('--lr-rotation', type=float, default=5e-3,
                           help='Learning rate for the rotation.')
        group.add_argument('--lr-material', type=float, default=1e-3,
                           help='Learning rate for the material.')
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
                           help='Warmup learning rate.')
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
                           help='Weight of the L1 image loss.')
        group.add_argument('--w-img-l2', type=float, default=0.,
                           help='Weight of the L2 image loss.')
        group.add_argument('--w-perceptual', type=float, default=1e-2,
                           help='Weight of the perceptual loss.')
        group.add_argument('--w-latent', type=float, default=1e-1,
                           help='Weight of the latent encoding loss.')
        group.add_argument('--w-rcf', type=float, default=0.,
                           help='Weight of the RCF loss.')
        group.add_argument('--w-overshoot', type=float, default=10.0,
                           help='Weight of the overshoot loss.')
        group.add_argument('--w-symmetry', type=float, default=0.1,
                           help='Weight of the symmetry loss.')
        group.add_argument('--w-z-pos', type=float, default=10.0,
                           help='Weight of the z position loss.')
        group.add_argument('--w-rotation-xy', type=float, default=1.0,
                            help='Weight of the rotation xy loss.')
        group.add_argument('--w-patches', type=float, default=0.,
                           help='Weight of the patch loss.')
        group.add_argument('--w-fullsize', type=float, default=1.,
                           help='Weight of the combined losses on the full sized image. Only needed when using patches.')
        group.add_argument('--w-switch-probs', type=float, default=0.1,
                            help='Weight of the conjugate face switching probabilities loss regulariser term.')
        group.add_argument('--w-anchors', type=float, default=1.,
                            help='Weight of the anchors loss.')

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
        group.add_argument('--plot-every-n-steps', type=int, default=10,
                           help='Plot every n batches.')

        # Plotting args
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
