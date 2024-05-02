from pathlib import Path
from typing import List, Optional, Tuple, Union

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from ccdc.io import EntryReader
from timm.optim import create_optimizer_v2
from torch import nn
from torch.optim import LBFGS
from torchmin import Minimizer
from torchvision.transforms import Compose, GaussianBlur

from crystalsizer3d import LOGS_PATH, ROOT_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.nn.models.rcf import RCF
from crystalsizer3d.util.ema import EMA
from crystalsizer3d.util.utils import to_numpy

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('llvm_ad_rgb')
    device = torch.device('cpu')

from mitsuba import ScalarTransform4f as T

# dr.set_log_level(dr.LogLevel.Info)
# dr.set_thread_count(1)

save_plots = True
show_plots = False
# save_plots = False
# show_plots = True

RCF_CHECKPOINT = ROOT_PATH / 'tmp' / 'bsds500_pascal_model.pth'

SHAPE_NAME = 'crystal'
VERTEX_KEY = SHAPE_NAME + '.vertex_positions'
FACES_KEY = SHAPE_NAME + '.faces'
BSDF_KEY = SHAPE_NAME + '.bsdf'
COLOUR_KEY = BSDF_KEY + '.reflectance.value'


def _generate_crystal(
        distances: List[float] = [1.0, 0.5, 0.2],
        scale: float = 10.,
        origin: List[float] = [0, 0, 20],
        rotvec: List[float] = [0, 0, 0],
) -> Crystal:
    """
    Generate a beta-form LGA crystal.
    """
    reader = EntryReader()
    crystal = reader.crystal('LGLUAC01')
    miller_indices = [(1, 1, 1), (0, 1, 2), (0, 0, 2)]
    lattice_unit_cell = [crystal.cell_lengths[0], crystal.cell_lengths[1], crystal.cell_lengths[2]]
    lattice_angles = [crystal.cell_angles[0], crystal.cell_angles[1], crystal.cell_angles[2]]
    point_group_symbol = '222'  # crystal.spacegroup_symbol

    crystal = Crystal(
        lattice_unit_cell=lattice_unit_cell,
        lattice_angles=lattice_angles,
        miller_indices=miller_indices,
        point_group_symbol=point_group_symbol,
        distances=torch.tensor(distances) * scale,
        origin=origin,
        rotation=rotvec,
    )
    crystal.to(device)

    return crystal


def create_scene(crystal: Crystal, spp=256, res=400) -> mi.Scene:
    """
    Create a Mitsuba scene containing the given crystal.
    """
    from crystalsizer3d.scene_components.utils import build_crystal_mesh
    crystal_mesh = build_crystal_mesh(
        crystal,
        material_bsdf={
            'type': 'roughdielectric',
            'distribution': 'beckmann',
            'alpha': 0.02,
            'int_ior': 1.78,
        },
        shape_name=SHAPE_NAME,
        bsdf_key=BSDF_KEY
    )

    scene = mi.load_dict({
        'type': 'scene',

        # Camera and rendering parameters
        'integrator': {
            'type': 'prb_projective',
            # 'type': 'prbvolpath',
            # 'type': 'ptracer',
            # 'type': 'path',
            # 'type': 'direct_projective',
            # 'type': 'volpathmis',
            # 'type': 'volpath',
            'max_depth': 64,
            'rr_depth': 5,
            'sppi': 0,
            'guiding': 'grid',
            'guiding_rounds': 10
        },
        'sensor': {
            'type': 'perspective',
            # 'near_clip': 0.001,
            # 'far_clip': 1000,
            'fov': 27,
            'to_world': T.look_at(
                origin=[0, 0, 100],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                # 'type': 'independent',
                'type': 'stratified',  # seems better than independent
                # 'type': 'multijitter',  # better than indep, but maybe worse than strat
                # 'type': 'orthogonal',  # diverges a bit like indep
                # 'type': 'ldsampler',  # seems decent
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': res,
                'height': res,
                'filter': {'type': 'gaussian'},
                'sample_border': True,
            },
        },

        # Emitters
        'light': {
            'type': 'rectangle',
            'to_world': T.scale(50),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': 0.5
                }
            },
        },
        # 'light2': {
        #     'type': 'rectangle',
        #     'to_world': T.look_at(
        #         origin=[0, 0, 100],
        #         target=[0, 0, 0],
        #         up=[0, 1, 0]
        #     ),
        #     'emitter': {
        #         'type': 'area',
        #         'radiance': {
        #             'type': 'rgb',
        #             'value': 5000.0
        #         }
        #     },
        # },

        # Shapes
        'surface': {
            'type': 'rectangle',
            'to_world': T.translate([0, 0, 1]) @ T.scale(25),
            'surface_material': {
                'type': 'dielectric',
                'int_ior': 1.,
                # 'specular_transmittance': {
                #     'type': 'bitmap',
                #     # 'bitmap': mi.Bitmap(dr.ones(mi.TensorXf, (12, 12, 3)))
                #     # 'bitmap': mi.Bitmap(surf),
                #     'filename': str(ROOT_PATH / 'tmp' / 'grid_1000x1000.png'),
                #     # 'type': 'rgb',
                #     # 'value': (1,0,0),
                #     'wrap_mode': 'clamp',
                # }
            },

        },
        SHAPE_NAME: crystal_mesh
    })

    return scene


def load_vgg() -> nn.Module:
    """
    Load the VGG perceptual model.
    """
    vgg = timm.create_model(
        # model_name='timm/convnext_tiny.in12k_ft_in1k',
        model_name='timm/bat_resnext26ts.ch_in1k',
        pretrained=True,
        num_classes=0,
        features_only=True
    )
    # vgg = torch.jit.script(vgg)
    vgg.to(device)
    vgg.eval()
    # model = vgg

    data_config = timm.data.resolve_model_data_config(vgg)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    transforms = Compose([
        transforms.transforms[0],
        transforms.transforms[1],
        transforms.transforms[3],
    ])

    def model(x):
        x = transforms(x)
        return vgg(x)

    # model = torch.jit.script(model)

    return model


def load_rcf() -> Tuple[RCF, List[EMA]]:
    """
    Load the RCF edge finder model.
    """
    assert RCF_CHECKPOINT.exists(), f'RCF checkpoint not found at {RCF_CHECKPOINT}'
    logger.info(f'Loading RCF model checkpoint from {RCF_CHECKPOINT}.')
    rcf = RCF()
    checkpoint = torch.load(RCF_CHECKPOINT)
    rcf.load_state_dict(checkpoint, strict=False)
    rcf = torch.jit.script(rcf)
    rcf.to(device)
    rcf.eval()

    rcf_emas = [[EMA(decay=0.99) for _ in range(2)] for _ in range(6)]

    return rcf, rcf_emas


# @torch.jit.script
def to_multiscale(img: torch.Tensor, blur: GaussianBlur) -> List[torch.Tensor]:
    """
    Generate downsampled and blurred images.
    """
    imgs = [img.clone().permute(2, 0, 1)[None, ...]]
    while min(imgs[-1].shape[-2:]) > blur.kernel_size[0] + 2:
        imgs.append(blur(F.interpolate(imgs[-1], scale_factor=0.5, mode='bilinear')))
    imgs = [i[0].permute(1, 2, 0) for i in imgs]
    return imgs


# @torch.jit.script
def loss_(
        target: Union[torch.Tensor, List[torch.Tensor]],
        samples: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
        loss_type: str = 'l2',
        decay_factor: float = 1.,
        model: Optional[nn.Module] = None,
        rcf_emas: Optional[List[EMA]] = None
):
    """
    Calculate losses.
    """

    # If the input is a list of lists then it is a multiscale probabilistic samples
    if type(samples) == list and type(samples[0]) == list:
        samples = [torch.stack([b[i] for b in samples]) for i in range(len(samples[0]))]
        target = [a[None, ...].expand_as(b) for a, b in zip(target, samples)]
    elif isinstance(target, torch.Tensor) and isinstance(samples, torch.Tensor) and target.shape != samples.shape:
        if target.ndim == 3 and samples.ndim == 3:
            raise RuntimeError(f'Image shapes do not match: {target.shape} != {samples.shape}')
        if target.ndim == 3:
            assert samples.ndim == 4, f'Image shapes do not match: {target.shape} != {samples.shape}'
            target = target[None, ...].expand_as(samples)
        else:
            assert target.ndim == 4, f'Image shapes do not match: {target.shape} != {samples.shape}'
            assert len(target) == 1, f'Target image must be a single image, not a batch. {len(target)} received.'
            target = target.expand_as(samples[0])

    # Multiscale losses
    if isinstance(target, list):
        if loss_type in ['vgg', 'rcf']:
            return loss_(target[0], samples[0], loss_type=loss_type, decay_factor=decay_factor, model=model,
                         rcf_emas=rcf_emas)
        losses = [loss_(a, b, loss_type=loss_type, decay_factor=decay_factor) for a, b in zip(target, samples)]
        for i in range(len(losses)):
            losses[i] = losses[i] * decay_factor**i
        return sum(losses)

    # Single scale loss
    if loss_type == 'l.5':
        return torch.mean(((target - samples).abs() + 1e-6).sqrt())
    elif loss_type == 'l1':
        return torch.mean((target - samples).abs())
    elif loss_type == 'l2':
        return torch.mean((target - samples)**2)
    elif loss_type == 'l4':
        return torch.mean((target - samples)**4)
    elif loss_type == 'vgg':
        assert model is not None, 'VGG model must be provided for VGG loss.'
        if target.ndim == 3 and samples.ndim == 3:
            vgg_input = torch.stack([target, samples]).permute(0, 3, 1, 2)
        else:
            vgg_input = torch.cat([target[0][None, ...], samples]).permute(0, 3, 1, 2)
        vgg_feats = model(vgg_input)
        losses = [loss_(f[0], f[1:], loss_type='l1') for f in vgg_feats]
        for i in range(len(losses)):
            losses[i] = losses[i] * decay_factor**i
        return sum(losses)
    elif loss_type == 'rcf':
        assert model is not None, 'RCF model must be provided for VGG loss.'
        if target.ndim == 3 and samples.ndim == 3:
            rcf_input = torch.stack([target, samples]).permute(0, 3, 1, 2)
        else:
            rcf_input = torch.cat([target[0][None, ...], samples]).permute(0, 3, 1, 2)
        rcf_feats = model(rcf_input, apply_sigmoid=False)
        losses = [loss_(f[0], f[1:], loss_type='l1') for f in rcf_feats]

        # if rcf_emas is not None:
        #     normed_feats = []
        #     for i, f in enumerate(rcf_feats):
        #         min_val = rcf_emas[i][0](f.min().item())
        #         max_val = rcf_emas[i][1](f.max().item())
        #         nf = torch.clamp((f - min_val) / (max_val - min_val), min=0, max=1)
        #         normed_feats.append(nf)
        #     rcf_feats = normed_feats
        # else:
        #     rcf_feats = [torch.stack([
        #         (a - a.min()) / (a.max() - a.min()),
        #         (b - b.min()) / (b.max() - b.min())
        #     ]) for a, b in rcf_feats]

        # loss = torch.mean((rcf_feats[0][0] - rcf_feats[0][1]).abs())
        # loss = torch.mean((rcf_feats[-1][0] - rcf_feats[-1][1])**2)
        # return loss, rcf_feats

        # if loss_type != 'rcf' and loss_type[4:] == 'l1':
        #     losses = [torch.mean((a - b).abs()) for a, b in rcf_feats]
        # else:
        #     losses = [torch.mean((a - b)**2) for a, b in rcf_feats]
        for i in range(len(losses)):
            losses[i] = losses[i] * decay_factor**i
        return sum(losses), rcf_feats
    else:
        raise ValueError(f'Unknown loss type: {loss_type}.')


def plot_scene():
    """
    Debug function to plot the basic scene.
    """
    spp = 2**9
    crystal = _generate_crystal(
        distances=[1.0, 0.6, 0.6],
        scale=10,
        origin=[0, 0, 20],
        rotvec=[0.2, 0.2, -0.3],
    )
    scene = create_scene(crystal=crystal, spp=spp)
    image = mi.render(scene)
    plt.imshow(image**(1.0 / 2.2))
    plt.axis('off')
    plt.show()


def plot_comparison(
        img_target: torch.Tensor,
        img_samples: List[torch.Tensor],
        rcf_feats_target: Optional[List[torch.Tensor]] = None,
        rcf_feats_samples: Optional[List[torch.Tensor]] = None,
        i: int = 0,
        loss: float = 0.0,
        save_dir: Optional[Path] = None,
        plot_n_samples: int = -1
):
    """
    Plot the target and optimised images side by side.
    """
    if rcf_feats_target is not None:
        n_rows = 2
    else:
        n_rows = 1
    if len(img_samples) > 0:
        if plot_n_samples == -1:
            n_cols = 1 + len(img_samples)
        else:
            n_cols = 1 + min(len(img_samples), plot_n_samples)
    else:
        n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.5), squeeze=False)
    for ax, img in zip(axes[0], [img_target, *img_samples]):
        if isinstance(img, list):
            img = img[0]
        img = np.clip(to_numpy(img)**(1.0 / 2.2), 0, 1)
        ax.imshow(img)
        ax.axis('off')
    if rcf_feats_target is not None:
        assert rcf_feats_samples is not None, 'RCF features must be provided for comparison.'
        for ax, f in zip(axes[1], [rcf_feats_target[0], *rcf_feats_samples[0]]):
            f = (f - f.min()) / (f.max() - f.min())
            img = np.clip(to_numpy(f), 0, 1)
            ax.imshow(img, cmap='Blues')
            ax.axis('off')
    fig.suptitle(f'Iteration {i} Loss: {loss:.4E}')
    fig.tight_layout()
    if save_plots:
        assert save_dir is not None, 'Save directory must be provided to save plots.'
        plt.savefig(save_dir / f'iteration_{i:05d}.png')
    if show_plots:
        plt.show()
    plt.close(fig)
    # exit()


def optimise_scene():
    # Parameters
    spp = 512
    res = 200
    use_min_lib = False
    # opt_algorithm = 'radam'
    # opt_algorithm = 'adamw'
    # opt_algorithm = 'rmsprop'
    opt_algorithm = 'sgd'
    # opt_algorithm = 'adadelta'
    # opt_algorithm = 'adabelief'
    # opt_algorithm = 'lbfgs'
    # opt_algorithm = 'newton-cg'

    # lr = 0.1
    # lr = 0.05
    # lr = 1e-2
    # lr = 1e-3
    lr = 1e-4
    # lr = 1e-5
    n_iterations = 1500
    plot_freq = 3
    plot_n_samples = 4
    multiscale = True
    probabilistic = False
    n_samples = 20
    if not probabilistic:
        n_samples = 1

    if save_plots:
        save_dir = LOGS_PATH / (START_TIMESTAMP +
                                f'_spp{spp}' +
                                f'_res{res}' +
                                # (f'_ms' if multiscale else '') +
                                (f'_prob{n_samples}' if probabilistic else '') +
                                f'_{opt_algorithm}' +
                                f'_lr{lr}' +
                                f'_{n_iterations}')
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    # Set up target and initial scenes
    crystal_target = _generate_crystal(
        distances=[1.0, 0.5, 0.2],
        scale=10,
        origin=[0, 0, 20],
        rotvec=[0, 0, 0.5],
    )
    crystal_opt = _generate_crystal(
        distances=[1.0, 0.5, 0.2],
        # distances=[1.0, 0.3, 0.1],
        scale=10,
        origin=[0, 0, 20],
        rotvec=[0, 0.2, 0.2],
        # rotvec=[0.5, 0.5, -0.3],
    )

    """
    Tests:
    target: d=[1.0, 0.5, 0.2], r=[0, 0, 0.5]
    initial: d=[1.0, 0.5, 0.2], r=[0, 0.2, 0.2], lr=1e-2 (unstable), lr=1e-3 (wobbles, but stable), lr=1e-4 (stable)    
    """

    crystal_target = _generate_crystal(
        distances=[1.0, 0.6, 0.6],
        scale=10,
        origin=[0, 0, 20],
        rotvec=[0.2, 0.2, -0.3],
    )
    crystal_opt = _generate_crystal(
        distances=[1.0, 0.5, 0.5],
        scale=10,
        origin=[0, 0, 20],
        rotvec=[0, 0, 0],
    )

    scene_target = create_scene(crystal=crystal_target, spp=spp, res=res)
    scene = create_scene(crystal=crystal_opt, spp=spp, res=res)
    scene_params = mi.traverse(scene)

    # Save the initial scene image
    if save_plots:
        img = mi.render(scene_target).numpy()
        img = np.clip(img**(1.0 / 2.2), 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).convert('L').save(save_dir / 'target.png')

    @dr.wrap_ad(source='torch', target='drjit')
    def render_image(vertices, faces, seed=1):
        scene_params[VERTEX_KEY] = dr.ravel(vertices)
        scene_params[FACES_KEY] = dr.ravel(faces)
        scene_params.update()
        return mi.render(scene, scene_params, seed=seed)

    # Instantiate vision models
    vgg = load_vgg()
    rcf, rcf_emas = load_rcf()

    if multiscale:
        # Create a GaussianBlur to apply to the downsampled images to ensure gradient when nothing intersects
        blur = GaussianBlur(kernel_size=5, sigma=1.0)
        blur = torch.jit.script(blur)
        blur.to(device)
        img_target = mi.render(scene_target).torch()
        img_target = to_multiscale(img_target, blur)
        resolution_pyramid = [t.shape[0] for t in img_target[::-1]]
        logger.info(f'Resolution pyramid has {len(img_target)} levels: '
                    f'{", ".join([str(res) for res in resolution_pyramid])}')

    # Set up optimiser
    params = {
        'distances': [crystal_opt.distances],
        'origin': [crystal_opt.origin],
        'rotvec': [crystal_opt.rotation],
    }
    if probabilistic:
        # params['distances'].append(crystal_opt.distances_logvar)
        # params['origin'].append(crystal_opt.origin_logvar)
        # params['rotvec'].append(crystal_opt.rotvec_logvar)
        params['distances_logvar'] = [crystal_opt.distances_logvar]
        params['origin_logvar'] = [crystal_opt.origin_logvar]
        params['rotvec_logvar'] = [crystal_opt.rotation_logvar]

    if use_min_lib:
        opt = Minimizer(crystal_opt.parameters(),
                        method=opt_algorithm,
                        tol=1e-6,
                        max_iter=200,
                        disp=2)
    else:
        if opt_algorithm == 'lbfgs':
            opt = LBFGS(crystal_opt.parameters(), lr=lr, max_iter=20, max_eval=1000,
                        history_size=1000)  # , line_search_fn='strong_wolfe')
        else:
            opt = create_optimizer_v2(
                opt=opt_algorithm,
                # betas=(0.8, 0.99),
                # betas=(0.99, 0.999),
                # momentum=0.9,
                lr=lr,
                model_or_params=[
                    {'params': params['origin'], 'lr': lr},
                    {'params': params['rotvec'], 'lr': lr},
                    # {'params': params['origin'], 'lr': lr / 2},
                    # {'params': params['rotvec'], 'lr': lr / 10},
                    {'params': params['distances'], 'lr': lr * 100},
                    # {'params': self.hex.temp, 'lr': self.lr_temp}
                    # {'params': params['distances_logvar'], 'lr': lr * 10},
                    # {'params': params['origin_logvar'], 'lr': lr * 10},
                    # {'params': params['rotvec_logvar'], 'lr': lr * 10},
                ],
                # model_or_params=crystal_opt.parameters(),
                # model_or_params=agent.parameters(),
                weight_decay=0
            )
    # y_ema = EMA()

    # Optimise the scene
    losses = []
    img_losses = []
    vgg_losses = []
    rcf_losses = []
    d_diffs = []
    o_diffs = []
    r_diffs = []
    for i in range(n_iterations):
        img_target = None
        img_samples = []
        rcf_feats_target = None
        rcf_feats_samples = []
        loss = None

        def closure():
            nonlocal img_target, img_samples, rcf_feats_target, rcf_feats_samples, loss
            crystal_opt.clamp_parameters()
            opt.zero_grad()

            if probabilistic:
                # Sample parameters
                std_dists = torch.exp(0.5 * crystal_opt.distances_logvar)
                eps_dists = torch.randn(n_samples, *std_dists.size(), device=device)
                distances_samples = crystal_opt.distances + eps_dists * std_dists.unsqueeze(0)
                std_origin = torch.exp(0.5 * crystal_opt.origin_logvar)
                eps_origin = torch.randn(n_samples, *std_origin.size(), device=device)
                origin_samples = crystal_opt.origin + eps_origin * std_origin.unsqueeze(0)
                std_rotvec = torch.exp(0.5 * crystal_opt.rotation_logvar)
                eps_rotvec = torch.randn(n_samples, *std_rotvec.size(), device=device)
                rotvec_samples = crystal_opt.rotation + eps_rotvec * std_rotvec.unsqueeze(0)

                # Set first sample to the mean
                distances_samples[0] = crystal_opt.distances
                origin_samples[0] = crystal_opt.origin
                rotvec_samples[0] = crystal_opt.rotation

                # Clamp parameters
                distances_samples = torch.clamp(distances_samples, 1e-1, None)
                origin_samples = torch.clamp(origin_samples, -1e3, 1e3)

            # Render new target image
            img_target = mi.render(scene_target, seed=i).torch()
            img_target = torch.clip(img_target, 0, 1)
            if multiscale:
                img_target = to_multiscale(img_target, blur)

            # Render new image samples
            img_samples = []
            for j in range(n_samples):
                # Build the mesh with updated parameters
                if probabilistic:
                    v, f = crystal_opt.build_mesh(
                        distances=distances_samples[j],
                        origin=origin_samples[j],
                        rotation=rotvec_samples[j]
                    )
                else:
                    v, f = crystal_opt.build_mesh()

                # Render new image
                img_i = render_image(v, f, seed=i)
                img_i = torch.clip(img_i, 0, 1)

                # rcf_loss, rcf_feats = image_loss(img_target, img_i, loss_type='rcf', model=rcf, rcf_emas=rcf_emas)
                # img_target = rcf_feats[0][0].expand(3, res, res).permute(1,2,0).detach()
                # img_i = rcf_feats[0][1].expand(3, res, res).permute(1,2,0)

                if multiscale:
                    # img_target = to_multiscale(img_target, blur)
                    img_i = to_multiscale(img_i, blur)
                img_samples.append(img_i)

            # Calculate losses
            lp5_loss = loss_(img_target, img_samples, loss_type='l.5', decay_factor=0.1)
            l1_loss = loss_(img_target, img_samples, loss_type='l1', decay_factor=2)
            l2_loss = loss_(img_target, img_samples, loss_type='l2', decay_factor=2)
            l4_loss = loss_(img_target, img_samples, loss_type='l4', decay_factor=10)
            # img_loss = lp5_loss + 0.5 * l1_loss + 0.25 * l2_loss
            # img_loss = lp5_loss + l1_loss + l2_loss
            # img_loss = lp5_loss   #+ 0.5*l2_loss
            # img_loss = l1_loss + 2 * l2_loss  #+ 4 * l4_loss
            # img_loss = l1_loss + l2_loss + l4_loss
            # img_loss = l1_loss + 0.5 * l2_loss + 0.25 * l4_loss
            # img_loss = l1_loss + l2_loss  #+ 100 * l4_loss
            # img_loss = l4_loss + 1/2 * l2_loss + 1/4 * l1_loss
            # img_loss = l1_loss
            img_loss = l2_loss  # + 0.5*l2_loss
            # img_loss = l2_loss + 0.5 * l1_loss + 0.25 * lp5_loss

            vgg_loss = loss_(img_target, img_samples, loss_type='vgg', decay_factor=1, model=vgg)
            # rcf_loss, rcf_feats = loss_(img_target, img_samples, loss_type='rcf', decay_factor=1, model=rcf)  #, rcf_emas=rcf_emas)
            # rcf_feats_target = [f[0, 0] for f in rcf_feats]
            # rcf_feats_samples = [f[1:, 0] for f in rcf_feats]

            # Combine losses and backpropagate errors
            # loss = img_loss
            loss = img_loss + vgg_loss
            # loss_j = img_loss + 1e-3 * vgg_loss + 1e-2 * rcf_loss
            # loss = img_loss + rcf_loss + vgg_loss
            # loss = 10*img_loss + 0.1*vgg_loss
            # loss_j = img_loss   #+ 100*rcf_loss  #+ 0.1*vgg_loss
            # loss = rcf_loss  #+ 10 * img_loss
            # loss = vgg_loss

            # loss = loss + loss_j

            if not use_min_lib:
                loss.backward()

            # Logging
            d_diff = torch.norm(crystal_opt.distances - crystal_target.distances)
            o_diff = torch.norm(crystal_opt.origin - crystal_target.origin)
            r_diff = torch.norm(crystal_opt.rotation - crystal_target.rotation)
            img_losses.append(img_loss.item())
            vgg_losses.append(vgg_loss.item())
            # rcf_losses.append(rcf_loss.item())
            losses.append(loss.item())
            d_diffs.append(d_diff.item())
            o_diffs.append(o_diff.item())
            r_diffs.append(r_diff.item())
            logger.info(
                f'Iteration {i} ' +
                f'Loss: {loss.item():.3E} ' +
                f'Img-loss: {img_loss.item():.3E} ' +
                f'L1: {l1_loss.item():.3E} ' +
                f'L2: {l2_loss.item():.3E} ' +
                f'L4: {l4_loss.item():.3E} ' +
                f'vgg: {vgg_loss.item():.3E} ' +
                # f'rcf: {rcf_loss.item():.3E} ' +
                f'd-error: {d_diff.item():.3E} ' +
                f'(d=[' + ','.join([f'{v:.2f}' for v in crystal_opt.distances]) + ']' +
                (
                    f'[{",".join([f"{v:.3E}" for v in crystal_opt.distances_logvar.exp()])}]' if probabilistic else '') + ') ' +
                f'o-error: {o_diff.item():.3E} ' +
                f'(o=[' + ','.join([f'{v:.2f}' for v in crystal_opt.origin]) + ']' +
                (
                    f'[{",".join([f"{v:.3E}" for v in crystal_opt.origin_logvar.exp()])}]' if probabilistic else '') + ') ' +
                f'r-error: {r_diff.item():.3E} ' +
                f'(r=[' + ','.join([f'{v:.2f}' for v in crystal_opt.rotation]) + ']' +
                (
                    f'[{",".join([f"{v:.3E}" for v in crystal_opt.rotation_logvar.exp()])}]' if probabilistic else '') + ')'
            )

            return loss

        opt.step(closure)

        # Plot
        if i % plot_freq == 0:
            # plot_comparison(img_target, img_samples, rcf_feats_target, rcf_feats_samples, i, loss, save_dir,
            #                 plot_n_samples)
            # Image.fromarray(img_samples[0]).convert('L').save(save_dir / f'opt_{i:05d}.png')
            img = np.clip(to_numpy(img_samples[0][0])**(1.0 / 2.2), 0, 1)
            Image.fromarray((img * 255).astype(np.uint8)).convert('L').save(save_dir / f'opt_{i:05d}.png')

        # if i > 200:
        #     multiscale = False

    # Plot the final scene comparison
    plot_comparison(img_target, img_samples, rcf_feats_target, rcf_feats_samples, i, loss, save_dir, plot_n_samples)

    # Plot the losses
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.plot(losses)
    ax.set_title('Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    ax.set_yscale('log')
    ax = axes[1]
    ax.plot(d_diffs, label='Distances')
    ax.plot(o_diffs, label='Origin')
    ax.plot(r_diffs, label='Rotation')
    ax.set_title('Parameter Convergence')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'losses.png')
    if show_plots:
        plt.show()


def track_losses():
    """
    Track the losses over time.
    """
    loss_keys = ['l.5', 'l1', 'l2', 'l4', 'vgg', 'rcf']

    # Parameters
    spp = 128
    res = 200
    n_steps = 100
    plot_freq = 3

    if save_plots:
        save_dir = LOGS_PATH / (START_TIMESTAMP +
                                f'_spp{spp}' +
                                f'_res{res}' +
                                f'_tracking' +
                                f'_{n_steps}')
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    # Set up target and initial scenes
    crystal_target = _generate_crystal(
        distances=[1.0, 0.5, 0.2],
        scale=10,
        origin=[0, 0, 20],
        rotvec=[0, 0, 0.5],
    )
    crystal_opt = _generate_crystal(
        # distances=[1.0, 0.5, 0.2],
        # distances=[1.0, 0.3, 0.1],
        distances=[2.0, 0.7, 0.1],
        scale=10,
        origin=[0, 0, 20],
        # rotvec=[0, 0.2, 0.2],
        # rotvec=[0.5, 0.5, -0.3],
        rotvec=[1.5, 1.5, -0.8],
    )
    scene_target = create_scene(crystal=crystal_target, spp=spp, res=res)
    scene = create_scene(crystal=crystal_opt, spp=spp, res=res)
    scene_params = mi.traverse(scene)

    # Parameter progressions
    k = torch.linspace(0, 1, n_steps)
    k3 = k[None, ...].expand(3, n_steps).T.to(device)
    k4 = k[None, ...].expand(4, n_steps).T.to(device)
    distances = k3 * (crystal_target.distances[None, ...] - crystal_opt.distances[None, ...]) + crystal_opt.distances[
        None, ...]
    origin = k3 * (crystal_target.origin[None, ...] - crystal_opt.origin[None, ...]) + crystal_opt.origin[None, ...]
    rotvec = k4 * (crystal_target.rotation[None, ...] - crystal_opt.rotation[None, ...]) + crystal_opt.rotation[
        None, ...]

    @dr.wrap_ad(source='torch', target='drjit')
    def render_image(vertices, faces, seed=1):
        scene_params[VERTEX_KEY] = dr.ravel(vertices)
        scene_params[FACES_KEY] = dr.ravel(faces)
        scene_params.update()
        return mi.render(scene, scene_params, seed=seed)

    # Create a GaussianBlur to apply to the downsampled images to ensure gradient when nothing intersects
    blur = GaussianBlur(kernel_size=5, sigma=1.0)
    blur = torch.jit.script(blur)
    blur.to(device)
    img_target = mi.render(scene_target).torch()
    img_target = to_multiscale(img_target, blur)
    resolution_pyramid = [t.shape[0] for t in img_target[::-1]]
    logger.info(f'Resolution pyramid has {len(img_target)} levels: '
                f'{", ".join([str(res) for res in resolution_pyramid])}')

    # Instantiate vision models
    vgg = load_vgg()
    rcf, _ = load_rcf()
    nn_input = img_target[0][None, ...].permute(0, 3, 1, 2)
    vgg_feats = vgg(nn_input)
    rcf_feats = rcf(nn_input)

    # Track the losses as we move from the initial to the target parameters
    losses = {}
    for lk in loss_keys:
        if lk == 'vgg':
            losses[lk] = np.zeros((n_steps, len(vgg_feats)))
        elif lk == 'rcf':
            losses[lk] = np.zeros((n_steps, len(rcf_feats)))
        else:
            losses[lk] = np.zeros((n_steps, len(resolution_pyramid)))
    for i in range(n_steps):
        if (i + 1) % 10 == 0:
            logger.info(f'Iteration {i + 1}/{n_steps}.')

        # Render new target image
        img_target = mi.render(scene_target, seed=i).torch()
        img_target = torch.clip(img_target, 0, 1)
        img_target = to_multiscale(img_target, blur)

        # Render new tracking image
        v, f = crystal_opt.build_mesh(
            distances=distances[i],
            origin=origin[i],
            rotation=rotvec[i]
        )
        img_i = render_image(v, f, seed=i)
        img_i = torch.clip(img_i, 0, 1)
        img_i = to_multiscale(img_i, blur)
        nn_input = torch.stack([img_target[0], img_i[0]]).permute(0, 3, 1, 2)

        # Calculate losses
        for lk in loss_keys:
            if lk == 'l.5':
                ls = [torch.mean(((a - b).abs() + 1e-6).sqrt()) for a, b in zip(img_target, img_i)]
            elif lk == 'l1':
                ls = [torch.mean((a - b).abs()) for a, b in zip(img_target, img_i)]
            elif lk == 'l2':
                ls = [torch.mean((a - b)**2) for a, b in zip(img_target, img_i)]
            elif lk == 'l4':
                ls = [torch.mean((a - b)**4) for a, b in zip(img_target, img_i)]
            elif lk == 'vgg':
                vgg_feats = vgg(nn_input)
                ls = [torch.mean((f[0] - f[1:])**2) for f in vgg_feats]
            elif lk == 'rcf':
                rcf_feats = rcf(nn_input)
                ls = [torch.mean((f[0] - f[1:])**2) for f in rcf_feats]
            else:
                raise ValueError(f'Unknown loss type: {lk}.')
            losses[lk][i] = [l.item() for l in ls]

        # Plot
        if i % plot_freq == 0:
            plot_comparison(img_target, img_i, i=i, save_dir=save_dir)

    # Plot the final scene comparison
    plot_comparison(img_target, img_i, i=i, save_dir=save_dir)

    # Plot the losses
    n_rows = len(loss_keys)
    n_cols = max(len(resolution_pyramid), len(vgg_feats), len(rcf_feats))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2), sharex=True)

    for i, (lk, losses_k) in enumerate(losses.items()):
        for j, l in enumerate(losses_k.T):
            ax = axes[i, j]
            if j == 0:
                ax.set_ylabel(lk)
            if lk in ['vgg', 'rcf']:
                ax.set_title(f'{lk}{j}')
            else:
                res = resolution_pyramid[::-1][j]
                ax.set_title(f'{res}x{res}')
            ax.plot(l)
            if i == n_rows - 1:
                ax.set_xlabel('Iteration')
            ax.set_ylim(bottom=l[int(n_steps * 0.9)])
            ax.set_yscale('log')
    fig.tight_layout()
    if save_plots:
        plt.savefig(save_dir / 'losses.png')
    if show_plots:
        plt.show()


if __name__ == '__main__':
    # plot_scene()
    optimise_scene()
    # track_losses()
