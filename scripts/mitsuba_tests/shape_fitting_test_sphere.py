from typing import List, Tuple

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry import axis_angle_to_rotation_matrix
from pytorch3d.utils import ico_sphere
from torch import nn
from torchvision.transforms import GaussianBlur

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.util.utils import to_numpy

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('scalar_rgb')
    device = torch.device('cpu')

from mitsuba import ScalarTransform4f as T

# save_plots = False
# show_plots = True
save_plots = True
show_plots = False

SHAPE_NAME = 'sphere'
VERTEX_KEY = SHAPE_NAME + '.vertex_positions'
FACES_KEY = SHAPE_NAME + '.faces'
BSDF_KEY = SHAPE_NAME + '.bsdf'
COLOUR_KEY = BSDF_KEY + '.reflectance.value'


class Sphere(nn.Module):

    def __init__(
            self,
            scale: float = 1.0,
            origin: List[float] = [0, 0, 0],
            rotvec: List[float] = [0, 0, 0],
            colour: List[float] = [1, 1, 1]
    ):
        """
        Create a sphere with the given scale, origin, and rotation.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=True)
        self.origin = nn.Parameter(torch.tensor(origin, dtype=torch.float32), requires_grad=True)
        self.rotvec = nn.Parameter(torch.tensor(rotvec, dtype=torch.float32), requires_grad=True)
        self.colour = nn.Parameter(torch.tensor(colour, dtype=torch.float32), requires_grad=True)

    def build_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a sphere mesh with the given origin and radius.
        """

        # Build basic sphere
        sphere = ico_sphere(level=1, device=self.origin.device)
        vertices = sphere.verts_packed()
        faces = sphere.faces_packed()

        # Apply scaling
        vertices = vertices * self.scale

        # Apply translation
        vertices = vertices + self.origin

        # Apply rotation
        R = axis_angle_to_rotation_matrix(self.rotvec[None, :]).squeeze(0)
        vertices = vertices @ R.T

        return vertices, faces


def build_mitsuba_mesh(shape: Sphere) -> mi.Mesh:
    """
    Convert a Sphere object into a Mitsuba mesh.
    """
    # Build the mesh in pytorch and convert the parameters to Mitsuba format
    vertices, faces = shape.build_mesh()
    nv, nf = len(vertices), len(faces)
    vertices = mi.TensorXf(vertices)
    faces = mi.TensorXi64(faces)

    # Set up the material properties
    props = mi.Properties()
    props[BSDF_KEY] = mi.load_dict({
        'type': 'diffuse',
        'reflectance': {
            'type': 'rgb',
            'value': shape.colour.tolist()
        }
    })

    # Construct the mitsuba mesh and set the vertex positions and faces
    mesh = mi.Mesh(
        SHAPE_NAME,
        vertex_count=nv,
        face_count=nf,
        has_vertex_normals=False,
        has_vertex_texcoords=False,
        props=props
    )
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['faces'] = dr.ravel(faces)

    return mesh


def create_scene(shape: Sphere, spp=256, res=400) -> mi.Scene:
    """
    Create a Mitsuba scene containing the given shape.
    """
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'prb_projective',
            'max_depth': 16,
            'rr_depth': 3,
            # 'sppi': 0
        },
        'sensor': {
            'type': 'perspective',
            'to_world': T.look_at(
                origin=[0, 0, 10],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
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
        'light': {
            'type': 'rectangle',
            'to_world': T.scale(50) @ T.look_at(
                origin=[0, 0, 10],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': np.ones(3) * 50.0
                }
            },
        },
        SHAPE_NAME: build_mitsuba_mesh(shape)
    })

    return scene


def to_multiscale(img: torch.Tensor, blur: GaussianBlur) -> List[torch.Tensor]:
    """
    Generate downsampled and blurred images.
    """
    imgs = [img.clone().permute(2, 0, 1)[None, ...]]
    while min(imgs[-1].shape[-2:]) > blur.kernel_size[0] + 2:
        imgs.append(blur(F.interpolate(imgs[-1], scale_factor=0.5, mode='bilinear')))
    imgs = [i[0].permute(1, 2, 0) for i in imgs]
    return imgs


def plot_scene():
    spp = 2**9
    sphere = Sphere(scale=2)
    scene = create_scene(shape=sphere.mesh)
    image = mi.render(scene, spp=spp)
    plt.imshow(image**(1.0 / 2.2))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def optimise_scene():
    if save_plots:
        save_dir = LOGS_PATH / 'fit_sphere' / START_TIMESTAMP
        save_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    res = 100
    spp = 16
    n_iterations = 2000
    lr = 0.1
    plot_freq = 5
    multiscale = True

    # Set up target and initial scenes
    sphere_target = Sphere(
        scale=2,
        origin=[1, 0.5, 0.1],
        rotvec=[0.5, 0.5, 0],
        colour=[0, 1, 0]
    )
    sphere_opt = Sphere(
        scale=1,
        origin=[-1, -0.2, 1],
        rotvec=[0, 5, -2],
        colour=[0, 0, 1]
    )
    sphere_target.to(device)
    sphere_opt.to(device)
    scene_target = create_scene(shape=sphere_target, spp=spp, res=res)
    scene = create_scene(shape=sphere_opt, spp=spp, res=res)
    params = mi.traverse(scene)

    if multiscale:
        # Create a GaussianBlur to apply to the downsampled images to ensure gradient when nothing intersects
        blur = GaussianBlur(kernel_size=5, sigma=1.0)
        blur = torch.jit.script(blur)
        blur.to(device)

    @dr.wrap_ad(source='torch', target='drjit')
    def render_image(vertices, faces, colour, seed=1):
        params[VERTEX_KEY] = dr.ravel(vertices)
        params[FACES_KEY] = dr.ravel(faces)
        params[COLOUR_KEY] = dr.unravel(mi.Color3f, colour)
        params.update()
        return mi.render(scene, params, seed=seed)

    def plot(img_opt):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, img in zip(axes, [img_target, img_opt]):
            if isinstance(img, list):
                img = img[0]
            img = to_numpy(img)
            ax.imshow(img**(1.0 / 2.2))
            ax.axis('off')
        fig.suptitle(f'Iteration {i} Loss: {loss:.4E}')
        fig.tight_layout()
        if save_plots:
            plt.savefig(save_dir / f'iteration_{i:05d}.png')
        if show_plots:
            plt.show()
        plt.close(fig)

    # Set up optimiser
    opt = torch.optim.Adam(sphere_opt.parameters(), lr=lr)

    # Optimise the scene
    losses = []
    s_diffs = []
    o_diffs = []
    r_diffs = []
    c_diffs = []
    for i in range(n_iterations):
        if i > 0:
            # Take a gradient step and update the parameters
            loss.backward()
            # r_err.backward()   # Debug to check that the radius is being updated
            opt.step()
        opt.zero_grad()

        # Rebuild the mesh using the new radius
        vertices, faces = sphere_opt.build_mesh()

        # Render new images - colour needs cloning as Mitsuba doesn't map nn.Parameters
        img_i = render_image(vertices, faces, sphere_opt.colour.clone(), seed=i)
        img_target = mi.render(scene_target, seed=i).torch()
        if multiscale:
            img_target = to_multiscale(img_target, blur)
            img_i = to_multiscale(img_i, blur)
            if i == 0:
                resolution_pyramid = [t.shape[0] for t in img_target[::-1]]
                logger.info(f'Resolution pyramid has {len(img_target)} levels: '
                            f'{", ".join([str(res) for res in resolution_pyramid])}')

        # Calculate losses
        # loss = torch.mean((img_i - image_target)**2)
        losses = [torch.mean((a - b)**2) for a, b in zip(img_target, img_i)]
        loss = sum(losses)
        s_diff = torch.abs(sphere_opt.scale - sphere_target.scale)
        o_diff = torch.norm(sphere_opt.origin - sphere_target.origin)
        r_diff = torch.norm(sphere_opt.rotvec - sphere_target.rotvec)
        c_diff = torch.norm(sphere_opt.colour - sphere_target.colour)
        losses.append(loss.item())
        s_diffs.append(s_diff.item())
        o_diffs.append(o_diff.item())
        r_diffs.append(r_diff.item())
        c_diffs.append(c_diff.item())
        logger.info(
            f'Iteration {i} ' +
            f'Loss: {loss.item():.3E} ' +
            f's-error: {s_diff.item():.3E} (s={sphere_opt.scale.item():.2f}) ' +
            f'o-error: {o_diff.item():.3E} (o=[' + ','.join([f'{v:.2f}' for v in sphere_opt.origin]) + ') ' +
            f'r-error: {r_diff.item():.3E} (r=[' + ','.join([f'{v:.2f}' for v in sphere_opt.rotvec]) + ') ' +
            f'c-error: {c_diff.item():.3E} (c=[' + ','.join([f'{v:.2f}' for v in sphere_opt.colour]) + ')'
        )

        # Plot
        if i % plot_freq == 0:
            plot(img_i)

    # Plot the final scene comparison
    plot(img_i)

    # Plot the losses
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.plot(losses)
    ax.set_title('Image Loss (L2)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    ax.set_yscale('log')
    ax = axes[1]
    ax.plot(s_diffs, label='Scale')
    ax.plot(o_diffs, label='Origin')
    ax.plot(r_diffs, label='Rotation')
    ax.plot(c_diffs, label='Colour')
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


if __name__ == '__main__':
    # plot_scene()
    optimise_scene()
