from typing import Tuple

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import torch
from pytorch3d.utils import ico_sphere
from torch import nn

from crystalsizer3d import ROOT_PATH

USE_CUDA = 1

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('llvm_ad_rgb')
    device = torch.device('cpu')

from mitsuba import ScalarTransform4f as T

SCENE_PATH = ROOT_PATH / 'tmp' / 'mitsuba' / 'test'

SHAPE_NAME = 'sphere'
VERTEX_KEY = SHAPE_NAME + '.vertex_positions'
FACES_KEY = SHAPE_NAME + '.faces'
BSDF_KEY = SHAPE_NAME + '.bsdf'


class Sphere(nn.Module):
    def __init__(self, scale: float = 1.0):
        """
        Create a sphere with the given scale, origin, and rotation.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=True)

    def build_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a sphere mesh with the given scale.
        """
        sphere = ico_sphere(level=1, device=self.scale.device)
        vertices = sphere.verts_packed()
        faces = sphere.faces_packed()
        vertices = vertices * self.scale
        return vertices, faces


def build_mitsuba_mesh(shape: Sphere, bsdf: dict) -> mi.Mesh:
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
    props['bsdf'] = mi.load_dict(bsdf)

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


def create_scene(shape, spp, res, integrator):
    return mi.load_dict({
        'type': 'scene',
        'integrator': integrator,
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': T.look_at(target=[0, 0, 0], origin=[0, 0, 10], up=[0, 1, 0]),
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
            'type': 'point',
            'to_world': T.look_at(
                origin=[0, 0, 10],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'intensity': {
                'type': 'spectrum',
                'value': 300.0,
            }
        },
        SHAPE_NAME: shape
    })


def shape_optimisation(
        opt_with: str = 'mitsuba',
        integrator: dict = {'type': 'path'},
        bsdf: dict = {'type': 'diffuse'}
):
    print(f'----- Running optimisation for {opt_with} and integrator {integrator} ------')

    # Parameters
    spp = 256
    res = 100
    lr = 0.1
    n_iterations = 2

    @dr.wrap_ad(source='torch', target='drjit')
    def render_image_wrapper(vertices, faces, seed=1):
        params[VERTEX_KEY] = dr.ravel(vertices)
        params[FACES_KEY] = dr.ravel(faces)
        params.update()
        return mi.render(scene, params, seed=seed)

    # Target shape
    target_shape = {
        'type': 'ply',
        'filename': str(SCENE_PATH / 'meshes' / 'suzanne.ply'),
        'bsdf': bsdf
    }

    # Optimisable shape
    sphere = Sphere(scale=2)
    sphere.to(device)

    # Make scenes
    scene_target = create_scene(target_shape, spp, res, integrator)
    scene = create_scene(build_mitsuba_mesh(sphere, bsdf), spp, res, integrator)
    params = mi.traverse(scene)
    img_target = mi.render(scene_target)
    img_init = mi.render(scene)
    img_i = mi.render(scene)

    # Optimise
    if opt_with == 'mitsuba':
        opt = mi.ad.Adam(lr=lr)
        opt[VERTEX_KEY] = params[VERTEX_KEY]
    else:
        img_target = img_target.torch()
        img_init = img_init.torch()
        opt = torch.optim.Adam(sphere.parameters(), lr=lr, weight_decay=0)

    for i in range(n_iterations):
        if opt_with == 'mitsuba':
            params.update(opt)
            img_i = mi.render(scene, params, seed=i)
            loss = dr.mean(dr.sqr(img_i - img_target))
            loss_val = float(loss[0])
            dr.backward(loss)
            opt.step()
        else:
            opt.zero_grad()
            v, f = sphere.build_mesh()
            img_i = render_image_wrapper(v, f, seed=i)
            loss = ((img_i - img_target)**2).mean()
            loss.backward()
            loss_val = loss.item()
            opt.step()
        print(f'Iteration {1 + i}: Loss = {loss_val:6f}')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(4, 2))
    for ax, img in zip(axes, [img_target, img_init, img_i]):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        else:
            img = img.numpy()
        img = img.clip(0, 1)
        ax.imshow(img)
        ax.axis('off')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    # exit()

    print(f'---- Optimisation complete for {opt_with} and integrator {integrator} ----')
    print('')


if __name__ == '__main__':
    # ---- CPU + Mitsuba
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'direct'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'path'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'volpath'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'volpathmis'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'prb'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'direct_projective'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'prb_projective'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'prbvolpath'}) # E1

    # ---- CPU + PyTorch
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'direct'})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'path'})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'volpath'})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'volpathmis'})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'prb'})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'direct_projective'})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'direct_projective', 'sppi': 0})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'prb_projective'})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'prb_projective', 'sppi': 0})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'prbvolpath'}) # E1

    # ---- CUDA + Mitsuba
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'direct'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'path'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'volpath'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'volpathmis'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'prb'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'direct_projective'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'prb_projective'})
    # shape_optimisation(opt_with='mitsuba', integrator={'type': 'prbvolpath'}) # E1

    # ---- CUDA + PyTorch: Lots of errors!!
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'direct'})
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'path'})  # E2
    # shape_optimisation(opt_with='pytorch', integrator={'type': 'volpath'})
    shape_optimisation(opt_with='pytorch', integrator={'type': 'volpathmis'})  # E2
    shape_optimisation(opt_with='pytorch', integrator={'type': 'prb'})
    shape_optimisation(opt_with='pytorch', integrator={'type': 'direct_projective'})  # E0
    shape_optimisation(opt_with='pytorch', integrator={'type': 'direct_projective', 'sppi': 0})
    shape_optimisation(opt_with='pytorch', integrator={'type': 'prb_projective'})  # E0
    shape_optimisation(opt_with='pytorch', integrator={'type': 'prb_projective', 'sppi': 0})
    shape_optimisation(opt_with='pytorch', integrator={'type': 'prbvolpath'})  # E1

    exit()

    # ---- roughdielectric BSDF tests...
    bsdf = {
        'type': 'roughdielectric',
        'distribution': 'beckmann',
        'alpha': 0.1,
        'int_ior': 1.5,
    }

    # ---- CPU + Mitsuba + roughdielectric
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'direct'})
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'path'})  # Very slow!
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'volpath'})  # Very slow!
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'volpathmis'})  # Very slow!
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'prb'})
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'direct_projective'})
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'prb_projective'})
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'prbvolpath'}) # E1

    # ---- CPU + PyTorch + roughdielectric
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'direct'})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'path'})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'volpath'})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'volpathmis'})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'prb'})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'direct_projective'})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'direct_projective', 'sppi': 0})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'prb_projective'})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'prb_projective', 'sppi': 0})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'prbvolpath'}) # E1

    # ---- CUDA + Mitsuba + roughdielectric
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'direct'})
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'path'})
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'volpath'}) #E3
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'volpathmis'})  todo!
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'prb'})
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'direct_projective'})
    shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'prb_projective'})
    # shape_optimisation(opt_with='mitsuba', bsdf=bsdf, integrator={'type': 'prbvolpath'})  # E1

    # ---- CUDA + PyTorch + roughdielectric
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'direct'})
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'path'})  # E2
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'volpath'})  # todo
    # shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'volpathmis'})  # todo
    shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'prb'})
    shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'direct_projective'})  # E0
    shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'direct_projective', 'sppi': 0})
    shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'prb_projective'})  # E0
    shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'prb_projective', 'sppi': 0})
    shape_optimisation(opt_with='pytorch', bsdf=bsdf, integrator={'type': 'prbvolpath'})  # E1

'''
E0: Process finished with exit code 139 (interrupted by signal 11:SIGSEGV)
E1: drjit.Exception: loop_process_state(): one of the supplied loop state variables of type Float is attached to the AD graph (i.e., grad_enabled(..) is true). However, propagating derivatives through multiple iterations of a recorded loop is not supported (and never will be). Please see the documentation on differentiating loops for details and suggested alternatives.
E2: Critical Dr.Jit compiler failure: cuda_check(): API error 0700 (CUDA_ERROR_ILLEGAL_ADDRESS): "an illegal memory access was encountered" in /project/ext/drjit-core/src/init.cpp:454.
E3: Process finished with exit code 143 (interrupted by signal 15:SIGTERM)

'''
