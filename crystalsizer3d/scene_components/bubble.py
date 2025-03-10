from typing import List

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
from torch import nn

from crystalsizer3d import MI_CPU_VARIANT, USE_CUDA
from crystalsizer3d.util.ico_sphere import ico_sphere

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
else:
    mi.set_variant(MI_CPU_VARIANT)

ico_sphere_cache = {}


def load_ico_sphere(level: int = 0):
    """
    Load an icosphere from the cache, generating it if necessary.
    """
    if level not in ico_sphere_cache:
        ico_sphere_cache[level] = ico_sphere(level=level)
    return ico_sphere_cache[level].clone()


class Bubble(nn.Module):
    vertices: torch.Tensor
    faces: torch.Tensor

    def __init__(
            self,
            shape_name: str = 'bubble',
            origin: List[float] = [0, 0, 0],
            scale: float = 1.0,
            colour: List[float] = [1, 1, 1],
            roughness: float = 0.05,
            ior: float = 1.5,
            resolution_level: int = 4,
    ):
        """
        Create a sphere with the given scale, origin, and rotation.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=True)
        self.origin = nn.Parameter(torch.tensor(origin, dtype=torch.float32), requires_grad=True)
        self.colour = nn.Parameter(torch.tensor(colour, dtype=torch.float32), requires_grad=True)
        self.roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float32), requires_grad=True)
        self.ior = nn.Parameter(torch.tensor(ior, dtype=torch.float32), requires_grad=True)
        self.resolution_level = resolution_level

        # Buffers for the vertices and faces
        self.register_buffer('vertices', torch.empty(0))
        self.register_buffer('faces', torch.empty(0))

        # Parameter keys
        self.SHAPE_NAME = shape_name
        self.VERTEX_KEY = self.SHAPE_NAME + '.vertex_positions'
        self.FACES_KEY = self.SHAPE_NAME + '.faces'
        self.BSDF_KEY = self.SHAPE_NAME + '.bsdf'
        self.COLOUR_KEY = self.BSDF_KEY + '.reflectance.value'

        # Build the mesh
        self.build_mesh()

    def build_mesh(self):
        """
        Create a sphere mesh with the given origin and radius.
        """
        # Build basic sphere
        sphere = load_ico_sphere(level=self.resolution_level).to(self.origin.device)
        vertices = sphere.verts_packed()
        faces = sphere.faces_packed()

        # Apply scaling
        vertices = vertices * self.scale

        # Apply translation
        vertices = vertices + self.origin

        # Store the pytorch vertices and faces
        self.vertices = vertices.clone()
        self.faces = faces.clone()

    def build_mitsuba_mesh(self) -> mi.Mesh:
        """
        Convert the Bubble object into a Mitsuba mesh.
        """
        nv, nf = len(self.vertices), len(self.faces)
        vertices = mi.TensorXf(self.vertices)
        faces = mi.TensorXi64(self.faces)

        # Set up the material properties
        bsdf = {
            'type': 'roughdielectric',
            'distribution': 'beckmann',
            'alpha': self.roughness.item(),
            'int_ior': self.ior.item(),
        }
        props = mi.Properties()
        props[self.BSDF_KEY] = mi.load_dict(bsdf)

        # Construct the mitsuba mesh and set the vertex positions and faces
        mesh = mi.Mesh(
            self.SHAPE_NAME,
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


def make_bubbles(
        n_bubbles: int,
        min_x: float = 0.,
        max_x: float = 0.,
        min_y: float = 0.,
        max_y: float = 0.,
        min_z: float = 0.,
        max_z: float = 0.,
        min_scale: float = 1.,
        max_scale: float = 1.,
        min_roughness: float = 0.05,
        max_roughness: float = 0.2,
        min_ior: float = 1.1,
        max_ior: float = 1.8,
        device: torch.device = torch.device('cpu'),
) -> List[Bubble]:
    """
    Generate some bubbles to go in the growth cell.
    """
    bubbles = []
    for i in range(n_bubbles):
        bubble = Bubble(
            shape_name=f'bubble_{i}',
            origin=[
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y),
                np.random.uniform(min_z, max_z),
            ],
            scale=np.random.uniform(min_scale, max_scale),
            roughness=np.random.uniform(min_roughness, max_roughness),
            ior=np.random.uniform(min_ior, max_ior),
        )
        bubble.to(device)
        bubbles.append(bubble)
    return bubbles
