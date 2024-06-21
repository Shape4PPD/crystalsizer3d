from typing import Optional, Tuple

import drjit as dr
import mitsuba as mi
import torch
import trimesh.remesh
from trimesh import Trimesh

from crystalsizer3d.crystal import Crystal
from crystalsizer3d.scene_components.textures import NoiseTexture
from crystalsizer3d.util.utils import to_numpy


class RenderError(RuntimeError):
    def __init__(self, message: str, idx: int = None):
        super().__init__(message)
        self.idx = idx


def build_crystal_mesh(
        crystal: Crystal,
        material_bsdf: dict,
        shape_name: str = 'crystal',
        bsdf_key: str = 'crystal.bsdf',
        remesh_max_edge: Optional[float] = None
) -> mi.Mesh:
    """
    Convert the Crystal object into a Mitsuba mesh.
    """
    vertices, faces = crystal.build_mesh()

    # Remesh if required
    if remesh_max_edge is not None:
        m = Trimesh(vertices=to_numpy(vertices), faces=to_numpy(faces))
        v2, f2 = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=remesh_max_edge)
        vertices = torch.from_numpy(v2).to(torch.float32).to(vertices.device)
        faces = torch.from_numpy(f2).to(vertices.device)

    nv, nf = len(vertices), len(faces)
    vertices = mi.TensorXf(vertices)
    faces = mi.TensorXi64(faces)

    # Set up the material properties
    if crystal.use_bumpmap:
        bsdf = {
            'type': 'bumpmap',
            'texture': {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(mi.TensorXf(crystal.bumpmap)),
                'wrap_mode': 'clamp',
                'raw': True
            },
            'bsdf': material_bsdf
        }
    else:
        bsdf = material_bsdf
    props = mi.Properties()
    props[bsdf_key] = mi.load_dict(bsdf)

    # Construct the mitsuba mesh and set the vertex positions and faces
    mesh = mi.Mesh(
        shape_name,
        vertex_count=nv,
        face_count=nf,
        has_vertex_normals=False,
        has_vertex_texcoords=crystal.use_bumpmap,
        props=props
    )
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['faces'] = dr.ravel(faces)

    # Update the texture coordinates if a bumpmap is used
    if crystal.use_bumpmap:
        tex_coords = mi.TensorXf(crystal.uv_map)
        mesh_params['vertex_texcoords'] = dr.ravel(tex_coords)

    return mesh


def project_to_image(mi_scene: mi.Scene, points: torch.Tensor) -> torch.Tensor:
    """
    Project 3D points to the image plane.
    """
    device = points.device
    projection_matrix, crop_size = get_projection_components(mi_scene)
    projection_matrix = projection_matrix.to(device)
    if points.ndim == 1:
        points = points[None, :]
    points_homogeneous = torch.cat([points, torch.ones(len(points), 1, device=device)], dim=1)
    hpp = points_homogeneous @ projection_matrix.T
    pp = hpp[:, :3] / hpp[:, 3][:, None]
    dim = torch.tensor(crop_size, device=device)[None, :]
    uv = pp[:, :2] * dim
    return uv


projection_components_cache = {}


def get_projection_components(mi_scene: mi.Scene) -> Tuple[torch.Tensor, int]:
    """
    Get the projection matrix.
    """
    if mi_scene.ptr not in projection_components_cache:
        params = mi.traverse(mi_scene)
        sensor = mi_scene.sensors()[0]
        film = sensor.film()

        # Create the projection matrix
        prj = mi.perspective_projection(film.size(), film.crop_size(), film.crop_offset(), params['sensor.x_fov'],
                                        sensor.near_clip(), sensor.far_clip())

        # Get the inverse camera world transform
        wti = sensor.world_transform().inverse()

        # Combine to get the final projection matrix
        projection_components_cache[mi_scene.ptr] = (torch.tensor((prj @ wti).matrix)[0], film.crop_size())

    return projection_components_cache[mi_scene.ptr]
