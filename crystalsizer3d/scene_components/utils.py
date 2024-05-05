from typing import Optional

import drjit as dr
import mitsuba as mi
import numpy as np
import torch

from crystalsizer3d.crystal import Crystal


class RenderError(RuntimeError):
    def __init__(self, message: str, idx: int = None):
        super().__init__(message)
        self.idx = idx


def build_crystal_mesh(
        crystal: Crystal,
        material_bsdf: dict,
        shape_name: str = 'crystal',
        bsdf_key: str = 'crystal.bsdf',
) -> mi.Mesh:
    """
    Convert the Crystal object into a Mitsuba mesh.
    """
    # Build the mesh in pytorch and convert the parameters to Mitsuba format
    vertices, faces = crystal.build_mesh()
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
    params = mi.traverse(mi_scene)
    sensor = mi_scene.sensors()[0]
    film = sensor.film()

    # Create the projection matrix
    prj = mi.perspective_projection(film.size(), film.crop_size(), film.crop_offset(), params['sensor.x_fov'],
                                    sensor.near_clip(), sensor.far_clip())

    # Get the inverse camera world transform
    wti = sensor.world_transform().inverse()

    # Project the points
    if points.ndim == 1:
        points = points[None, :]
    points_homogeneous = torch.cat([points, torch.ones(len(points), 1, device=device)], dim=1)
    hpp = points_homogeneous @ torch.tensor((prj @ wti).matrix, device=device)[0].T
    pp = hpp[:, :3] / hpp[:, 3][:, None]
    dim = torch.tensor(film.crop_size(), device=device)[None, :]
    uv = pp[:, :2] * dim
    return uv


def render_crystal_scene(
        crystal: Optional[Crystal] = None,
        scene: Optional['Scene'] = None,
        **kwargs
) -> np.ndarray:
    """
    Render a crystal scene.
    """
    from crystalsizer3d.scene_components.scene import Scene
    assert not (crystal is None and scene is None), 'Either a crystal or scene must be provided.'
    assert not (crystal is not None and scene is not None), 'Only one of a crystal or scene can be provided.'
    if crystal is not None:
        scene = Scene(crystal=crystal, **kwargs)
    img = scene.render()
    return img
