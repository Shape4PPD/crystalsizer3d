from typing import Tuple

import torch
from kornia.geometry import axis_angle_to_rotation_matrix, quaternion_to_rotation_matrix

from crystalsizer3d.util.geometry import merge_vertices


def calculate_polyhedral_vertices(
        distances: torch.Tensor,
        scale: torch.Tensor,
        origin: torch.Tensor,
        rotation: torch.Tensor,
        symmetry_idx: torch.Tensor,
        plane_normals: torch.Tensor,
        tol: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Take the face normals and distances and calculate where the vertices and edges lie.
    """
    assert distances.ndim == 2, 'Distances must be a 2D tensor.'
    device = distances.device
    bs = distances.shape[0]

    # Expand the distances to take account of symmetries
    d = distances.clone()
    d[d < 0] = 1e8  # Set negative distances to a large value
    d = d.clip(0, None)
    if symmetry_idx is not None:
        d = d[:, symmetry_idx]

    # Calculate all intersection points of all combinations of 3 planes
    all_combinations = torch.combinations(torch.arange(len(plane_normals), device=device), r=3)
    N = plane_normals[all_combinations]  # shape: (C, 3, 3)

    # Check if the planes form a basis in 3-space and remove any invalid combinations
    det = torch.det(N)
    valid_combos = torch.abs(det) > 1e-6
    N = N[valid_combos]

    # Solve the systems of equations
    A = N[None, ...].expand(bs, *N.shape).reshape(-1, 3, 3)
    b = d[:, all_combinations[valid_combos]].reshape(-1, 3)
    intersection_points = torch.linalg.solve(A, b)
    intersection_points = intersection_points.reshape(bs, -1, 3)

    # Restrict to points that are in the polyhedron and exclude any that should have grown to infinity
    T = intersection_points @ plane_normals.T
    is_interior = torch.all((T <= d[:, None] + tol) & (T.abs() < 1e4), dim=2)

    # Pad the vertices so that all batch entries have the same number of vertices
    vertices = []
    for i in range(bs):
        v = intersection_points[i, is_interior[i]]
        vc, _ = merge_vertices(v, tol)
        vertices.append(vc)
    n_vertices = torch.tensor([len(v) for v in vertices], device=device)
    max_vertices = n_vertices.amax()
    vertices = torch.stack([
        torch.cat([v, torch.zeros(max_vertices - len(v), 3, device=device)])
        for v in vertices
    ])

    # Apply the rotation
    if rotation.shape[-1] == 3:
        if torch.allclose(rotation, torch.zeros(3, device=device)):
            rotation = rotation + torch.randn(3, device=device) * 1e-8
        R = axis_angle_to_rotation_matrix(rotation)
    elif rotation.shape[-1] == 4:
        R = quaternion_to_rotation_matrix(rotation)
    else:
        raise ValueError(f'Unsupported rotation shape: {rotation.shape}')
    vertices = vertices @ R.transpose(1, 2)

    # Apply the scaling
    scale = scale.clip(1e-6, None)
    vertices = vertices * scale[:, None, None]

    # Apply the origin translation
    vertices = vertices + origin[:, None, :]

    # Re-zero the vertices that were padded
    for i in range(bs):
        vertices[i, n_vertices[i]:] = torch.zeros(3, device=device)

    return vertices, n_vertices
