from typing import Optional, Tuple, Union

import numpy as np
import torch
from kornia.geometry import axis_angle_to_rotation_matrix, quaternion_to_rotation_matrix, rotation_matrix_to_axis_angle, \
    rotation_matrix_to_quaternion

from crystalsizer3d.util.utils import to_numpy


def normalise(v: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalise an array along its final dimension.
    """
    if isinstance(v, torch.Tensor):
        return normalise_pt(v)
    else:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


@torch.jit.script
def normalise_pt(v: torch.Tensor) -> torch.Tensor:
    """
    Normalise a tensor along its final dimension.
    """
    return v / torch.norm(v, dim=-1, keepdim=True)


def line_equation_coefficients(
        p1: torch.Tensor,
        p2: torch.Tensor,
        perpendicular: bool = False,
        eps: float = 1e-6
) -> torch.Tensor:
    """
    Calculate the coefficients of the line that passes through p1 and p2 in the form ax + by + c = 0.
    If perpendicular is True, the coefficients of the line perpendicular to this line that passes through the midpoint of p1 and p2 are returned.
    """
    diff = p2 - p1
    midpoint = (p1 + p2) / 2
    one = torch.tensor(1., device=p1.device)
    zero = torch.tensor(0., device=p1.device)
    if perpendicular and diff[1].abs() < eps:
        return torch.stack([zero, one, -midpoint[1]])
    elif not perpendicular and diff[0].abs() < eps:
        return torch.stack([one, zero, -midpoint[0]])

    # Calculate slope (x)
    m = diff[1] / diff[0]
    if perpendicular:
        m = -1 / m

    # Calculate y-intercept (b)
    b = midpoint[1] - m * midpoint[0]

    return torch.stack([-m, one, -b])


def line_intersection(
        l1: Tuple[float, float, float],
        l2: Tuple[float, float, float]
) -> Optional[torch.Tensor]:
    """
    Calculate the intersection point of two lines in the form ax + by + c = 0.
    """
    a1, b1, c1 = l1
    a2, b2, c2 = l2

    # Compute determinant
    det = a1 * b2 - a2 * b1

    # Check if lines are parallel
    if det.abs() < 1e-6:
        return None  # Lines are parallel, no intersection

    # Calculate intersection point
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det

    return torch.stack([x, y])


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor, EPS: float = 1e-4) -> torch.Tensor:
    """
    Compute the geodesic distance between two rotation matrices.
    """
    R = torch.matmul(R2, R1.transpose(-2, -1))
    trace = torch.einsum('...ii', R)
    trace_temp = (trace - 1) / 2
    trace_temp = torch.clamp(trace_temp, -1 + EPS, 1 - EPS)
    theta = torch.acos(trace_temp)
    return theta


def get_closest_rotation(
        R0: Union[np.ndarray, torch.Tensor],
        rotation_group: torch.Tensor,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get the closest rotation in a group to a target rotation matrix.
    """
    return_numpy = False
    if isinstance(R0, np.ndarray):
        R0 = torch.from_numpy(R0)
        return_numpy = True
    R0_dev = R0.device
    if R0_dev != rotation_group.device:
        R0 = R0.to(rotation_group.device)
    mode = 'rotation_matrix'
    if R0.ndim == 1:
        if len(R0) == 4:
            mode = 'quaternion'
            R0 = quaternion_to_rotation_matrix(R0[None, ...])[0]
        elif len(R0) == 3:
            mode = 'axis_angle'
            R0 = axis_angle_to_rotation_matrix(R0[None, ...])[0]
        else:
            raise ValueError('Invalid rotation parameters.')
    assert R0.shape == (3, 3), 'Invalid rotation matrix shape.'
    assert rotation_group.ndim == 3 and rotation_group.shape[1:] == (3, 3), \
        'Invalid rotation group shape.'

    # Expand the rotation matrix to match the shape of the rotation group
    R0 = R0[None, ...].expand(len(rotation_group), -1, -1)

    # Calculate the angular differences between the target rotation and each rotation in the group
    angular_differences = geodesic_distance(R0, rotation_group)

    # Get the rotation with the smallest angular difference
    min_idx = int(torch.argmin(angular_differences))
    R_star = rotation_group[min_idx]

    # Convert the rotation to the appropriate format
    if mode == 'quaternion':
        R_star = rotation_matrix_to_quaternion(R_star[None, ...])[0]
    elif mode == 'axis_angle':
        R_star = rotation_matrix_to_axis_angle(R_star[None, ...])[0]

    # Return the result as a numpy array if the input was a numpy array
    if return_numpy:
        return to_numpy(R_star)

    # Return the result as a tensor on the original device
    return R_star.to(R0_dev)


def align_points_to_xy_plane(
        points: torch.Tensor,
        centroid: Optional[torch.Tensor] = None,
        cross_idx: int = 1,
        tol: float = 1e-3
) -> torch.Tensor:
    """
    Align a set of 3D points to the xy plane.
    """
    # Step 1: Calculate centroid
    if centroid is None:
        centroid = torch.mean(points, dim=0)

    # Step 2: Translate points to centroid
    translated_points = points - centroid

    # Check if the points are too close together
    if translated_points.norm(dim=-1, p=2).amax() < tol:
        raise ValueError('Points are too close together')

    # Step 3: Compute normal vector of the plane
    normal_vector = torch.cross(translated_points[0], translated_points[cross_idx], dim=0)
    normal_vector = normalise(normal_vector)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=centroid.dtype, device=centroid.device)

    # If the normal vector is already aligned with the z-axis, return the points
    if torch.allclose(normal_vector, z_axis):
        return translated_points

    # If the normal vector is aligned with the negative z-axis, rotate around the x-axis
    if torch.allclose(normal_vector, -z_axis):
        rotated_points = translated_points
        rotated_points[:, 0] = -rotated_points[:, 0]
        return rotated_points

    # Step 4: Compute rotation matrix
    rotation_axis = normalise(torch.cross(normal_vector, z_axis, dim=0))
    rotation_angle = torch.acos(torch.dot(normal_vector, z_axis))
    rotvec = rotation_axis * rotation_angle
    R = axis_angle_to_rotation_matrix(rotvec[None, :]).squeeze(0)

    # Step 5: Apply rotation to points
    rotated_points = (R @ translated_points.T).T

    # Step 6: Verify that the points are aligned to the xy plane, if not try calculating the normal vector with a different point
    if rotated_points[:, 2].abs().amax() > torch.norm(rotated_points, dim=-1).max() * tol:
        if cross_idx < len(points) - 1:
            return align_points_to_xy_plane(points, centroid, cross_idx + 1)
        raise ValueError('Points are not aligned to the xy plane')

    return rotated_points


# @torch.jit.script
def rotate_2d_points_to_square(
        points: torch.Tensor,
        max_adjustments: int = 20,
        tol: float = 5 * (np.pi / 180)
) -> torch.Tensor:
    """
    Rotate a set of 2D points to a square.
    """
    points = points - torch.mean(points, dim=0)
    n_adjustments = 0
    while n_adjustments < max_adjustments:
        ptp = points.amax(dim=0) - points.amin(dim=0)
        angle = torch.atan(ptp[1] / ptp[0]) - torch.pi / 4
        if angle.abs() < tol:
            break
        s, c = torch.sin(angle), torch.cos(angle)
        points = points @ torch.tensor([[c, -s], [s, c]], device=points.device)
        n_adjustments += 1
    return points


# @torch.jit.script
def calculate_relative_angles(vertices: torch.Tensor, centroid: torch.Tensor, tol: float = 1e-3) -> torch.Tensor:
    """
    Calculate the angles of a set of vertices relative to the centroid.
    """
    try:
        vertices = align_points_to_xy_plane(vertices, centroid, tol=tol)
        angles = torch.atan2(vertices[:, 1], vertices[:, 0])
    except ValueError:
        angles = torch.linspace(0, 2 * torch.pi, steps=len(vertices) + 1, device=vertices.device)[:-1]
    return angles


@torch.jit.script
def merge_vertices(vertices: torch.Tensor, epsilon: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cluster a set of vertices based on spatial quantisation.
    """
    # Subtract the mean
    vertices_og = vertices
    vertices = vertices - vertices.mean(dim=0)

    # Compute hash key for each point based on spatial quantisation
    hash_keys = torch.round(vertices / epsilon)

    # Use hash_keys to create a unique identifier for each cluster
    unique_clusters, cluster_indices = torch.unique(hash_keys, dim=0, return_inverse=True)
    if len(unique_clusters) == len(vertices):
        return vertices_og, torch.arange(len(vertices))

    # Calculate mean of points within each cluster
    clustered_vertices = torch.zeros_like(unique_clusters, dtype=vertices.dtype)
    counts = torch.zeros(len(unique_clusters), dtype=torch.int64, device=vertices.device)

    # Accumulate points and counts for each cluster
    clustered_vertices.scatter_add_(0, cluster_indices.unsqueeze(1).expand_as(vertices), vertices_og)
    counts.scatter_add_(0, cluster_indices, torch.ones_like(cluster_indices))

    # Avoid division by zero
    counts[counts == 0] = 1

    # Calculate mean for each cluster
    cluster_centroids = clustered_vertices / counts.unsqueeze(1).expand_as(clustered_vertices)

    return cluster_centroids, cluster_indices
