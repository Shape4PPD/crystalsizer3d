from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from kornia.geometry import quaternion_to_rotation_matrix, rotation_matrix_to_axis_angle, rotation_matrix_to_quaternion
from torch import Tensor

from crystalsizer3d.util.utils import to_numpy


def normalise(v: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Normalise an array along its final dimension.
    """
    if isinstance(v, Tensor):
        return normalise_pt(v)
    else:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


@torch.jit.script
def normalise_pt(v: Tensor) -> Tensor:
    """
    Normalise a tensor along its final dimension.
    """
    return v / torch.norm(v, dim=-1, keepdim=True)


@torch.jit.script
def line_equation_coefficients(
        p1: Tensor,
        p2: Tensor,
        perpendicular: bool = False,
        eps: float = 1e-6
) -> Tensor:
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


@torch.jit.script
def line_intersection(
        l1: Tensor,
        l2: Tensor
) -> Optional[Tensor]:
    """
    Calculate the intersection point of two lines in the form ax + by + c = 0.
    """
    assert l1.shape == l2.shape == (3,), 'Invalid line equation coefficients.'
    a1, b1, c1 = l1[0], l1[1], l1[2]
    a2, b2, c2 = l2[0], l2[1], l2[2]

    # Compute determinant
    det = a1 * b2 - a2 * b1

    # Check if lines are parallel
    if det.abs() < 1e-6:
        return None  # Lines are parallel, no intersection

    # Calculate intersection point
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det

    return torch.stack([x, y])


@torch.jit.script
def is_point_in_bounds(p: Tensor, bounds: List[Tensor], eps: float = 1e-6) -> bool:
    """
    Check if point p lies within the bounds defined by bounds.
    """
    bounds = torch.stack(bounds)
    mins = bounds.amin(dim=0)
    maxs = bounds.amax(dim=0)
    min_x, min_y = mins[0], mins[1]
    max_x, max_y = maxs[0], maxs[1]
    return bool((min_x - eps <= p[0] <= max_x + eps)
                and (min_y - eps <= p[1] <= max_y + eps))


@torch.jit.script
def polygon_area(vertices: Tensor) -> Tensor:
    """
    Calculate the area of a 2D polygon given its vertices.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 2, f'Invalid vertices shape: {vertices.shape}'
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Calculate the area using the shoelace formula
    area = 0.5 * torch.abs(torch.dot(x, torch.roll(y, -1)) - torch.dot(y, torch.roll(x, -1)))

    return area


@torch.jit.script
def point_in_polygon(points: Tensor, polygon: Tensor, eps: float = 1e-6):
    """
    Check if points are inside a convex polygon using a vectorised ray-casting algorithm.
    Args:
        points: Tensor of shape (N, 2) representing N points.
        polygon: Tensor of shape (M, 2) representing the polygon vertices.
        eps: Small tolerance for floating-point comparisons.

    Returns:
        Tensor of shape (N,) where each entry is True if the point is inside the polygon.
    """
    edges_poly = torch.stack([polygon, torch.roll(polygon, -1, dims=0)], dim=1)

    # Check the ray intersection count for each point
    p = points.unsqueeze(1)  # (N, 1, 2)
    v1 = edges_poly[:, 0]  # (M, 2)
    v2 = edges_poly[:, 1]  # (M, 2)

    # Vectorised ray-casting logic
    cond1 = (v1[:, 1] > p[:, :, 1]) != (v2[:, 1] > p[:, :, 1])
    slope = (v2[:, 0] - v1[:, 0]) / (v2[:, 1] - v1[:, 1] + eps)
    cond2 = p[:, :, 0] < (v1[:, 0] + slope * (p[:, :, 1] - v1[:, 1]))

    intersect_count = (cond1 & cond2).sum(dim=1)
    return intersect_count % 2 == 1  # True if odd number of intersections


@torch.jit.script
def merge_vertices(vertices: Tensor, epsilon: float = 1e-3) -> Tuple[Tensor, Tensor]:
    """
    Cluster a set of vertices based on spatial quantisation.
    """
    device = vertices.device

    # Subtract the mean
    vertices_og = vertices
    vertices = vertices - vertices.mean(dim=0)

    # Compute hash key for each point based on spatial quantisation
    hash_keys = torch.round(vertices / epsilon)

    # Use hash_keys to create a unique identifier for each cluster
    unique_clusters, cluster_indices = torch.unique(hash_keys, dim=0, return_inverse=True)
    if len(unique_clusters) == len(vertices):
        return vertices_og, torch.arange(len(vertices), device=device)

    # Calculate mean of points within each cluster
    clustered_vertices = torch.zeros_like(unique_clusters, dtype=vertices.dtype)
    counts = torch.zeros(len(unique_clusters), dtype=torch.int64, device=device)

    # Accumulate points and counts for each cluster
    clustered_vertices.scatter_add_(0, cluster_indices.unsqueeze(1).expand_as(vertices), vertices_og)
    counts.scatter_add_(0, cluster_indices, torch.ones_like(cluster_indices))

    # Avoid division by zero
    counts[counts == 0] = 1

    # Calculate mean for each cluster
    cluster_centroids = clustered_vertices / counts.unsqueeze(1).expand_as(clustered_vertices)

    return cluster_centroids, cluster_indices


@torch.jit.script
def line_segment_intersections(edges1: Tensor, edges2: Tensor, eps: float = 1e-6):
    """
    Compute the intersection points of multiple line segments in batch.
    Args:
        edges1: Tensor of edges with shape (N, 2, 2).
        edges2: Tensor of edges with shape (M, 2, 2).
        eps: Small tolerance for floating-point comparisons.

    Returns:
        Tensor of shape (N, M, 2) with intersection points or NaNs if no intersection.
    """
    p = edges1[:, 0][:, None]  # (N, 1, 2)
    r = (edges1[:, 1] - edges1[:, 0])[:, None]  # (N, 1, 2)
    q = edges2[:, 0][None, ...]  # (1, M, 2)
    s = (edges2[:, 1] - edges2[:, 0])[None, ...]  # (1, M, 2)

    r_cross_s = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]  # (N, M)
    pq = q - p  # (N, M, 2)

    t = (pq[..., 0] * s[..., 1] - pq[..., 1] * s[..., 0]) / (r_cross_s + eps)  # (N, M)
    u = (pq[..., 0] * r[..., 1] - pq[..., 1] * r[..., 0]) / (r_cross_s + eps)  # (N, M)

    intersect_mask = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & (r_cross_s.abs() > eps)
    intersections = p + t[..., None] * r  # (N, M, 2)
    intersections[~intersect_mask] = torch.nan  # NaN for no intersections

    return intersections


@torch.jit.script
def line_segments_in_polygon(
        edges: Tensor,
        polygon: Tensor,
        tol: float = 1e-2,
        eps: float = 1e-6
) -> Tensor:
    """
    Return the line segments from the input edges that are contained within or intersect the polygon.

    Args:
        edges: Tensor of shape (N, 2, 2) representing N line segments.
        polygon: Tensor of shape (M, 2) representing the vertices of the convex polygon.
        tol: Tolerance for considering nearby points as duplicates.
        eps: Small tolerance for floating-point comparisons.

    Returns:
        A tensor containing line segments that are inside or intersect the polygon.
    """
    polygon = merge_vertices(polygon, epsilon=tol)[0]  # (M, 2)
    p2 = polygon - polygon.mean(dim=0)
    angles = torch.atan2(p2[:, 1], p2[:, 0])
    sorted_idxs = torch.argsort(angles)
    polygon = polygon[sorted_idxs]

    # Check if edge endpoints are inside the polygon
    endpoints_inside = point_in_polygon(edges.view(-1, 2), polygon)  # (N * 2,)
    edge_start_in = endpoints_inside[::2]
    edge_end_in = endpoints_inside[1::2]

    # Get polygon edges
    polygon_edges = torch.stack([polygon, torch.roll(polygon, -1, dims=0)], dim=1)  # (M, 2, 2)

    # Calculate intersections of edges with polygon edges, filtering out the invalid ones
    intersections = line_segment_intersections(edges, polygon_edges, eps=eps)  # (N, M, 2)
    valid_intersections = ~torch.isnan(intersections[..., 0])  # (N, M)

    # Assemble the valid segments for each edge
    segments_list = []
    for i in range(edges.shape[0]):
        segment_points = []

        # If the start point is inside the polygon, add it
        if edge_start_in[i]:
            segment_points.append(edges[i, 0])

        # Add valid intersection points
        valid_points = intersections[i][valid_intersections[i]]
        segment_points.extend([p for p in valid_points])

        # If the end point is inside the polygon, add it
        if edge_end_in[i]:
            segment_points.append(edges[i, 1])

        # Remove duplicates considering point tolerance
        if len(segment_points) > 0:
            segment = torch.stack(segment_points)
            is_close = torch.isclose(segment[None, :], segment[:, None], atol=tol).all(dim=-1)
            unique_mask = ~is_close.triu(1).any(dim=1)
            segment = segment[unique_mask]

            # Store the segment if it has two distinct points
            if len(segment) == 2:
                segments_list.append(segment)

    # Stack the segments into a single tensor
    if len(segments_list) > 0:
        segments = torch.stack(segments_list)
    else:
        segments = torch.empty((0, 2, 2), device=edges.device)

    # Remove duplicate segments (both endpoints match to within tolerance)
    if len(segments) > 1:
        rounded_segments = torch.round(segments / tol) * tol
        _, unique_indices = torch.unique(rounded_segments.view(-1, 4), dim=0, return_inverse=True)
        segments = segments[torch.unique(unique_indices)]

    return segments


@torch.jit.script
def geodesic_distance(R1: Tensor, R2: Tensor, eps: float = 1e-4) -> Tensor:
    """
    Compute the geodesic distance between two rotation matrices.
    """
    R = torch.matmul(R2, R1.transpose(-2, -1))
    trace = torch.einsum('...ii', R)
    trace_temp = (trace - 1) / 2
    trace_temp = torch.clamp(trace_temp, -1 + eps, 1 - eps)
    theta = torch.acos(trace_temp)
    return theta


def get_closest_rotation(
        R0: Union[np.ndarray, Tensor],
        rotation_group: Tensor,
) -> Union[np.ndarray, Tensor]:
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


@torch.jit.script
def _compute_rotation_matrix(axis_angle: Tensor, theta2: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Taken from kornia/geometry/conversions.py, adapted to be jit compatible.
    """
    # We want to be careful to only evaluate the square root if the
    # norm of the axis_angle vector is greater than zero. Otherwise
    # we get a division by zero.
    k_one = 1.0
    theta = torch.sqrt(theta2)
    wxyz = axis_angle / (theta + eps)
    wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    r00 = cos_theta + wx * wx * (k_one - cos_theta)
    r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
    r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
    r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
    r11 = cos_theta + wy * wy * (k_one - cos_theta)
    r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
    r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
    r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
    r22 = cos_theta + wz * wz * (k_one - cos_theta)
    rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
    return rotation_matrix.view(-1, 3, 3)


@torch.jit.script
def _compute_rotation_matrix_taylor(axis_angle: Tensor) -> Tensor:
    """
    Taken from kornia/geometry/conversions.py, adapted to be jit compatible.
    """
    rx, ry, rz = torch.chunk(axis_angle, 3, dim=1)
    k_one = torch.ones_like(rx)
    rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
    return rotation_matrix.view(-1, 3, 3)


@torch.jit.script
def axis_angle_to_rotation_matrix(axis_angle: Tensor) -> Tensor:
    """
    Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.
     -- Taken from kornia/geometry/conversions.py, adapted to be jit compatible
    """
    if not isinstance(axis_angle, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(axis_angle)}")

    if not axis_angle.shape[-1] == 3:
        raise ValueError(f"Input size must be a (*, 3) tensor. Got {axis_angle.shape}")

    # stolen from ceres/rotation.h
    _axis_angle = torch.unsqueeze(axis_angle, dim=1)
    theta2 = torch.matmul(_axis_angle, _axis_angle.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(axis_angle, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(axis_angle)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (~mask).type_as(theta2)

    # create output pose matrix with masked values
    rotation_matrix = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx3x3


@torch.jit.script
def align_points_to_xy_plane(
        points: Tensor,
        centroid: Optional[Tensor] = None,
        tol: float = 1e-2
) -> Tuple[Tensor, bool]:
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
        return points, False

    cross_idx = 1
    while cross_idx < len(points):
        # Step 3: Compute normal vector of the plane
        normal_vector = torch.cross(translated_points[0], translated_points[cross_idx], dim=0)
        normal_vector = normalise_pt(normal_vector)
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=centroid.dtype, device=centroid.device)

        # If the normal vector is already aligned with the z-axis, return the points
        if torch.allclose(normal_vector, z_axis):
            return translated_points, True

        # If the normal vector is aligned with the negative z-axis, rotate around the x-axis
        if torch.allclose(normal_vector, -z_axis):
            rotated_points = translated_points
            rotated_points[:, 0] = -rotated_points[:, 0]
            return rotated_points, True

        # Step 4: Compute rotation matrix
        rotation_axis = normalise_pt(torch.cross(normal_vector, z_axis, dim=0))
        rotation_angle = torch.acos(torch.dot(normal_vector, z_axis))
        rotvec = rotation_axis * rotation_angle
        R = axis_angle_to_rotation_matrix(rotvec[None, :]).squeeze(0)

        # Step 5: Apply rotation to points
        rotated_points = (R @ translated_points.T).T

        # Step 6: Verify that the points are aligned to the xy plane, if not try calculating the normal vector with a different point
        if rotated_points[:, 2].abs().amax() < torch.norm(rotated_points, dim=-1).max() * tol:
            return rotated_points, True
        cross_idx += 1

    return points, False


@torch.jit.script
def rotate_2d_points_to_square(
        points: Tensor,
        max_adjustments: int = 20,
        tol: float = 5 * (np.pi / 180)
) -> Tensor:
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


@torch.jit.script
def calculate_relative_angles(vertices: Tensor, centroid: Tensor, tol: float = 1e-2) -> Tensor:
    """
    Calculate the angles of a set of vertices relative to the centroid.
    """
    vertices, success = align_points_to_xy_plane(vertices, centroid, tol=tol)
    if not success:
        angles = torch.linspace(0, 2 * torch.pi, steps=len(vertices) + 1,
                                device=vertices.device, dtype=vertices.dtype)[:-1]
    else:
        angles = torch.atan2(vertices[:, 1], vertices[:, 0])
    return angles


@torch.jit.script
def sort_face_vertices(vertices: Tensor) -> Tensor:
    """
    Sort the vertices of a face in clockwise order.
    """
    device = vertices.device
    dtype = vertices.dtype

    # Calculate the angles of each vertex relative to the centroid
    centroid = torch.mean(vertices, dim=0)
    angles = calculate_relative_angles(vertices, centroid)

    # Sort the vertices based on the angles
    sorted_idxs = torch.argsort(angles)
    sorted_vertices = vertices[sorted_idxs]

    # Flip the order if the normal is pointing inwards
    normal = torch.zeros(3, device=device, dtype=dtype)
    normal_norm = normal.norm()
    largest_normal = normal
    largest_normal_norm = normal_norm
    i = 0
    while normal_norm < 1e-3 and i < len(sorted_vertices) - 1:
        normal = torch.cross(sorted_vertices[i] - centroid, sorted_vertices[i + 1] - centroid, dim=0)
        normal_norm = normal.norm()
        if normal_norm > largest_normal_norm:
            largest_normal = normal
            largest_normal_norm = normal_norm
        i += 1
    if normal_norm < 1e-3:
        normal = largest_normal
    if torch.dot(normal, centroid) < 0:
        sorted_idxs = sorted_idxs.flip(0)

    return sorted_idxs
