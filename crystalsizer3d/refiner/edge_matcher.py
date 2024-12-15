from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crystalsizer3d.refiner.keypoint_detection import to_relative_coordinates
from crystalsizer3d.util.geometry import normalise


def to_absolute_coordinates(coords: Tensor, image_size: Tensor) -> Tensor:
    """
    Convert relative coordinates to absolute coordinates.
    """
    return torch.stack([
        (coords[:, 0] * 0.5 + 0.5) * image_size[0],
        (0.5 - coords[:, 1] * 0.5) * image_size[1]
    ], dim=1)


class EdgeMatcher(nn.Module):
    def __init__(
            self,
            points_per_unit: float = 0.05,
            points_jitter: float = 0.25,
            n_samples_per_point: int = 1000,
            reach: float = 10,
            max_dist: float = 0.05
    ):
        super(EdgeMatcher, self).__init__()
        self.points_per_unit = points_per_unit
        self.points_jitter = points_jitter
        self.n_samples_per_line = n_samples_per_point
        self.reach = reach
        self.max_dist = max_dist

    def forward(self, edge_segments: Tensor, edge_image: Tensor) -> Tuple[Tensor, Tensor]:
        assert edge_image.ndim == 2, 'Distance image must be 2D.'
        image_size = edge_image.shape[0]

        # Invert the edge image, so we look for minima instead of maxima
        edge_image = -edge_image

        # Convert edge segments to absolute coordinates
        self.edge_segments = {k: to_absolute_coordinates(v.to(edge_image.device), edge_image.shape)
                              for k, v in edge_segments.items()}
        self._calculate_edge_points()
        self.edge_points_rel = to_relative_coordinates(self.edge_points, image_size)

        # Sample points along lines for all reference points
        line_points = self.sample_points_along_line_full_image(
            self.edge_points, self.edge_normals, self.n_samples_per_line, edge_image.shape
        )

        # Find local minima for all line points
        with torch.no_grad():
            minima_positions, minima_values = self.find_minima(edge_image, line_points, self.edge_points)
        deltas = minima_positions - self.edge_points

        # Put deltas into relative coordinates and calculate distances
        deltas_rel = deltas / image_size * 2
        distances = torch.norm(deltas_rel, dim=1)

        # Filter out points that are too far from the minima
        mask = distances < self.max_dist
        deltas = torch.where(mask[:, None], deltas, torch.zeros_like(deltas))
        self.deltas_rel = torch.where(mask[:, None], deltas_rel, torch.zeros_like(deltas_rel))
        distances = torch.where(mask, distances, torch.zeros_like(distances))

        # The loss is the average point-to-minima distance
        loss = torch.mean(distances)

        return loss, deltas

    def _num_points(self, start_point, end_point):
        distance = torch.norm(end_point - start_point)
        num_points = self.points_per_unit * distance
        return int(torch.ceil(num_points).item())

    def _edge_points(self, segment):
        """
        Generates points between two given points
        """
        start_point, end_point = segment
        num_points = self._num_points(start_point, end_point)
        # Generate a linear space between 0 and 1
        # generate two more points for vertices and remove
        t = torch.linspace(0, 1, steps=num_points + 2, device=segment.device)
        if self.points_jitter > 0:
            noise = torch.randn_like(t) * self.points_jitter / num_points
            noise[0] = noise[-1] = 0
            t += noise
            t = t.clamp(min=0, max=1)

        # Interpolate between start_point and end_point
        points = (1 - t).unsqueeze(1) * start_point + t.unsqueeze(1) * end_point
        # Compute the direction vector
        direction = end_point - start_point
        # Find a normal vector to the direction vector
        normal_vector = torch.tensor([-direction[1], direction[0]], dtype=torch.float32, device=segment.device)
        # remove first and last point (corners)
        normal_vector = torch.tile(normal_vector, (points.size(0), 1))
        return points[1:-1], normal_vector[1:-1]

    def _calculate_edge_points(self):
        """
        Calculate edge points, with normals for each edge segment
        """
        edge_points = []
        edge_normals = []
        for refracted_face_idx, edge_segments in self.edge_segments.items():
            if len(edge_segments) == 0:
                continue
            for segment in edge_segments:
                # get points between two end point
                points, normals = self._edge_points(segment)
                edge_points.append(points)
                edge_normals.append(normals)
        edge_points = torch.cat(edge_points, dim=0)
        edge_normals = torch.cat(edge_normals, dim=0)
        self.edge_points = edge_points
        self.edge_normals = edge_normals

    def sample_points_along_line_full_image(self, ref_points, normals, num_samples, image_shape):
        # Normalise the direction vectors
        direction_vectors_norm = normalise(normals)

        # Find the intersection points with the image boundaries for all points
        intersections = torch.stack([
            self.find_intersections_with_boundaries(
                ref_points[i], direction_vectors_norm[i], image_shape)
            for i in range(ref_points.size(0))
        ])

        # Sample points uniformly between the two intersection points for each reference point
        sampled_points = torch.linspace(0, 1, steps=num_samples, device=ref_points.device).unsqueeze(1).unsqueeze(0) * (
                intersections[:, 1] - intersections[:, 0]).unsqueeze(1) + intersections[:, 0].unsqueeze(1)

        # Clamp the points to ensure they are within image boundaries
        sampled_points = torch.stack([
            torch.clamp(sampled_points[..., 0], 0, image_shape[1] - 1),  # x-coordinates
            torch.clamp(sampled_points[..., 1], 0, image_shape[0] - 1)  # y-coordinates
        ], dim=-1)

        return sampled_points

    def find_intersections_with_boundaries(self, point, direction_vector, image_shape):
        height, width = image_shape
        # Maximum finite float value for the given dtype
        max_float = torch.finfo(direction_vector.dtype).max

        # Parametric line equation: p(t) = point + t * direction_vector
        # We need to find t such that p(t) lies on the image boundary

        # Handle intersections with left (x=0) and right (x=width-1) boundaries
        if torch.isclose(direction_vector[0], torch.tensor(0.0), atol=1e-3):
            t_left, t_right = -max_float, max_float
        else:
            # Intersection with left boundary
            t_left = (0 - point[0]) / direction_vector[0]
            # Intersection with right boundary
            t_right = (width - 1 - point[0]) / direction_vector[0]

        # Handle intersections with top (y=0) and bottom (y=height-1) boundaries
        if torch.isclose(direction_vector[1], torch.tensor(0.0), atol=1e-3):
            t_top, t_bottom = -max_float, max_float
        else:
            # Intersection with top boundary
            t_top = (0 - point[1]) / direction_vector[1]
            # Intersection with bottom boundary
            t_bottom = (height - 1 - point[1]) / direction_vector[1]

        # Take the maximum of left/top and the minimum of right/bottom
        t_min = max(t_left, t_top)
        t_max = min(t_right, t_bottom)

        # Compute the actual intersection points using the t values
        intersection_1 = point + t_min * direction_vector
        intersection_2 = point + t_max * direction_vector

        return torch.stack([intersection_1, intersection_2])

    def points_in_image(self, points, edge_image):
        # Height and Width of the edge image
        H, W = edge_image.shape

        # Normalize points to [-1, 1] range for grid_sample
        normalized_points = torch.stack([
            2.0 * points[:, 0] / (W - 1) - 1.0,  # Normalize x (width)
            2.0 * points[:, 1] / (H - 1) - 1.0  # Normalize y (height)
        ], dim=1)
        # Reshape grid to match F.grid_sample requirements
        grid = normalized_points.view(1, -1, 1, 2)

        # Clone grid to avoid inadvertently retaining the graph in subsequent usage
        grid = grid.clone()

        # Interpolate values using bilinear sampling
        interpolated_values = F.grid_sample(
            edge_image[None, None, ...],
            grid,
            mode='bilinear',
            align_corners=True
        )
        return interpolated_values

    def find_minima(self, image, sampled_points, reference_points):
        # Sample the image along the line to get the intensity values
        image_values = self.points_in_image(
            sampled_points.view(-1, 2),
            image
        ).view(
            sampled_points.size(0),
            sampled_points.size(1)
        )

        # Calculate distances from reference points to all sampled points
        dist = torch.norm(sampled_points - reference_points.unsqueeze(1), dim=2)

        # Reduce the values at the current points slightly to ensure there are always minima available
        ref_inds = torch.argmin(dist, dim=1)
        image_values[torch.arange(len(sampled_points)), ref_inds] -= 1e-6

        # Scale the intensity values by the distance from the reference points
        adjusted_values = image_values * torch.exp(-dist / self.reach)

        # Find the nearest (scaled) minima for each reference point
        minima_indices = torch.argmin(adjusted_values, dim=1)
        minima_positions = sampled_points[torch.arange(len(sampled_points)), minima_indices]
        minima_values = adjusted_values[torch.arange(len(sampled_points)), minima_indices]

        return minima_positions, minima_values
