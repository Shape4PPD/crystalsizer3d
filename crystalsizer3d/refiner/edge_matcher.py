
import torch
from torch import nn 
import torch.nn.functional as F


class EdgeMatcher(nn.Module):
    def __init__(self):
        super(EdgeMatcher, self).__init__()

    def forward(self, ref_points, normals, distance_image):        
        # Sample points along lines for all reference points
        line_points = self.sample_points_along_line_full_image(ref_points, normals, 1000, distance_image.shape[-2:])
        
        # Get reference values for all points 
        reference_values = self.points_in_image(ref_points, distance_image)
        
        # Find nearest local minima for all line points
        nearest_minima_positions, nearest_minima_values = self.find_closest_minima(distance_image, line_points, ref_points)
        
        
        # Filter out None values
        valid_indices = [i for i, val in enumerate(nearest_minima_values) if val is not None]
        if not valid_indices:
            return torch.tensor(0.0)  # Return zero if no valid minima found
        
        valid_indices = torch.tensor(valid_indices, device=distance_image.device)
        valid_ref_points = ref_points[valid_indices]
        valid_reference_values = reference_values[:,:, valid_indices]
        valid_nearest_minima_positions = torch.stack([nearest_minima_positions[i] for i in valid_indices])
        valid_nearest_minima_values = torch.stack([nearest_minima_values[i] for i in valid_indices])
        
        # Calculate distances and set them
        distances = valid_nearest_minima_positions - valid_ref_points
        
        
        # Compute the mean squared error loss
        losses = torch.mean((valid_reference_values.squeeze() - valid_nearest_minima_values.squeeze()) ** 2)
        
        # Return the mean loss across all valid points
        return torch.mean(losses), distances

    def sample_points_along_line_full_image(self, ref_points, normals, num_samples, image_shape):
        # Normalize the direction vectors
        direction_vectors_norm = normals / torch.norm(normals, dim=1, keepdim=True)

        # Find the intersection points with the image boundaries for all points
        intersections = torch.stack([
            self.find_intersections_with_boundaries(ref_points[i], direction_vectors_norm[i], image_shape)
            for i in range(ref_points.size(0))
        ])

        # Sample points uniformly between the two intersection points for each reference point
        sampled_points = torch.linspace(0, 1, steps=num_samples, device=ref_points.device).unsqueeze(1).unsqueeze(0) * (intersections[:, 1] - intersections[:, 0]).unsqueeze(1) + intersections[:, 0].unsqueeze(1)

        # Clamp the points to ensure they are within image boundaries
        sampled_points = torch.stack([
            torch.clamp(sampled_points[..., 0], 0, image_shape[1] - 1),  # x-coordinates
            torch.clamp(sampled_points[..., 1], 0, image_shape[0] - 1)   # y-coordinates
        ], dim=-1)

        return sampled_points

    def find_intersections_with_boundaries(self, point, direction_vector, image_shape):
        height, width = image_shape
        max_float = torch.finfo(direction_vector.dtype).max  # Maximum finite float value for the given dtype

        # Parametric line equation: p(t) = point + t * direction_vector
        # We need to find t such that p(t) lies on the image boundary

        # Handle intersections with left (x=0) and right (x=width-1) boundaries
        if torch.isclose(direction_vector[0], torch.tensor(0.0), atol=1e-3):
            t_left, t_right = -max_float, max_float
        else:
            t_left = (0 - point[0]) / direction_vector[0]  # Intersection with left boundary
            t_right = (width - 1 - point[0]) / direction_vector[0]  # Intersection with right boundary
        
        # Handle intersections with top (y=0) and bottom (y=height-1) boundaries
        if torch.isclose(direction_vector[1], torch.tensor(0.0), atol=1e-3):
            t_top, t_bottom = -max_float, max_float
        else:
            t_top = (0 - point[1]) / direction_vector[1]  # Intersection with top boundary
            t_bottom = (height - 1 - point[1]) / direction_vector[1]  # Intersection with bottom boundary

        # Take the maximum of left/top and the minimum of right/bottom
        t_min = max(t_left, t_top)
        t_max = min(t_right, t_bottom)

        # Compute the actual intersection points using the t values
        intersection_1 = point + t_min * direction_vector
        intersection_2 = point + t_max * direction_vector

        return torch.stack([intersection_1, intersection_2])

    def points_in_image(self, points, distance_image):
        H, W = distance_image.shape[-2], distance_image.shape[-1]  # Height and Width of the distance image
        try:
            grid = points.get_grid_sample_points(H, W)
        except:
            grid = points.clone()
            grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0  # x (width)
            grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0  # y (height)
            grid = grid.view(1, -1, 1, 2)
        interpolated_values = F.grid_sample(
            distance_image,
            grid,
            mode='bilinear',
            align_corners=True
            )
        return interpolated_values

    def find_closest_minima(self, image, sampled_points, reference_points):
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
        ref_inds = torch.argmin(dist, dim=1)

       # Find local minima for each set of sampled points
        minima_indices = self.find_local_minima_batch(image_values)

        # Find the nearest local minima for each reference point
        nearest_minima_positions = []
        nearest_minima_values = []
        for i in range(len(minima_indices)):
            if len(minima_indices[i]) > 0:
                closest_minima_idx = minima_indices[i][(minima_indices[i] - ref_inds[i]).abs().argmin()].item()
                nearest_minima_positions.append(sampled_points[i, closest_minima_idx])
                nearest_minima_values.append(image_values[i, closest_minima_idx])
            else:
                nearest_minima_positions.append(None)
                nearest_minima_values.append(None)

        return nearest_minima_positions, nearest_minima_values
    
    def find_local_minima(self, tensor):
        # Create shifted versions of the tensor
        shifted_left = torch.roll(tensor, shifts=1)
        shifted_right = torch.roll(tensor, shifts=-1)

        # Compare each element to its neighbors to find local minima
        minima_mask = (tensor < shifted_left) & (tensor < shifted_right)
        
        # Exclude the first and last elements (edge cases, as they have no two neighbors)
        minima_mask[0] = minima_mask[-1] = False
        
        # Get the indices of the minima
        minima_indices = torch.nonzero(minima_mask.squeeze()).squeeze()
        if minima_indices.dim() == 0:
            minima_indices = minima_indices.unsqueeze(0)
        return minima_indices

    def find_local_minima_batch(self, tensor):
        # Create shifted versions of the tensor
        shifted_left = torch.roll(tensor, shifts=1, dims=1)
        shifted_right = torch.roll(tensor, shifts=-1, dims=1)

        # Compare each element to its neighbors to find local minima
        minima_mask = (tensor < shifted_left) & (tensor < shifted_right)
        
        # Exclude the first and last elements (edge cases, as they have no two neighbors)
        minima_mask[:, 0] = minima_mask[:, -1] = False
        
        # Get the indices of the minima
        minima_indices = [torch.nonzero(minima_mask[i]).squeeze() for i in range(minima_mask.size(0))]
        minima_indices = [indices if indices.dim() > 0 else indices.unsqueeze(0) for indices in minima_indices]
        return minima_indices