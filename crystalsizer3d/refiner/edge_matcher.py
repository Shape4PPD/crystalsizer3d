import torch
from torch import nn, Tensor
import torch.nn.functional as F
from crystalsizer3d import logger

def to_absolute_coordinates(coords: Tensor, image_size: int) -> Tensor:
    """
    Convert relative coordinates to absolute coordinates.
    """
    return torch.stack([
        (coords[:, 0] * 0.5 + 0.5) * image_size[2],
        (0.5 - coords[:, 1] * 0.5) * image_size[3]
    ], dim=1)

            
class EdgeMatcher(nn.Module):
    def __init__(self,points_per_unit = 0.05):
        super(EdgeMatcher, self).__init__()
        self.points_per_unit = points_per_unit

    def forward(self, edge_segments, distance_image):
        """

        Args:
            edge_segments (tensor): edge segments from projector
            normals (tensor): normal vector at each point
            distance_image (tensor): image used from RCF 

        Returns:
            _type_: _description_
        """ 
        
        self.edge_segments = {k: to_absolute_coordinates(v.to(distance_image.device),distance_image.shape) for k, v in edge_segments.items()}      
        
        self._calculate_edge_points()
        
        # Sample points along lines for all reference points
        line_points = self.sample_points_along_line_full_image(self.edge_points, self.edge_normals, 1000, distance_image.shape[-2:])
        
        # Get reference values for all points 
        reference_values = self.points_in_image(self.edge_points, distance_image)
        
        # Find nearest local minima for all line points
        nearest_minima_positions, nearest_minima_values = self.find_closest_minima(distance_image, line_points, self.edge_points)
        
        # Filter out None values
        valid_indices = [i for i, val in enumerate(nearest_minima_values) if val is not None]
        if not valid_indices:
            return torch.tensor(0.0)  # Return zero if no valid minima found
        
        valid_indices = torch.tensor(valid_indices, device=distance_image.device)
        valid_ref_points = self.edge_points[valid_indices]
        valid_reference_values = reference_values[:,:, valid_indices]
        valid_nearest_minima_positions = torch.stack([nearest_minima_positions[i] for i in valid_indices])
        valid_nearest_minima_values = torch.stack([nearest_minima_values[i] for i in valid_indices])
        
        # Calculate distances and set them
        distances = valid_nearest_minima_positions - valid_ref_points
        
        
        # Compute the mean squared error loss
        losses = torch.mean((valid_reference_values.squeeze() - valid_nearest_minima_values.squeeze()) ** 2)
        
        # Return the mean loss across all valid points
        return torch.mean(losses), distances

    def _num_points(self,start_point,end_point):
        distance = torch.norm(end_point - start_point)
        num_points = self.points_per_unit * distance
        return int(torch.ceil(num_points).item())
    
    def _edge_points(self, segment):
        """
        Generates points between two given points
        """
        start_point, end_point = segment
        num_points = self._num_points(start_point,end_point)
        # Generate a linear space between 0 and 1
        # generate two more points for vertices and remove
        t = torch.linspace(0, 1, steps=num_points+2,device=segment.device)
        # Interpolate between start_point and end_point
        points = (1 - t).unsqueeze(1) * start_point + t.unsqueeze(1) * end_point
        # Compute the direction vector
        direction = end_point - start_point
        # Find a normal vector to the direction vector
        normal_vector = torch.tensor([-direction[1], direction[0]], dtype=torch.float32, device=segment.device)
        # remove first and last point (cornors)
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
        edge_points = torch.cat(edge_points,dim=0)
        edge_normals = torch.cat(edge_normals,dim=0)
        self.edge_points = edge_points
        self.edge_normals = edge_normals

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

        # Normalize points to [-1, 1] range for grid_sample
        normalized_points = torch.stack([
            2.0 * points[:, 0] / (W - 1) - 1.0,  # Normalize x (width)
            2.0 * points[:, 1] / (H - 1) - 1.0   # Normalize y (height)
        ], dim=1)
        # Reshape grid to match F.grid_sample requirements
        grid = normalized_points.view(1, -1, 1, 2)
        
        # Clone grid to avoid inadvertently retaining the graph in subsequent usage
        grid = grid.clone()

        # Interpolate values using bilinear sampling
        interpolated_values = F.grid_sample(
            distance_image.detach(),
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
        
        nearest_minima_positions = torch.stack([
            sampled_points[i, minima_indices[i][(minima_indices[i] - ref_inds[i]).abs().argmin()].item()]
            for i in range(len(minima_indices))
            if len(minima_indices[i]) > 0
        ])

        nearest_minima_values = torch.stack([
            image_values[i, minima_indices[i][(minima_indices[i] - ref_inds[i]).abs().argmin()].item()]
            for i in range(len(minima_indices))
            if len(minima_indices[i]) > 0
        ])

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

        # Find indices of local minima in a way that preserves gradients
        minima_indices = []
        for i in range(minima_mask.size(0)):
            # Ensure that we directly index into the tensor without detaching
            indices = torch.nonzero(minima_mask[i], as_tuple=True)[0]
            minima_indices.append(indices)

        return minima_indices
