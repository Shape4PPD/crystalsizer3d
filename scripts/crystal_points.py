""" Class that generates a projection, but as points rather than an image """

from typing import List, Optional, Tuple
import torch
from crystalsizer3d.crystal import Crystal
from kornia.geometry import axis_angle_to_rotation_matrix, center_crop, quaternion_to_rotation_matrix

from crystalsizer3d.util.geometry import is_point_in_bounds, line_equation_coefficients, line_intersection, normalise
from crystalsizer3d.util.utils import init_tensor


import numpy as np
import matplotlib.pyplot as plt

#windows thing
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

TEST_CRYSTALS = {
    'alpha': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 0, -1), (1, 1, 1), (1, 1, -1), (0, 1, 1), (0, 1, -1), (1, 0, 0)],
        'distances': [0.53, 0.50, 1.13, 1.04, 1.22, 1.00, 1.30],
        'point_group_symbol': '222',
        'scale': 3.0,
    },
    'alpha2': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 0, 0), (1, 1, 0), (0, 0, -1), (0, -1, -1),
                           (0, 1, -1), (0, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1),
                           (1, 1, -1), (-1, 0, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'distances': [0.3796, 0.7174, 0.6786, 0.8145, 0.8202, 0.7282, 0.4324, 0.9445, 0.7954, 0.8493, 0.6460, 0.5496,
                      0.7618, 0.6710, 0.8263, 0.6061, 1.0000, 0.9338, 0.7891, 0.9057],
        'point_group_symbol': '222',  # 222?
        'scale': 12,
        'material_ior': 1.7,
        'origin': [-2.2178, -0.9920, 5.7441],
        'rotation': [0., 0., -0.2],
        # 'rotation': [0.6168,  0.3305, -0.4568],
    },
    'alpha3': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [[0, 0, 1], [0, 0, -1], [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1],
                           [-1, 1, -1], [-1, -1, 1], [-1, -1, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 1], [0, 1, -1],
                           [0, -1, 1], [0, -1, -1]],
        'distances': [1.0, 1.0, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 2.0, 2.0, 2.0, 2.0, 1.8, 1.8],
        'point_group_symbol': '1',
        'scale': 3,
        # 'origin': [-2.2178, -0.9920,  5.7441],
        # 'rotation': [0.6168,  0.3305, -0.4568],
    },
    'beta': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 1, 1), (0, 2, 1), (1, 0, -1), (0, 2, -1), (0, 1, 0)],
        'distances': [16.0, 5.0, 16.0, 5.0, 2.39],
        'point_group_symbol': '222',
        'scale': 25.0,
    },
}

device = torch.device('cuda')


class ProjectorPoints:
    vertices: torch.Tensor
    faces: List[torch.Tensor]
    face_normals: torch.Tensor
    distances: torch.Tensor
    vertices_2d: torch.Tensor
    image: torch.Tensor

    def __init__(
            self,
            crystal: Crystal,
            camera_axis: List[int] = [0, 0, 1],
            zoom: float = 1.,
    ):
    
    ### this should essentially be the same as the projector class, except it gives points instead of lines
       
    
        self.crystal = crystal
        self.device = crystal.origin.device
        
        self.view_axis = normalise(init_tensor(camera_axis, device=self.device))
        
        
    def project(self) -> torch.Tensor:
        """
        Project the crystal onto an image.
        """
        self.vertices = self.crystal.vertices.clone()
        self.faces = [
            self.crystal.faces[tuple(hkl)].clone()
            if tuple(hkl) in self.crystal.faces else torch.tensor([], device=self.device)
            for hkl in self.crystal.all_miller_indices.tolist()
        ]
        self.distances = self.crystal.all_distances.clone()

        # Apply crystal rotation to the face normals
        if self.crystal.rotation.shape == (3,):
            R = axis_angle_to_rotation_matrix(self.crystal.rotation[None, ...])[0]
        else:
            R = quaternion_to_rotation_matrix(self.crystal.rotation[None, ...])[0]
        self.face_normals = self.crystal.N @ R.T


        self.generate_points()
        
        # # Orthogonally project original vertices
        # self.vertices_2d = self._orthogonal_projection(self.vertices)

        
        
    def generate_points(self) -> torch.Tensor:    
        """
        Project 3d vertices into 2d plane, while calculating refraction of back planes
        """
        self.vertices_2d = self.project_to_2d(self.vertices)
        
        all_points = []
        
        # draw non-refracted lines first
        for face, normal, distance in zip(self.faces, self.face_normals, self.distances):
            # want faces facing view axis
            if len(face) < 3 or normal @ self.view_axis > 0:
                continue
            vertices_2d = self.vertices_2d[face]
            # generate points on edge
            for i in range(len(vertices_2d)):
                cur_vertex = vertices_2d[i]
                next_vertex = vertices_2d[(i + 1) % len(vertices_2d)]
                edge_points = self._edge_points(cur_vertex,next_vertex)
                all_points.append(edge_points)

        # Flatten the list of lists into a single list of tensors
        flattened_list = [tensor for sublist in all_points for tensor in sublist]

        plot_2d_projection(torch.stack(flattened_list))
        
        return
        # plot_2d_projection(self.vertices_2d)
        refracted_points_list = []
        
    
        
        
        # Calculate refraction through each face
        for face, normal, distance in zip(self.faces, self.face_normals, self.distances):
            # want faces facing view axis
            if len(face) < 3 or normal @ self.view_axis > 0:
                continue
            
            # get the points but in the refracted plane through the top face
            refracted_points = self._refract_points(normal, distance)
            
            refracted_points = self._filter_points(face,refracted_points)
            
            # if refracted is not 
            refracted_points_list.append(refracted_points)
            
        refracted_points = torch.stack(refracted_points_list)
            
        self.plot_2d_projection(torch.stack([self.vertices_2d,refracted_points]))
            
    
    def _edge_points(self, start_point, end_point):
        """
        Generates points between two given points using PyTorch.

        Args:
        start_point (torch.Tensor): The starting point.
        end_point (torch.Tensor): The ending point.
        
        Returns:
        torch.Tensor: A tensor containing the generated points.
        """
        # Ensure start_point and end_point are tensors
        start_point = torch.tensor(start_point, dtype=torch.float32)
        end_point = torch.tensor(end_point, dtype=torch.float32)
        num_points = self._num_points(start_point,end_point)
        # Generate a linear space between 0 and 1
        t = torch.linspace(0, 1, steps=num_points)
        
        # Interpolate between start_point and end_point
        points = (1 - t).unsqueeze(1) * start_point + t.unsqueeze(1) * end_point
        
        return points
    
    def _num_points(self,start_point,end_point):
        #### update this function
        return 10
    
    def _filter_points(self, face: torch.Tensor, refracted_points: torch.Tensor) -> torch.Tensor:
        """
        This function takes a face, and calculates where each line from the refracted points intersects with the given face
        Then it figures out which of the points lie within the face
        """
        
        refracted_2d = self.project_to_2d(refracted_points)
        
        refracted_in_face = []
        
        current_face = []
        for i in range(len(face)):
            # for each edge of the face
            face_edge = torch.stack([self.vertices_2d[face[i]],self.vertices_2d[face[(i+1) % len(face)]]])
            current_face.append(face_edge)
            
            for refracted_face in self.faces:
                for j in range(len(refracted_face)):
                    refracted_edge = torch.stack([refracted_2d[refracted_face[j]]
                                                    ,refracted_2d[refracted_face[(j+1) % len(refracted_face)]]])             

                    intersect, does_intersect = self._line_intersection(face_edge,refracted_edge)
                    
                    if does_intersect:
                        refracted_in_face.append(intersect)
                        ### needs away of storing this with the line it came from
        refracted_in_face = torch.stack(refracted_in_face)
        current_face = torch.cat(current_face)
        plot_2d_projection([self.vertices_2d,refracted_in_face.detach(),current_face])
        
                    
        return refracted_in_face

    
    def _line_intersection(self,a, b):
        # Unpack points
        x1, y1 = a[0]
        x2, y2 = a[1]
        x3, y3 = b[0]
        x4, y4 = b[1]
        
        # Calculate denominators
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # Check if lines are parallel (denom == 0)
        if denom == 0:
            return None, False  # Lines are parallel and do not intersect
        
        # Calculate the intersection point
        intersect_x = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
        intersect_y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
        intersection = torch.tensor([intersect_x, intersect_y])
        
        # Check if the intersection point is within both line segments
        def is_between(p, q, r):
            return (min(p, q) <= r <= max(p, q))
        
        if (is_between(x1, x2, intersect_x) and is_between(y1, y2, intersect_y) and
            is_between(x3, x4, intersect_x) and is_between(y3, y4, intersect_y)):
            return intersection, True
        
        return None, False  # Intersection point is not within the segments
    
    def _refract_points(self, normal: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """
        Refract the crystal vertices in the plane given by the normal and the distance.
        """
        # Calculate the (negative) unit vector of the refracted direction
        eta = 1 / self.crystal.material_ior

        # Calculate the refracted vertices
        points = self.vertices
        incident = self.view_axis/self.view_axis.norm()
        theta_inc = -normal @ incident / normal.norm()
        cos_theta_t = torch.sqrt(1 - (eta**2) * ( 1 - torch.cos(theta_inc)**2 ))
        
        T = eta*incident + ( eta*torch.cos(theta_inc) - cos_theta_t) * normal / normal.norm()
        
        dot_product = points @ normal

        # Calculate the distance from each point to the plane
        d = torch.abs(dot_product - distance*self.crystal.scale) / torch.norm(normal)
        
        # add distance to plane
        points = points + (d[:, None] / cos_theta_t ) *T / T.norm()

        return points

    def project_to_2d(self, points):
        # Normalize the view direction
        view_direction = self.view_axis / torch.norm(self.view_axis)
        
        # Assuming the view direction is (x, y, z) and is normalized
        x, y, z = view_direction
        
        # Create a projection matrix for orthogonal projection
        # We need to find two orthogonal vectors to the view direction to form the basis for the 2D plane
        if z != 0:
            basis1 = torch.tensor([1, 0, -x/z])
        elif y != 0:
            basis1 = torch.tensor([1, -x/y, 0])
        else:
            basis1 = torch.tensor([0, 1, 0])
        
        basis1 = basis1 / torch.norm(basis1)
        
        basis2 = torch.cross(view_direction, basis1)
        basis2 = basis2 / torch.norm(basis2)
        
        # Create the projection matrix
        projection_matrix = torch.stack([basis1, basis2], dim=1)
        
        # Project the 3D points into the 2D plane
        projected_points = torch.matmul(points, projection_matrix)
        
        return projected_points


def plot_2d_projection(points: torch.Tensor):
    """
    Plots 2D points from a PyTorch tensor or a list of tensors.
    
    Args:
    points (torch.Tensor or list of torch.Tensor): A tensor of 2D points or a list of tensors of 2D points.
    """
    if isinstance(points, torch.Tensor):
        points = [points]  # Convert to list for uniform processing
    
    # Define a color map
    colours = plt.get_cmap('Set1')
    
    for i, tensor in enumerate(points):
        if tensor.dim() != 2 or tensor.size(1) != 2:
            raise ValueError("Each tensor must be of shape (N, 2) where N is the number of points.")
        
        # Plot the points
        plt.scatter(tensor[:, 0].numpy(), tensor[:, 1].numpy(), color=colours(i), label=f'Tensor {i+1}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
       
        
if __name__ == "__main__":
    crystal = Crystal(**TEST_CRYSTALS['alpha'])
    projector = ProjectorPoints(crystal)
    projector.project()
    pass