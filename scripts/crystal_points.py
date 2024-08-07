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

class Point2DTorch:
    def __init__(self, point, normal, marker):
        self.point = torch.tensor(point, dtype=torch.float32)
        self.normal = torch.tensor(normal, dtype=torch.float32)
        self.marker = torch.tensor(marker, dtype=torch.int) # 0 if external, 1 if refracted
    
    def __repr__(self):
        return f"Point2DTorch(point={self.point}, normal={self.normal}, marker={self.marker})"

    def get_point(self):
        return self.point
    
    def get_normal(self):
        return self.normal
    
    def get_marker(self):
        return self.marker
        
class PointCollection:
    def __init__(self):
        self.points = []
    
    def add_point(self, point, normal, marker):
        point_obj = Point2DTorch(point, normal, marker)
        self.points.append(point_obj)
    
    def add_points(self, points, normal, marker):
        """Add multiple points with the same normal

        Args:
            points (_type_): _description_
            normal (_type_): _description_
            marker (_type_): _description_
        """
        for point in points:
            point_obj = Point2DTorch(point, normal, marker)
            self.points.append(point_obj)
        
    def get_all_points(self):
        return self.points
    
    def get_all_points_tensor(self):
        points_list = [point.get_point() for point in self.points]
        return torch.stack(points_list)
    
    def get_points_and_normals_tensor(self):
        points_normals_list = [(point.get_point(), point.get_normal()) for point in self.points]
        points_normals_tensor = torch.stack([torch.stack(pn) for pn in points_normals_list])
        return points_normals_tensor
    
    def __repr__(self):
        return f"PointCollection({self.points})"


class ProjectorPoints:
    vertices: torch.Tensor
    faces: List[torch.Tensor]
    face_normals: torch.Tensor
    distances: torch.Tensor
    vertices_2d: torch.Tensor
    image: torch.Tensor
    points: PointCollection

    def __init__(
            self,
            crystal: Crystal,
            camera_axis: List[int] = [0, 0, 1],
            zoom: float = 1.,
            external_ior: float = 1.333,  # water
    ):
    
    ### this should essentially be the same as the projector class, except it gives points instead of lines
       
    
        self.crystal = crystal
        self.device = crystal.origin.device
        self.external_ior = external_ior
        
        self.point_collection = PointCollection()
        
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
        
        # get non-refracted points first
        for face, normal, distance in zip(self.faces, self.face_normals, self.distances):
            # want faces facing view axis
            if len(face) < 3 or normal @ self.view_axis > 0:
                continue
            vertices_2d = self.vertices_2d[face]
            # generate points on edge
            for i in range(len(vertices_2d)):
                cur_vertex = vertices_2d[i]
                next_vertex = vertices_2d[(i + 1) % len(vertices_2d)]
                points, normal = self._edge_points(cur_vertex,next_vertex)
                for point in points:
                    self.point_collection.add_point(point,normal,0) # 0 for front facing

        # Flatten the list of lists into a single list of tensors
        # print(self.point_collection)
        # plot_2d_projection(self.point_collection,plot_normals=False)
        
        self.refracted_points = PointCollection()
        
        # Calculate refraction through each face
        for face, normal, distance in zip(self.faces, self.face_normals, self.distances):
            # want faces facing view axis
            if len(face) < 3 or normal @ self.view_axis > 0:
                continue
            
            # get the points but in the refracted plane through the top face
            refracted_points = self._refract_points(normal, distance) #  this is a refracted version of self.vertices still in 3d
            
            # now we want to convert to 2d and only have points within the face o finterest
            refracted_points = self._filter_points(face,refracted_points)
            
        plot_2d_projection([self.refracted_points,self.point_collection],plot_normals=False)
            
    
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
        
        # Compute the direction vector
        direction = end_point - start_point
        
        # Find a normal vector to the direction vector
        normal_vector = torch.tensor([-direction[1], direction[0]], dtype=torch.float32)
        
        return points, normal_vector
    
    def _num_points(self,start_point,end_point):
        #### update this function
        return 10
    
    def _filter_points(self, face: torch.Tensor, refracted_points: torch.Tensor) -> torch.Tensor:
        """
        This function takes a face, and calculates where each line from the refracted points intersects with the given face
        Then it figures out which of the points lie within the face
        """
        
        # convert from 3d to 2d in plane of viewing
        refracted_2d = self.project_to_2d(refracted_points)
        face_vertices = self.vertices_2d[face]
        
        for refracted_face in self.faces:
            for j in range(len(refracted_face)): # each edge in that refracted face
                p0 = refracted_2d[refracted_face[j]]
                p1 = refracted_2d[refracted_face[(j+1) % len(refracted_face)]]
                refracted_edge = torch.stack([p0,p1])  
            
                # check atleast one point lies within the face
                # if both are in the face, return both
                p0_f = self._point_on_face(p0,face_vertices)
                p1_f = self._point_on_face(p1,face_vertices)
                
                if p0_f and p1_f:
                    # if both points are on the face then create and edge with points
                    points, normal = self._edge_points(p0,p1)
                    self.refracted_points.add_points(points,normal,1)  # 1 for external
                    ##### Testing
                    # change this from self.refractred after testing
                    continue
                
                for i in range(len(face)):
                    # for each edge of the face
                    face_edge = torch.stack([self.vertices_2d[face[i]],self.vertices_2d[face[(i+1) % len(face)]]]) 
                    # if one is in the face, findf intersection between point and line 
                    # this should also cover if both are outside but cross
                    # and also if they don't cross at all
                    intersect, does_intersect = self._line_intersection(face_edge,refracted_edge)
                    
                    if p0_f and does_intersect:
                        # then other point is not on face
                        points, normal = self._edge_points(p0,intersect)
                        self.refracted_points.add_points(points,normal,1) # 1 for external
                        ##### Testing
                        # change this from self.refractred after testing
                        continue
                    
                    if p1_f and does_intersect:
                        # then other point is not on face
                        points, normal = self._edge_points(intersect,p0)
                        self.refracted_points.add_points(points,normal,1)  # 1 for external
                        ##### Testing
                        # change this from self.refractred after testing
                        continue
                    
                    if does_intersect:
                        # if both points are not on face, but it does intersect
                        # i.e it crosses the face
                        
                        #first check if intersect is p0 or p1
                        if self._point_on_face(p0,face_edge): # checking if direction of face or not
                            # then intersect replaces p1
                            pass
                        else:
                            # then intersect replaces p0
                            pass
                        # need to figure that out                    

    def _point_on_face(self,point,face):
        inside = True
        n = len(face)
        
        for i in range(n):
            p1 = face[i]
            p2 = face[(i + 1) % n]
            edge = p2 - p1
            normal = torch.tensor([-edge[1], edge[0]])
            vector_to_point = point - p1
            
            if torch.dot(normal, vector_to_point) > 0:
                inside = False
                break
        
        return inside
    
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
        Identical of _refract_points from 
        """
        points = self.vertices
        # eta = n_1 / n_2
        eta = self.external_ior / self.crystal.material_ior

        # The incident vector is pointing towards the camera
        incident = -self.view_axis / self.view_axis.norm()  # this has mag of 1

        # Normalise the normal vector
        n_norm = normal.norm()
        normal = normal / normal.norm()

        # Calculate cosines and sine^2 for the incident and transmitted angles
        cos_theta_inc = incident @ normal
        sin2_theta_t = eta**2 * (1 - cos_theta_inc ** 2)

        # Calculate the distance from each point to the plane
        dot_product = points @ normal
        offset_distance = self.crystal.origin @ normal # offset from origin due to normal vectors being based off 0,0,0
        d = torch.abs(dot_product - distance * self.crystal.scale - offset_distance) / n_norm

        
        # Check for total internal reflection ###### not required, if true just don't return anything
        if sin2_theta_t > 1:
            R = incident - 2 * cos_theta_inc * normal
            points = points + d[:, None] * R / R.norm()

        # Calculate the refracted vertices
        else:
            # once you've calculated the refraction angles
            # you need to work out where it refracts on the plane
            cos_theta_t = torch.sqrt(1 - sin2_theta_t)
            theta_t = torch.arccos(cos_theta_t)
            theta_inc = torch.arccos(cos_theta_inc)
            
            # calculate magatude of translation in xy direction
            # S is the right angle triangle between inc and T, S dot inc = 0
            s_mag = d[:,None] * torch.sin(theta_inc - theta_t) / torch.cos(theta_t)
            # calculate unit vector translation in direction perpendicular to inc
            T = (eta) * incident + ((eta) * cos_theta_inc - cos_theta_t) * normal
            T = T / T.norm()
            # find how far T travels in inc direction
            T_in_inc = T * incident
            S = -T/T_in_inc.norm() + incident

            if torch.is_nonzero(S.norm()):
                S = S/ S.norm()
                shift = s_mag*S
            else:
                shift = 0

            points = points + shift
            
        # Add distance to plane
        
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


def plot_2d_projection(point_collections: PointCollection, plot_normals=False):
    """
    Plots 2D points from a PyTorch tensor or a list of tensors.
    
    Args:
    points (torch.Tensor or list of torch.Tensor): A tensor of 2D points or a list of tensors of 2D points.
    """
    if isinstance(point_collections, PointCollection):
        point_collections = [point_collections]  # Convert to list for uniform processing
    
    # Define a color map
    colours = plt.get_cmap('Set1')
    
    for i, point_collection in enumerate(point_collections):
        tensor = point_collection.get_points_and_normals_tensor()
        if tensor.dim() != 3 or tensor.size(1) != 2:
            raise ValueError("Each tensor must be of shape (N, 2, 2) where N is the number of points.")
        
        pointx = tensor[:, 0, 0].numpy()
        pointy = tensor[:,0, 1].numpy()
        normalx = tensor[:, 1, 0].numpy()
        normaly = tensor[:, 1, 1].numpy()
        # Plot the points
        plt.scatter(pointx, pointy, color=colours(i), label=f'Tensor {i+1}')
        if plot_normals:
            scale = 0.2
            for px,py,nx,ny in zip(pointx,pointy,normalx,normaly):
                plt.arrow(px, py, nx*scale, ny*scale, head_width=0.1, head_length=0.1, fc=colours(i), ec=colours(i))
    
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
       
        
if __name__ == "__main__":
    crystal = Crystal(**TEST_CRYSTALS['alpha'])
    projector = ProjectorPoints(crystal)
    projector.project()
    pass