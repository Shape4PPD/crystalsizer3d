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
        'material_ior': 1.7,#1.7,
        # 'origin': [-2.2178, -0.9920, 5.7441],
        'origin': [0., 0., 0.],
        'rotation': [0., 0., -0.2],
        # 'rotation': [0.0, np.pi, -0.2],
        # 'rotation': [0.6168,  0.3305, -0.4568],
        # 'material_roughness': 0.01
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
    def __init__(self, point, normal, marker, device):
        self.device = device
        self.point = torch.tensor(point, dtype=torch.float32,device=self.device)
        self.normal = torch.tensor(normal, dtype=torch.float32,device=self.device)
        self.marker = torch.tensor(marker, dtype=torch.int,device=self.device) # 0 if external, 1 if refracted
    
    def __repr__(self):
        return f"Point2DTorch(point={self.point}, normal={self.normal}, marker={self.marker})"

    def get_point(self):
        return self.point
    
    def get_normal(self):
        return self.normal
    
    def get_marker(self):
        return self.marker
        
class PointCollection:
    def __init__(self,device):
        self.device = device
        self.points = []
    
    def add_point(self, point, normal, marker):
        point_obj = Point2DTorch(point, normal, marker, self.device)
        self.points.append(point_obj)
    
    def add_points(self, points, normal, marker):
        """Add multiple points with the same normal

        Args:
            points (_type_): _description_
            normal (_type_): _description_
            marker (_type_): _description_
        """
        for point in points:
            point_obj = Point2DTorch(point, normal, marker, self.device)
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
    
    def __len__(self):
        return len(self.points)
    
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
            camera_axis: List[int] = [0, 0, -1],
            zoom: float = 1.,
            external_ior: float = 1.333,  # water
    ):
    
    ### this should essentially be the same as the projector class, except it gives points instead of lines
       
    
        self.crystal = crystal
        self.device = crystal.origin.device
        self.external_ior = external_ior
        
        self.point_collection = PointCollection(device=self.device)
        
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
        
        return self.point_collection

        
        
    def generate_points(self) -> torch.Tensor:    
        """
        Project 3d vertices into 2d plane, while calculating refraction of back planes
        """
        self.vertices_2d = self.project_to_2d(self.vertices)
        # self.visable_points = PointCollection(device=self.device) ##
        all_points = []
        
        # get non-refracted points first
        for face, face_normal, face_distance in zip(self.faces, self.face_normals, self.distances):
            # want faces facing view axis
            if len(face) < 3 or face_normal @ self.view_axis > 0:
                continue
            vertices_2d = self.vertices_2d[face]
            # generate points on edge
            for i in range(len(vertices_2d)):
                cur_vertex = vertices_2d[i]
                next_vertex = vertices_2d[(i + 1) % len(vertices_2d)]
                points, line_normal = self._edge_points(cur_vertex,next_vertex)
                for point in points:
                    self.point_collection.add_point(point,line_normal,0) # 0 for front facing
                    
            # for each from face, i.e. non-refracted face, calculate refraction through that face.
            self._calculate_refraction(face,face_normal, face_distance)
            
                    
    def _calculate_refraction(self,front_face,front_normal,front_distance):
        """Calculates refraction through a given face

        Args:
            front_face: List of indices for a given face
            front_normal: Normal vector of given face
            front_distance: distance value for given face
        """
        refracted_points = self._refract_points(front_normal, front_distance)
        # this just gives the corners of the refracted faces
        
        # now we filter the ones we want
        refracted_2d = self.project_to_2d(refracted_points) # testing

        # we want to calculate if any of the lines between verties cross the front face
        front_face_2d = self.vertices_2d[front_face]
        
        #if face has no area. i.e its being seen from 90 degrees on, don't bother calculating anything
        if self._ploygon_area(front_face_2d) == 0:
            return
        
        # for refracted face, just get the refracted edges without repeats
        # for each refracted face, facing away from the camera
        rear_faces = []
        for face, face_normal, distance in zip(self.faces, self.face_normals, self.distances):
            # want faces facing away from view axis
            if len(face) < 3 or face_normal @ self.view_axis < 0:
                continue
            rear_faces.append(face)
        unique_egdes = self._extract_unique_edges(rear_faces)

        # for each refracted edge, facing away from the camera
        for edge in unique_egdes:
            ref_edge_2d = refracted_2d[edge]
            # look at each edge of each refracted face
            visable_points = []
            start_vertex = ref_edge_2d[0]
            end_vertex = ref_edge_2d[1]
            
            intersections, start_inside, end_inside = self._line_face_intersection(front_face_2d,[start_vertex,end_vertex]) 

            # see if it crosses face edges at all
            if len(intersections) == 0:
                # if no intersections, and both outside do nothing, it doesn't cross the face
                if start_inside and end_inside:
                    # if both inside, then keep booth points
                    visable_points.append(start_vertex)
                    visable_points.append(end_vertex)
            else:
                if start_inside:
                    #then end point needs to be replaced
                    visable_points.append(start_vertex)
                    visable_points.append(intersections[0])
                elif end_inside:
                    #then start point needs to be replaced
                    visable_points.append(intersections[0])
                    visable_points.append(end_vertex)
                else:
                    assert len(intersections) <= 2
                    if len(intersections) == 2:
                        #both outside need both intersection points
                        visable_points.append(intersections[0])
                        visable_points.append(intersections[1])
            if len(visable_points) > 0:
                points, line_normal = self._edge_points(visable_points[0],visable_points[1])
                self.point_collection.add_points(points,line_normal,1)
    
    def _line_face_intersection(self,vertices,test_edge):
        """Checks whether an edge crosses a face in 2d

        Args:
            vertices: face given in 2d coordinates
            edge: [xy1, xy2] two vertices given as a list
        """
        
        n = vertices.shape[0]
        intersections = 0
        intersection_points = []
        
        # Check if the start or end point is inside the polygon
        start_inside = self._is_point_in_polygon(vertices, test_edge[0])
        end_inside = self._is_point_in_polygon(vertices, test_edge[1])
        
        # Check for intersections with each polygon edge
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            
            intersection, intersect = self._line_intersection(test_edge,[v1,v2])
            if intersect:
                intersections += 1
                intersection_points.append(intersection)
            # Check whether start or end point is on polygon
            if torch.equal(v1,test_edge[0]):
                intersections += 1
                intersection_points.append(v1)
            if torch.equal(v1,test_edge[1]):
                intersections += 1
                intersection_points.append(v1)
            
        return intersection_points, start_inside, end_inside
       
    def _ploygon_area(self,vertices):
        # Assuming vertices is an Nx2 tensor where N is the number of points
        x = vertices[:, 0]
        y = vertices[:, 1]
        
        # Calculate the area using the shoelace formula
        area = 0.5 * torch.abs(torch.dot(x, torch.roll(y, -1)) - torch.dot(y, torch.roll(x, -1)))
        
        return area
        
    def _is_point_in_polygon(self, vertices, point):
        n = vertices.shape[0]
        intersections = 0

        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]

            if ((v1[1] > point[1]) != (v2[1] > point[1])):
                intersection_x = (v2[0] - v1[0]) * (point[1] - v1[1]) / (v2[1] - v1[1]) + v1[0]
                if point[0] < intersection_x:
                    intersections += 1

        return intersections % 2 == 1
        
    
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
        start_point = torch.tensor(start_point, dtype=torch.float32,device=self.device)
        end_point = torch.tensor(end_point, dtype=torch.float32,device=self.device)
        num_points = self._num_points(start_point,end_point)
        # Generate a linear space between 0 and 1
        t = torch.linspace(0, 1, steps=num_points,device=self.device)
        
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
    
    def _extract_unique_edges(self,faces):
    # Initialize an empty set for edges
        edges = set()

        # Iterate over each face
        for face in faces:
            num_vertices = face.size(0)  # Get the number of vertices in the face
            
            for i in range(num_vertices):
                # Create edge between vertex i and vertex i+1, wrapping around to the start
                v1, v2 = face[i].item(), face[(i + 1) % num_vertices].item()

                # Store edge in a consistent order
                edge = tuple(sorted((v1, v2)))
                
                # Add edge to set
                edges.add(edge)
        
        # Convert set to a tensor of edges
        unique_edges = torch.tensor(list(edges))
        return unique_edges
           

    def _point_on_face(self,point,face):
        inside = True
        n = face.shape[0]
        
        for i in range(n):
            p1 = face[i]
            p2 = face[(i + 1) % n]
            edge = p2 - p1
            normal = torch.tensor([-edge[1], edge[0]])
            vector_to_point = point - p1
            
            if torch.dot(normal, vector_to_point) > 0:
                inside = False
                break
            
            if range(n) == 2:
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
            basis1 = torch.tensor([1, 0, -x/z],device=self.device)
        elif y != 0:
            basis1 = torch.tensor([1, -x/y, 0],device=self.device)
        else:
            basis1 = torch.tensor([0, 1, 0],device=self.device)
        
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
    try:
        plot_2d_projection.counter += 1
    except:
         plot_2d_projection.counter = 0
    if isinstance(point_collections, PointCollection):
        point_collections = [point_collections]  # Convert to list for uniform processing
    
    # Define a color map
    colours = plt.get_cmap('Set1')
    # colours = plt.get_cmap('tab20')
    
    for i, point_collection in enumerate(point_collections):
        if len(point_collection) == 0:
            continue
        tensor = point_collection.get_points_and_normals_tensor()
        if tensor.dim() != 3 or tensor.size(1) != 2:
            raise ValueError("Each tensor must be of shape (N, 2, 2) where N is the number of points.")
        
        pointx = tensor[:, 0, 0].cpu().numpy()
        pointy = tensor[:,0, 1].cpu().numpy()
        normalx = tensor[:, 1, 0].cpu().numpy()
        normaly = tensor[:, 1, 1].cpu().numpy()
        # Plot the points # plot_2d_projection.counter % 20
        plt.scatter(pointx, pointy, color=colours(i), label=f'Tensor {i+1}',s=2)
        if plot_normals:
            scale = 0.2
            for px,py,nx,ny in zip(pointx,pointy,normalx,normaly):
                plt.arrow(px, py, nx*scale, ny*scale, head_width=0.1, head_length=0.1, fc=colours(i), ec=colours(i))
    
    
    plt.xlim((-1.5,1.5))
    plt.ylim((2.0,-2.0))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
       
        
if __name__ == "__main__":
    crystal = Crystal(**TEST_CRYSTALS['alpha2'])
    crystal.scale.data= init_tensor(1.2, device=crystal.scale.device)
    crystal.origin.data[:2] = torch.tensor([0, 0], device=crystal.origin.device)
    crystal.origin.data[2] -= crystal.vertices[:, 2].min()
    v, f = crystal.build_mesh()
    crystal.to(device)
    projector = ProjectorPoints(crystal,
                                external_ior=1.333,)
    points = projector.project()
    plot_2d_projection(points)