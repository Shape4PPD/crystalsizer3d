from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.geometry import axis_angle_to_rotation_matrix, center_crop, quaternion_to_rotation_matrix
from kornia.utils import draw_line
from kornia.utils.draw import _batch_polygons, _get_convex_edges
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from torch import Tensor
from torch.nn.functional import interpolate

from crystalsizer3d.crystal import Crystal
from crystalsizer3d.util.geometry import is_point_in_bounds, line_equation_coefficients, line_face_intersection, \
    line_intersection, normalise, polygon_area
from crystalsizer3d.util.utils import init_tensor

# A vertex can appear in multiple faces due to refraction, so we need to store the face index (or 'facing') as well
ProjectedVertexKey = Tuple[int, Union[str, int]]


class TotalInternalReflectionError(Exception):
    pass


def replace_convex_polygon(
        images: Tensor,
        polygons: Tensor,
        replacement: Tensor
) -> Tensor:
    """
    Replaces a convex polygons with an image on a batch of image tensors.
    This is modified from the kornia function draw_convex_polygon, to allow full image replacement.

    Args:
        images: is tensor of BxCxHxW.
        polygons: represents polygons as points, either BxNx2 or List of variable length polygons.
            N is the number of points.
            2 is (x, y).
        replacement: is tensor of BxCxHxW.

    Returns:
        This operation modifies image inplace but also returns the drawn tensor for
        convenience with same shape the of the input BxCxHxW.

    Note:
        This function assumes a coordinate system (0, h - 1), (0, w - 1) in the image, with (0, 0) being the center
        of the top-left pixel and (w - 1, h - 1) being the center of the bottom-right coordinate.
    """
    KORNIA_CHECK_SHAPE(images, ["B", "C", "H", "W"])
    b_i, c_i, h_i, w_i, device = *images.shape, images.device
    if isinstance(polygons, List):
        polygons = _batch_polygons(polygons)
    b_p, _, xy, device_p, dtype_p = *polygons.shape, polygons.device, polygons.dtype
    KORNIA_CHECK_SHAPE(replacement, ["B", "C", "H", "W"])
    b_r, c_r, h_r, w_r, device_r = *replacement.shape, replacement.device
    KORNIA_CHECK(xy == 2, "Polygon vertices must be xy, i.e. 2-dimensional")
    KORNIA_CHECK(b_i == b_p == b_r, "Image, polygon, and color must have same batch dimension")
    KORNIA_CHECK(device == device_p == device_r, "Image, polygon, and color must have same device")
    x_left, x_right = _get_convex_edges(polygons, h_i, w_i)
    ws = torch.arange(w_i, device=device, dtype=dtype_p)[None, None, :]
    fill_region = (ws >= x_left[..., :, None]) & (ws <= x_right[..., :, None])
    images = (~fill_region[:, None]) * images + fill_region[:, None] * replacement
    return images


class Projector:
    vertices: Tensor
    vertex_ids: Tensor
    faces: List[Tensor]
    face_normals: Tensor
    distances: Tensor
    vertices_2d: Tensor
    projected_vertices: Dict[int, Tensor]
    projected_vertex_keys: List[ProjectedVertexKey]
    projected_vertex_coords: Tensor
    rear_facing_edges: Tensor
    image: Tensor

    def __init__(
            self,
            crystal: Crystal,
            image_size: Tuple[int, int] = (1000, 1000),
            camera_axis: List[int] = [0, 0, -1],
            zoom: float = 1.,
            background_image: np.ndarray = None,
            transparent_background: bool = False,
            external_ior: float = 1.333,  # water
            colour_facing_towards: List[float] = [1, 0, 0],
            colour_facing_away: List[float] = [0, 0, 1],
            multi_line: bool = True,
    ):
        """
        Project a crystal onto an image.
        """
        self.crystal = crystal
        self.device = crystal.origin.device
        self.external_ior = external_ior

        # Set image size and aspect ratio
        self.image_size = image_size
        self.aspect_ratio = image_size[0] / image_size[1]
        self.view_axis = normalise(init_tensor(camera_axis, device=self.device))
        self.zoom = zoom
        self.x_range = init_tensor([-self.aspect_ratio, self.aspect_ratio], device=self.device) / self.zoom
        self.y_range = init_tensor([-1, 1], device=self.device) / self.zoom

        # Background
        self.background_image = None
        self.set_background(background_image)
        self.transparent_background = transparent_background

        # Drawing mode
        self.multi_line = multi_line

        # Colours
        self.colour_facing_towards = init_tensor(colour_facing_towards, device=self.device)
        self.colour_facing_away = init_tensor(colour_facing_away, device=self.device)

        # Create blank canvas image
        self.canvas_image = torch.zeros(3, *image_size, device=self.device)

        # Set the image boundary lines
        h, w = image_size
        self.boundary_lines = [
            line_equation_coefficients(p1, p2)
            for p1, p2 in torch.tensor([
                [(1, 1), (1, h - 2)],  # Left
                [(w - 2, 1), (w - 2, h - 2)],  # Right
                [(1, h - 2), (w - 2, h - 2)],  # Bottom
                [(1, 1), (w - 2, 1)]  # Top
            ], device=self.device)
        ]

        # Do the projection
        self.image = self.project()

    def _to_relative_coords(self, coords: Tensor) -> Tensor:
        """
        Convert absolute coordinates to relative coordinates.
        """
        image_size = torch.tensor(self.image_size, device=self.device)
        return torch.stack([
            (coords[0] / image_size[1] - 0.5) * 2,
            (0.5 - coords[1] / image_size[0]) * 2
        ])

    def set_background(self, background_image: Optional[Union[Tensor, np.ndarray]] = None):
        """
        Set the background image for the projector.
        """
        if background_image is not None:
            assert background_image.ndim == 3, 'Background image must be a three-channel colour image.'
            background_image = init_tensor(background_image, device=self.device)
            if background_image.shape[-1] == 3:
                background_image = background_image.permute(2, 0, 1)  # Convert to CxHxW
            if background_image.amax() > 1:  # Convert from 0-255 to 0-1
                background_image = background_image / 255
            if background_image.shape[-2:] != self.image_size:  # Resize the image to match
                min_size = min(background_image.shape[-2:])
                background_image = center_crop(background_image[None, ...], (min_size, min_size))
                background_image = interpolate(
                    background_image,
                    size=self.image_size,
                    mode='bilinear',
                    align_corners=False
                )[0]
        self.background_image = background_image

    def project(self, generate_image: bool = True) -> Optional[Tensor]:
        """
        Project the crystal onto an image.
        """
        self.vertices = self.crystal.vertices.clone()
        self.vertex_ids = self.crystal.vertex_ids.clone()
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

        # Orthogonally project original vertices
        self.vertices_2d = self._orthogonal_projection(self.vertices)

        # Project the vertices including all refractions
        self._project_vertices()

        # Extract the edges facing away from the camera - required for multi-line drawing
        self.rear_facing_edges = self._extract_rear_facing_edges()

        # Generate the refracted wireframe image
        if generate_image:
            self.image = self._generate_image()

            return self.image

    def _project_vertices(self):
        """
        Projected the vertices including all refractions.
        """
        pv = {}
        pv_keys = []
        pv_coords = []

        for face_idx, (face, normal, distance) in enumerate(zip(self.faces, self.face_normals, self.distances)):
            # Only consider the faces that are facing the camera
            is_facing = normal @ self.view_axis < 0
            if len(face) < 3 or not is_facing:
                continue

            # As this face is facing the camera, all the vertices are directly visible
            for idx in face:
                vertex_id = int(self.vertex_ids[idx])
                key = (vertex_id, 'facing')
                if key in pv_keys:
                    continue
                pv_keys.append(key)
                pv_coords.append(self._to_relative_coords(self.vertices_2d[idx]))

            # Refract the points and project them to 2d
            try:
                refracted_points = self._refract_points(normal, distance)
            except TotalInternalReflectionError:
                continue
            vertices_2d = self._orthogonal_projection(refracted_points)
            pv[face_idx] = vertices_2d

            # Check that the points on the face did not move
            face_vertices_og = self.vertices_2d[face]
            face_vertices = vertices_2d[face]
            # assert torch.allclose(face_vertices_og, face_vertices, rtol=1e-2), 'Face vertices moved during refraction.'

            # Filter the refracted vertices to the ones visible through this face
            polygon = Polygon(self.vertices_2d[face].tolist())
            for v_idx, v_ref in enumerate(vertices_2d):
                vertex_id = int(self.vertex_ids[v_idx])
                if vertex_id in face:  # Skip vertices that are already visible
                    continue
                point = Point(v_ref.tolist())
                if polygon.contains(point):
                    key = (vertex_id, face_idx)
                    pv_keys.append(key)
                    pv_coords.append(self._to_relative_coords(vertices_2d[v_idx]))
        if len(pv_coords) > 0:
            pv_coords = torch.stack(pv_coords)

        # Store the projected 2D vertices
        self.projected_vertices = pv
        self.projected_vertex_keys = pv_keys
        self.projected_vertex_coords = pv_coords

    def _generate_image(self) -> Tensor:
        """
        Generate the projected wireframe image including all refractions.
        """
        image = self.canvas_image.clone()

        # Draw hidden refracted edges first
        for face_idx, (face, normal, distance) in enumerate(zip(self.faces, self.face_normals, self.distances)):
            is_facing = normal @ self.view_axis < 0
            if len(face) < 3 or not is_facing:
                continue
            if face_idx not in self.projected_vertices:
                continue

            # Draw every line segment individually
            if self.multi_line:
                image = self._draw_edge_segments(face_idx, image)

            # Refract all the lines through the face then cut out the visible parts
            else:
                # Draw all refracted edges on an empty image
                vertices_2d = self.projected_vertices[face_idx]
                rf_image = self._draw_full_edges(vertices_2d, facing_camera=False)

                # Fill the face with the refracted image
                image = replace_convex_polygon(
                    images=image[None, ...],
                    polygons=self.vertices_2d[face][None, ...],
                    replacement=rf_image[None, ...]
                )[0]

        # Draw top edges on the image
        image = self._draw_full_edges(self.vertices_2d, image=image, facing_camera=True)

        # Apply the background image
        if self.background_image is not None:
            bg = image.sum(dim=0) == 0
            composite = self.background_image.clone()
            composite[:, ~bg] = image[:, ~bg]
            image = composite

        # Add transparency to the image
        if self.transparent_background:
            alpha = torch.zeros((1, *self.image_size), dtype=torch.uint8, device=image.device)
            alpha[0, image.sum(dim=0) == 0] = 0
            alpha[0, image.sum(dim=0) != 0] = 1
            image = torch.cat([image, alpha], dim=0)

        return image

    def _draw_edge_segments(self, face_idx: int, image: Tensor):
        """
        Draw all edge segments refracted through a given face onto the given image.
        """
        refracted_2d = self.projected_vertices[face_idx]
        front_face_2d = self.vertices_2d[self.faces[face_idx]]

        # If face has no area, (i.e. it's being viewed from 90 degrees), there's nothing to draw
        if polygon_area(front_face_2d) < 1e-3:
            return image

        # Consider the refracted projection of each rear-facing edge in the given face
        for edge in self.rear_facing_edges:
            start_vertex, end_vertex = refracted_2d[edge]
            intersections, start_inside, end_inside = line_face_intersection(front_face_2d, [start_vertex, end_vertex])
            visible_points = []

            # No intersections, either both inside or no intersections with the face
            if len(intersections) == 0:
                # If both vertices are inside, keep both
                if start_inside and end_inside:
                    visible_points = [start_vertex, end_vertex]

            # If there is a single intersections, we need to replace one of the points
            elif len(intersections) == 1:
                # End point needs to be replaced
                if start_inside:
                    visible_points = [start_vertex, intersections[0]]

                # Start point needs to be replaced
                elif end_inside:
                    visible_points = [intersections[0], end_vertex]

            # If there are two intersections, keep both
            elif len(intersections) == 2:
                visible_points = intersections

            # Draw the line segment on the image
            if len(visible_points) > 0:
                image = draw_line(image, visible_points[0], visible_points[1], self.colour_facing_away)

        return image

    def _extract_rear_facing_edges(self):
        """
        Extracts unique edges from faces that are facing away from the camera.
        """
        # Create a list of faces that are facing away from the camera
        rear_faces = []
        for face, face_normal in zip(self.faces, self.face_normals):
            if len(face) < 3 or face_normal @ self.view_axis < 0:
                continue
            rear_faces.append(face)

        # Initialise an empty set for edges
        edges = set()

        # Iterate over the faces
        for face in rear_faces:
            num_vertices = len(face)

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

    def _refract_points(self, normal: Tensor, distance: Tensor) -> Tensor:
        """
        Refract the crystal vertices in the plane given by the normal and the distance.
        """
        points = self.vertices
        eta = self.external_ior / self.crystal.material_ior

        # The incident vector is pointing towards the camera (normalised)
        incident = -self.view_axis / self.view_axis.norm()

        # Normalise the normal vector
        n_norm = normal.norm()
        normal = normal / normal.norm()

        # Calculate cosines and sine^2 for the incident and transmitted angles
        cos_theta_inc = incident @ normal
        sin2_theta_t = eta**2 * (1 - cos_theta_inc**2)

        # Check for total internal reflection
        if sin2_theta_t > 1:
            raise TotalInternalReflectionError()

        # Calculate the distance from each point to the plane
        dot_product = points @ normal
        offset_distance = self.crystal.origin @ normal  # offset from origin due to normal vectors being based off 0,0,0
        d = torch.abs(dot_product - distance * self.crystal.scale - offset_distance) / n_norm

        # Calculate the refraction angles and work out where it refracts on the plane
        cos_theta_t = torch.sqrt(torch.clamp(1 - sin2_theta_t, min=1e-3))
        theta_t = torch.arccos(cos_theta_t)
        theta_inc = torch.arccos(cos_theta_inc)

        # Calculate magnitude of translation in xy direction
        # S is the right angle triangle between inc and T, S dot inc = 0
        s_mag = d[:, None] * torch.sin(theta_inc - theta_t) / torch.cos(theta_t)

        # Calculate unit vector translation in direction perpendicular to inc
        T = eta * incident + (eta * cos_theta_inc - cos_theta_t) * normal
        T = T / T.norm()

        # Calculate how far T travels in incident direction
        T_in_inc = T * incident
        S = -T / T_in_inc.norm() + incident
        S_norm = S.norm()

        # Adjust the points
        if S_norm > 0:
            S = S / S_norm
            shift = s_mag * S
            points = points + shift

        return points

    def _orthogonal_projection(self, vertices: Tensor) -> Tensor:
        """
        Orthogonally project 3D vertices with custom x and y ranges.

        Args:
            vertices (Tensor): Tensor of shape (N, 3) representing 3D coordinates of vertices.

        Returns:
            Tensor: Tensor of shape (N, 2) representing 2D coordinates of projected vertices.
        """
        x_min, x_max = self.x_range
        y_max, y_min = self.y_range  # Flip the y-axis to match the image coordinates

        x_scale = (x_max - x_min) / self.image_size[1]
        y_scale = (y_max - y_min) / self.image_size[0]

        projected_vertices = vertices[:, :2].clone()
        projected_vertices[:, 0] = (projected_vertices[:, 0] - x_min) / x_scale
        projected_vertices[:, 1] = (projected_vertices[:, 1] - y_min) / y_scale

        return projected_vertices

    def _draw_full_edges(
            self,
            vertices: Tensor,
            image: Optional[Tensor] = None,
            facing_camera: bool = True,
    ) -> Tensor:
        """
        Draw edges on an image, colouring them based on visibility.

        Args:
            vertices (Tensor): Tensor of shape (N, 2) representing 2D coordinates of vertices.
            image (Tensor): Tensor of shape (3, H, W) representing the image.
            facing_camera (bool): Whether to draw edges facing the camera or away from it.

        Returns:
            Tensor: Tensor of shape (3, H, W) representing the image with edges drawn.
        """
        if image is None:
            image = self.canvas_image
        image = image.clone()  # Create a copy to avoid modifying the original image
        h, w = image.shape[-2:]
        if image.ndim == 4:
            image = image[0]  # Remove batch dimension

        def in_frame(p):
            return 1 <= p[0] < w - 1 and 1 <= p[1] < h - 1

        # Refract the vertices in every face
        for face, normal in zip(self.faces, self.face_normals):
            is_facing = normal @ self.view_axis > 0
            if len(face) < 3 or (facing_camera and is_facing) or (not facing_camera and not is_facing):
                continue

            # Loop over the faces and draw the edges
            for i in range(len(face)):
                idx0, idx1 = face[i], face[(i + 1) % len(face)]
                v0, v1 = vertices[idx0], vertices[idx1]

                # If either of the vertices is out of the frame, clip the line
                if not in_frame(v0) or not in_frame(v1):
                    l = line_equation_coefficients(v0, v1, eps=1e-3)

                    # Calculate all intersection points with boundary lines
                    intersections = []
                    for b in self.boundary_lines:
                        intersect = line_intersection(l, b)
                        if intersect is not None \
                                and in_frame(intersect) \
                                and is_point_in_bounds(intersect, [v0, v1]):
                            intersections.append(intersect)

                    # If there are no intersections, skip the line
                    if len(intersections) == 0 or (len(intersections) == 1 and not in_frame(v0) and not in_frame(v1)):
                        continue
                    assert len(intersections) <= 2, 'Expected maximum of 2 intersections.'
                    intersections = torch.stack(intersections)

                    # Update the vertices
                    if len(intersections) == 1:
                        if not in_frame(v0):
                            v0 = intersections[0]
                        else:
                            v1 = intersections[0]
                    else:
                        vc = (v0 + v1) / 2
                        ic = (intersections[0] + intersections[1]) / 2
                        if not in_frame(v0):
                            v0 = intersections[((v0 - vc) @ (intersections - ic)).argmax()]
                        if not in_frame(v1):
                            v1 = intersections[((v1 - vc) @ (intersections - ic)).argmax()]

                # Check if the edge is facing the camera or passing through another face
                if facing_camera:
                    colour = self.colour_facing_towards
                else:
                    colour = self.colour_facing_away
                image = draw_line(image, v0, v1, colour)

        return image
