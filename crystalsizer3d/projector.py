from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.util.geometry import is_point_in_bounds, line_equation_coefficients, line_intersection, normalise
from crystalsizer3d.util.utils import init_tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.geometry import axis_angle_to_rotation_matrix, center_crop, quaternion_to_rotation_matrix
from kornia.utils import draw_line
from kornia.utils.draw import _batch_polygons, _get_convex_edges
from torch import Tensor
from torch.nn.functional import interpolate


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
    faces: List[Tensor]
    face_normals: Tensor
    distances: Tensor
    vertices_2d: Tensor
    image: Tensor

    def __init__(
            self,
            crystal: Crystal,
            image_size: Tuple[int, int] = (1000, 1000),
            camera_axis: List[int] = [0, 0, -1],
            zoom: float = 1.,
            background_image: np.ndarray = None,
            external_ior: float = 1.333,  # water
            colour_facing_towards: List[float] = [1, 0, 0],
            colour_facing_away: List[float] = [0, 0, 1]
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

        # Background image
        self.background_image = None
        self.set_background(background_image)

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

    def project(self) -> Tensor:
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

        # Orthogonally project original vertices
        self.vertices_2d = self._orthogonal_projection(self.vertices)

        # Generate the refracted wireframe image
        self.image = self._generate_image()

        return self.image

    def _generate_image(self) -> Tensor:
        """
        Generate the projected wireframe image including all refractions.
        """
        image = self.canvas_image.clone()

        # Draw hidden refracted edges first
        for face, normal, distance in zip(self.faces, self.face_normals, self.distances):
            if len(face) < 3 or normal @ self.view_axis > 0:
                continue

            # Refract the points and project them to 2d
            refracted_points = self._refract_points(normal, distance)
            vertices_2d = self._orthogonal_projection(refracted_points)

            # Draw all refracted edges on an empty image
            rf_image = self._draw_edges(vertices_2d, facing_camera=False)

            # Fill the face with the refracted image
            image = replace_convex_polygon(
                images=image[None, ...],
                polygons=self.vertices_2d[face][None, ...],
                replacement=rf_image[None, ...]
            )[0]

        # Draw top edges on the image
        image = self._draw_edges(self.vertices_2d, image=image, facing_camera=True)

        # Apply the background image
        if self.background_image is not None:
            bg = image.sum(dim=0) == 0
            composite = self.background_image.clone()
            composite[:, ~bg] = image[:, ~bg]
            image = composite

        return image

    def _refract_points(self, normal: Tensor, distance: Tensor) -> Tensor:
        """
        Refract the crystal vertices in the plane given by the normal and the distance.
        """
        eta = self.external_ior / self.crystal.material_ior

        # Calculate the refracted vertices
        points = self.vertices
        incident = self.view_axis / self.view_axis.norm()
        n_norm = normal.norm()
        cos_theta_inc = torch.cos(-normal @ incident / n_norm)
        cos_theta_t = torch.sqrt(1 - (eta**2) * (1 - cos_theta_inc**2))
        T = eta * incident + (eta * cos_theta_inc - cos_theta_t) * normal / n_norm
        dot_product = points @ normal

        # Calculate the distance from each point to the plane
        d = torch.abs(dot_product - distance * self.crystal.scale) / n_norm

        # Add distance to plane
        points = points + (d[:, None] / cos_theta_t) * T / T.norm()

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

    def _draw_edges(
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
