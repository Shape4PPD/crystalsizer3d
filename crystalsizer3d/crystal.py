from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from kornia.geometry import axis_angle_to_rotation_matrix, quaternion_to_rotation_matrix
from kornia.utils import draw_convex_polygon
from pymatgen.symmetry.groups import PointGroup
from torch import nn

from crystalsizer3d import logger
from crystalsizer3d.util.geometry import align_points_to_xy_plane, calculate_relative_angles, merge_vertices, normalise, \
    rotate_2d_points_to_square
from crystalsizer3d.util.utils import init_tensor

ROTATION_MODE_QUATERNION = 'quaternion'
ROTATION_MODE_AXISANGLE = 'axisangle'
ROTATION_MODES = [
    ROTATION_MODE_QUATERNION,
    ROTATION_MODE_AXISANGLE
]


class Crystal(nn.Module):
    all_distances: torch.Tensor
    N: torch.Tensor
    vertices: torch.Tensor
    faces: Dict[int, torch.Tensor]
    mesh_vertices: torch.Tensor
    mesh_faces: torch.Tensor
    bumpmap: torch.Tensor
    uv_mask: torch.Tensor

    def __init__(
            self,
            lattice_unit_cell: List[float],
            lattice_angles: List[float],
            miller_indices: List[Tuple[int, int, int]],
            point_group_symbol: str = '1',

            scale: float = 1.0,
            distances: Optional[List[float]] = None,
            origin: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            rotation_mode: str = ROTATION_MODE_AXISANGLE,

            material_roughness: float = 0.1,
            material_ior: float = 1.5,

            use_bumpmap: bool = False,
            bumpmap_dim: int = 100,
            bumpmap: Optional[torch.Tensor] = None,

            merge_vertices: bool = False
    ):
        """
        Set up the initial crystal and calculate the crystal habit.

        Outputs
        .vertices - vertices of crystal habit
        .faces - face groups of indices related to the vertices
        .build_mesh() - returns triangulated mesh vertices and faces
        """
        super().__init__()

        # The 3D unit cell, crystallographic axes a,b,c
        assert len(lattice_unit_cell) == 3, 'Lattice unit cell must be a list of 3 floats'
        self.lattice_unit_cell = lattice_unit_cell

        # The angles alpha, beta, gamma within the unit cell
        assert len(lattice_angles) == 3, 'Lattice angles must be a list of 3 floats'
        self.lattice_angles = lattice_angles

        # Miller indices should be a list of tuples
        assert len(miller_indices) > 0 and all(len(idx) == 3 for idx in miller_indices), \
            'Miller indices must be a list of tuples of length 3'
        self.miller_indices = miller_indices.copy()

        # The symmetry group in Hermann–Mauguin notation
        self.point_group_symbol = point_group_symbol

        # Scale
        self.scale = nn.Parameter(init_tensor(scale), requires_grad=True)

        # Distances from the origin to the faces - should be equal to the number of provided miller indices
        if distances is None:
            distances = torch.ones(len(miller_indices))
        assert len(distances) == len(miller_indices), \
            'Number of distances must be equal to the number of provided miller indices!'
        self.distances = nn.Parameter(init_tensor(distances), requires_grad=True)
        self.distances_logvar = nn.Parameter(torch.ones_like(self.distances) * 2 * torch.log(torch.tensor(0.1)),
                                             requires_grad=True)

        # Origin
        if origin is None:
            origin = [0, 0, 0]
        self.origin = nn.Parameter(init_tensor(origin), requires_grad=True)
        self.origin_logvar = nn.Parameter(torch.ones_like(self.origin) * 2 * torch.log(torch.tensor(0.01)),
                                          requires_grad=True)

        # Rotation
        if rotation is None:
            if rotation_mode == ROTATION_MODE_AXISANGLE:
                rotation = [0, 0, 0]
            else:
                rotation = [1, 0, 0, 0]
        if rotation_mode == ROTATION_MODE_AXISANGLE:
            assert len(rotation) == 3, 'Rotation must be a list of 3 floats for axis-angle representation'
        elif rotation_mode == ROTATION_MODE_QUATERNION:
            assert len(rotation) == 4, 'Rotation must be a list of 4 floats for quaternion representation'
        else:
            raise ValueError(f'Unsupported rotation mode: {rotation_mode}')
        self.rotation_mode = rotation_mode
        self.rotation = nn.Parameter(init_tensor(rotation), requires_grad=True)
        self.rotation_logvar = nn.Parameter(torch.ones_like(self.rotation) * 2 * torch.log(torch.tensor(0.01)),
                                            requires_grad=True)

        # Material
        self.material_roughness = nn.Parameter(init_tensor(material_roughness), requires_grad=True)
        self.material_ior = nn.Parameter(init_tensor(material_ior), requires_grad=True)

        # Bumpmap texture
        if bumpmap_dim == -1:
            if use_bumpmap:
                logger.warning('Bumpmap dimension set to -1, disabling bumpmap')
            use_bumpmap = False
            bumpmap_dim = 10
        if bumpmap is None:
            bumpmap = torch.zeros(bumpmap_dim, bumpmap_dim)
        self.bumpmap = nn.Parameter(init_tensor(bumpmap), requires_grad=True)
        self.bumpmap_dim = bumpmap_dim
        self.use_bumpmap = use_bumpmap
        self.uv_faces = {}  # Location of the face vertices on the UV map

        # Mesh options
        self.merge_vertices = merge_vertices

        # Register buffers
        self.register_buffer('all_distances', torch.empty(0))
        self.register_buffer('N', torch.empty(0))
        self.register_buffer('vertices', torch.empty(0))
        self.register_buffer('mesh_vertices', torch.empty(0))
        self.register_buffer('mesh_faces', torch.empty(0, dtype=torch.int64))
        self.register_buffer('uv_mask', torch.empty(0, dtype=torch.bool))

        # Initialise the crystal
        with torch.no_grad():
            self._init()

    def to(self, *args, **kwargs):
        """
        Override the to method to ensure that the uv_faces are also moved to the correct device.
        """
        super().to(*args, **kwargs)
        for k, v in self.uv_faces.items():
            self.uv_faces[k] = v.to(*args, **kwargs)

    def clamp_parameters(self):
        """
        Clamp the parameters to a valid range.
        """
        with torch.no_grad():
            d = torch.clamp(self.distances, 1e-1, None)
            d = torch.sort(d, descending=True).values  # todo: parameterise or remove
            sf = 1 / d.amax()  # Fix the max distance to 1 and adjust scale
            self.distances.data = d * sf
            self.scale.data = torch.clamp(self.scale / sf, 1e-8, 1e8)

            self.origin.data = torch.clamp(self.origin, -1e3, 1e3)
            if self.rotation_mode == ROTATION_MODE_AXISANGLE:
                rv_norm = self.rotation.norm()
                if rv_norm > 2 * torch.pi:
                    rv_norm2 = torch.remainder(self.rotation.norm(), 2 * torch.pi)
                    self.rotation.data = self.rotation / rv_norm * rv_norm2
            elif self.rotation_mode == ROTATION_MODE_QUATERNION:
                rv_norm = self.rotation.norm()
                self.rotation.data = self.rotation / rv_norm
            else:
                raise ValueError(f'Unsupported rotation mode: {self.rotation_mode}')

    def _init(self):
        """
        Calculate symmetries and build the crystal habit.
        """
        # Step 1: Find all miller indices based of symmetry groups
        self._calculate_symmetry()

        # Step 2: Calculate lattice unit cell scaling
        self._calculate_reciprocal_lattice_vectors()

        # Step 3: Calculate face normal vectors
        mi = self.all_miller_indices.to(torch.float32)
        self.N = normalise(
            torch.matmul(self.lattice_vectors_star, mi.T).T
        )

        # Step 4: Build the crystal habit
        self.build_mesh()

    def _calculate_symmetry(self):
        """
        Calculate symmetries for a given Hermann–Mauguin point group and complete the full set of symmetric distances.
        This only needs doing once.
        """
        sym_group = PointGroup(self.point_group_symbol)
        symmetry_operations = sym_group.symmetry_ops

        all_miller_indices = self.miller_indices.copy()
        all_distances = self.distances.clone()
        symmetry_idx = list(range(len(all_distances)))
        for idx, (indices, dist) in enumerate(zip(self.miller_indices, self.distances)):
            for sym_op in symmetry_operations:
                rotated_indices = sym_op.operate(indices)
                # Check if the rotated indices already exist
                found = False
                for (mi, d) in zip(all_miller_indices, all_distances):
                    if np.array_equal(mi, rotated_indices):
                        found = True
                        break
                if not found:
                    all_miller_indices.append(rotated_indices)
                    all_distances = torch.cat([all_distances, dist[None, ...]])
                    symmetry_idx.append(idx)

        # Convert to tensors and attach to self
        self.all_miller_indices = torch.from_numpy(np.stack(all_miller_indices)).to(torch.int32)
        self.all_distances = all_distances
        self.symmetry_idx = torch.tensor(symmetry_idx)

    def _calculate_reciprocal_lattice_vectors(self):
        """
        Calculate the reciprocal lattice vectors.
        """
        a, b, c = self.lattice_unit_cell
        alpha, beta, gamma = np.radians(self.lattice_angles)

        # Calculate volume of the unit cell
        volume = (a * b * c
                  * np.sqrt(
                    1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
                    + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)))

        # Calculate reciprocal lattice vectors
        self.lattice_vectors_star = (2 * torch.pi / volume) * torch.diag(torch.tensor([
            b * c * np.sin(alpha),
            c * a * np.sin(beta),
            a * b * np.sin(gamma)
        ], dtype=torch.float32))

    def build_mesh(
            self,
            scale: Optional[torch.Tensor] = None,
            distances: Optional[torch.Tensor] = None,
            origin: Optional[torch.Tensor] = None,
            rotation: Optional[torch.Tensor] = None,
            tol: float = 1e-3
    ):
        """
        Take the face normals and distances and calculate where the vertices and edges lie.
        """
        device = self.origin.device
        if scale is None:
            scale = self.scale
        if distances is None:
            distances = self.distances
        if origin is None:
            origin = self.origin
        if rotation is None:
            rotation = self.rotation

        # Step 0: Update the distances, taking account of symmetries
        distances = torch.clip(distances, 0, None)
        self.all_distances = distances[self.symmetry_idx]

        # Step 1: Calculate all intersection points of all combinations of 3 planes
        all_combinations = list(combinations(range(len(self.N)), 3))
        intersection_points = []
        for combo in all_combinations:
            idx1, idx2, idx3 = combo
            point = self._plane_intersection(idx1, idx2, idx3)
            if point is not None:
                intersection_points.append(point)
        intersection_points = torch.stack(intersection_points)

        # Step 2: Only select points that are in the polyhedron
        T = intersection_points @ self.N.T
        R = torch.all(T <= self.all_distances + tol, dim=1)
        vertices = intersection_points[R]

        # Step 3: Take vertices, assign them to a face, and define faces
        R = self.N @ vertices.T
        exp_dist = self.all_distances.unsqueeze(1).expand_as(R)
        T = torch.abs(R - exp_dist) <= tol
        faces = {}
        mesh_vertices = []
        mesh_faces = []
        v_idx = 0
        for i, on_face in enumerate(T):
            face_vertex_idxs = torch.nonzero(on_face).squeeze()

            # If the face does not have enough vertices, skip it
            if len(face_vertex_idxs) < 3:
                continue

            # Merge nearby vertices
            face_vertices = vertices[on_face]
            if self.merge_vertices:
                face_vertices, _ = merge_vertices(face_vertices, tol)
            if len(face_vertices) < 3:
                continue
            centroid = torch.mean(face_vertices, dim=0)

            # Calculate the angles of each vertex relative to the centroid
            angles = calculate_relative_angles(face_vertices, centroid)

            # Sort the vertices based on the angles
            sorted_idxs = torch.argsort(angles)
            sorted_vertices = face_vertices[sorted_idxs]

            # Flip the order if the normal is pointing inwards
            normal = torch.cross(sorted_vertices[0] - centroid, sorted_vertices[1] - centroid, dim=0)
            if torch.dot(normal, centroid) < 0:
                sorted_idxs = sorted_idxs.flip(0)
            sorted_face_vertex_idxs = face_vertex_idxs[sorted_idxs]
            faces[i] = sorted_face_vertex_idxs

            # Build the triangular faces
            mfv = torch.stack([centroid, *face_vertices[sorted_idxs]])
            N = len(face_vertices)
            jdx = torch.arange(N, device=device)
            mfi = torch.stack([torch.zeros(N, device=device, dtype=torch.int64), jdx % N + 1, (jdx + 1) % N + 1])
            mfi = mfi + v_idx
            v_idx += N + 1
            mesh_vertices.append(mfv)
            mesh_faces.append(mfi)

        if len(mesh_faces) < 3:
            raise ValueError('Not enough faces to build a mesh!')

        # Step 4: Merge mesh vertices and correct face indices
        mesh_vertices = torch.cat(mesh_vertices)
        mesh_faces = torch.cat(mesh_faces, dim=1).T
        if self.merge_vertices:
            mesh_vertices, cluster_idxs = merge_vertices(mesh_vertices)
            mesh_faces = cluster_idxs[mesh_faces]

        # Step 5: Apply the rotation
        if self.rotation_mode == ROTATION_MODE_AXISANGLE:
            if torch.allclose(rotation, torch.zeros(3, device=device)):
                rotation = rotation + torch.randn(3, device=device) * 1e-8
            R = axis_angle_to_rotation_matrix(rotation[None, :]).squeeze(0)
        elif self.rotation_mode == ROTATION_MODE_QUATERNION:
            R = quaternion_to_rotation_matrix(rotation[None, :]).squeeze(0)
        else:
            raise ValueError(f'Unsupported rotation mode: {self.rotation_mode}')
        vertices = vertices @ R.T
        mesh_vertices = mesh_vertices @ R.T

        # Step 6: Apply the scaling
        vertices = vertices * scale
        mesh_vertices = mesh_vertices * scale

        # Step 7: Apply the origin translation
        vertices = vertices + origin
        mesh_vertices = mesh_vertices + origin

        # Set properties to self
        self.vertices = vertices
        self.faces = faces
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces

        # Step 8: Construct the UV map
        self._build_uv_map()

        return self.mesh_vertices.clone(), self.mesh_faces.clone()

    def _plane_intersection(self, n1: int, n2: int, n3: int) -> Optional[torch.Tensor]:
        """
        Calculate the intersection point of three planes.
        """
        A = torch.stack([self.N[n1], self.N[n2], self.N[n3]])
        # Check if the planes form a basis in 3-space
        if torch.det(A).abs() < 1e-6:
            return None
        b = torch.stack([self.all_distances[n1], self.all_distances[n2], self.all_distances[n3]])
        intersection_point = torch.linalg.solve(A, b.unsqueeze(1))
        return intersection_point.squeeze()

    def _build_uv_map(self):
        """
        Construct the UV map of the crystal habit.
        """
        if self.merge_vertices:
            raise RuntimeError('Cannot build UV map with merged vertices!')
        if not self.use_bumpmap or self.bumpmap is None:
            return
        device = self.origin.device

        # Detach the faces and vertices from the computation
        f, v = self.mesh_faces.detach(), self.mesh_vertices.detach()
        faces = {}
        uv_faces = {}

        # Construct a grid where each cell contains the planar coordinates of a face
        centroid_idxs = f[:, 0].unique(dim=0)
        grid_dim = np.sqrt(len(centroid_idxs)).astype(int) + 1
        rows = [[] for _ in range(grid_dim)]
        row_heights = [0] * grid_dim
        col_widths = [0] * grid_dim
        col_idxs = torch.stack([torch.arange(4)] * 4)
        row_idxs = col_idxs.T
        sorted_idxs = torch.argsort((row_idxs**2 + col_idxs**2).flatten())
        positions = torch.zeros_like(sorted_idxs)
        positions[sorted_idxs] = torch.arange(len(positions))
        positions = positions.reshape(4, 4)

        # Calculate the planar coordinates for each face
        for i, c_idx in enumerate(centroid_idxs):
            # Extract the face vertices and calculate the planar coordinates
            face = f[f[:, 0] == c_idx]
            fv_idxs = face.flatten().unique()
            faces[i] = fv_idxs
            try:
                v2 = align_points_to_xy_plane(v[fv_idxs[1:]])[:, :2]
                v2 = rotate_2d_points_to_square(v2)  # Rotate so the bounding box is square
            except Exception:
                # Align vertices in a circle around the centroid
                theta = torch.linspace(0, 2 * torch.pi, steps=len(fv_idxs), device=device)[:-1]
                v2 = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)

            # Add the centroid back in
            v2 = torch.cat([torch.zeros((1, 2), device=device), v2])

            # Put close to the origin, but keep positive
            v2 = v2 - v2.amin(dim=0)

            # Add the planar coordinates to the grid and update the row heights and column widths
            cell_pos = (positions == i).nonzero()[0]
            row_idx, col_idx = cell_pos[0].item(), cell_pos[1].item()
            rows[row_idx].append(v2)
            row_heights[row_idx] = max(row_heights[row_idx], v2[:, 1].max().item())
            col_widths[col_idx] = max(col_widths[col_idx], v2[:, 0].max().item())

        # Add a little padding
        pad = max(max(row_heights), max(col_widths)) * 0.1
        row_heights = [h + pad for h in row_heights]
        col_widths = [w + pad for w in col_widths]

        # Centre each set of coordinates in their cells, offset them and scale to [0,1]
        uv_map = torch.zeros((len(v), 2), device=device)
        sf = 1 / max(sum(row_heights), sum(col_widths))
        for row_idx, row in enumerate(rows):
            for col_idx, v2 in enumerate(row):
                cell_midpoint = torch.tensor([col_widths[col_idx], row_heights[row_idx]], device=device) / 2
                offset = torch.tensor([sum(col_widths[:col_idx]), sum(row_heights[:row_idx])], device=device)
                v2 = v2 - v2.mean(dim=0) + cell_midpoint + offset
                v2 = v2 * sf
                rows[row_idx][col_idx] = v2
                face_idx = positions[row_idx, col_idx].item()
                vertex_idxs = faces[face_idx]
                uv_map[vertex_idxs] = v2
                uv_faces[face_idx] = v2

        # Construct a mask image showing where the crystal faces lie on the UV map.
        uv_mask = torch.zeros_like(self.bumpmap)[None, None, ...]
        colour = torch.tensor([1.0], device=uv_mask.device)
        for face_uv in uv_faces.values():
            face_uv = face_uv[1:]  # Skip the centroid
            img_coords = (face_uv * (self.bumpmap_dim - 1)).round().to(torch.int64)
            uv_mask = draw_convex_polygon(uv_mask, img_coords[None, ...], colour)
        uv_mask = uv_mask.squeeze().to(torch.bool)

        self.uv_map = uv_map
        self.uv_faces = uv_faces
        self.uv_mask = uv_mask
        # plot_uv_map(rows, row_heights, col_widths, sf, mask)


def plot_uv_map(rows, row_heights, col_widths, sf, uv_mask):
    import matplotlib.pyplot as plt
    from crystalsizer3d.util.utils import to_numpy

    # Plot the UV map
    fig, ax = plt.subplots()
    ax.imshow(to_numpy(uv_mask), extent=(0, 1, 0, 1), origin='lower', cmap='gray')
    ax.axhline(y=0, color='k')
    for row_idx, row in enumerate(rows):
        ax.axhline(y=sum(row_heights[:row_idx]) * sf, color='k')
        for col_idx, v2 in enumerate(row):
            if row_idx == 0:
                ax.axvline(x=sum(col_widths[:col_idx]) * sf, color='k')
                if col_idx == len(row) - 1:
                    ax.axvline(x=sum(col_widths) * sf, color='k')
            ax.plot(*to_numpy(v2).T, '-o')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()
