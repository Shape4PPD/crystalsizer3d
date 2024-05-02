import cv2
import numpy as np
import torch

from crystalsizer3d.crystal import Crystal
from crystalsizer3d.util.utils import line_equation_coefficients, line_intersection, normalise, to_numpy


def generate_bumpmap(
        crystal: Crystal,
        n_defects: int,
        defect_min_width: float = 0.0001,
        defect_max_width: float = 0.001,
        defect_max_z: float = 1,
) -> torch.Tensor:
    """
    Create a bumpmap for the crystal with some line defects.
    """
    dim = crystal.bumpmap_dim
    device = crystal.origin.device
    bumpmap = np.zeros((dim, dim), dtype=np.float32)

    for i in range(n_defects):
        # Pick a random face
        face_idx = np.random.randint(0, len(crystal.faces))
        centroid_uv = crystal.uv_faces[face_idx][0]
        face_uv = crystal.uv_faces[face_idx][1:]

        # Pick two random adjacent vertices
        v0_idx = np.random.randint(0, len(face_uv))
        v1_idx = (v0_idx + 1) % len(face_uv)
        v0 = face_uv[v0_idx]
        v1 = face_uv[v1_idx]
        edge = v1 - v0
        midpoint = (v0 + v1) / 2

        # Find where a perpendicular line from the middle of this edge would intersect another edge
        l_perp = line_equation_coefficients(v0, v1, perpendicular=True)
        max_dist = np.inf
        for j in range(len(face_uv)):
            if j == v0_idx:
                continue
            u0 = face_uv[j]
            u1 = face_uv[(j + 1) % len(face_uv)]
            l = line_equation_coefficients(u0, u1)
            intersect = line_intersection(l, l_perp)
            if intersect is None:
                continue
            if torch.dot(intersect - midpoint, centroid_uv - midpoint) < 0:
                continue
            max_dist = min(max_dist, (intersect - midpoint).norm().item())

        # Pick a random perpendicular distance
        d = np.random.uniform(0, max_dist)

        # Pick a random line length
        edge_length = edge.norm().item()
        l = np.random.uniform(edge_length * 0.1, edge_length * 2)

        # Pick a random midpoint offset
        offset = np.random.uniform(-0.2, 0.2)

        # Calculate end points of the defect line parallel to the edge
        perp_vec = normalise(torch.tensor([-l_perp[1], l_perp[0]], device=device))
        if torch.dot(perp_vec, centroid_uv - midpoint) < 0:
            perp_vec = -perp_vec
        defect_start = midpoint + d * perp_vec - l * (0.5 + offset) * edge / edge_length
        defect_end = midpoint + d * perp_vec + l * (0.5 - offset) * edge / edge_length

        # Convert to image coordinates
        x0, y0 = to_numpy((defect_start * (dim - 1)).round().to(torch.int64))
        x1, y1 = to_numpy((defect_end * (dim - 1)).round().to(torch.int64))

        # Draw line between points
        w = max(1, int(np.random.uniform(defect_min_width, defect_max_width) * dim))
        defect = np.zeros((dim, dim), dtype=np.uint8)
        cv2.line(defect, (x0, y0), (x1, y1), 255, w, cv2.LINE_AA)

        # Scale the bump depth and add some noise to the line
        defect = defect.astype(np.float32) / 255  # * z
        line = defect > 0.1
        noise = np.random.normal(np.zeros(line.sum()), 0.2)
        defect[line] += noise
        z = (np.random.uniform() * 2 - 1) * defect_max_z
        defect[line] *= z

        # Add the defect to the bumpmap
        defect = defect * to_numpy(crystal.uv_mask)
        bumpmap += defect
        bumpmap = np.clip(bumpmap, -defect_max_z, defect_max_z)

    return torch.from_numpy(bumpmap).to(device)
