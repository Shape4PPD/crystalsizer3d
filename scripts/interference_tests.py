from typing import List

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
from ccdc.io import EntryReader

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA
from crystalsizer3d.crystal import Crystal, ROTATION_MODE_AXISANGLE
from crystalsizer3d.crystal_renderer import Scene
from crystalsizer3d.scene_components.bubble import make_bubbles
from crystalsizer3d.scene_components.bumpmap import generate_bumpmap
from crystalsizer3d.util.utils import to_numpy

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('llvm_ad_rgb')
    device = torch.device('cpu')

save_plots = True
show_plots = True


def _generate_crystal(
        distances: List[float] = [1.0, 0.5, 0.2],
        scale: float = 10.,
        origin: List[float] = [0, 0, 20],
        rotvec: List[float] = [0, 0, 0],
) -> Crystal:
    """
    Generate a beta-form LGA crystal.
    """
    reader = EntryReader()
    crystal = reader.crystal('LGLUAC11')
    miller_indices = [(1, 0, 1), (0, 2, 1), (0, 1, 0)]
    lattice_unit_cell = [crystal.cell_lengths[0], crystal.cell_lengths[1], crystal.cell_lengths[2]]
    lattice_angles = [crystal.cell_angles[0], crystal.cell_angles[1], crystal.cell_angles[2]]
    point_group_symbol = '222'  # crystal.spacegroup_symbol

    crystal = Crystal(
        lattice_unit_cell=lattice_unit_cell,
        lattice_angles=lattice_angles,
        miller_indices=miller_indices,
        point_group_symbol=point_group_symbol,
        distances=torch.tensor(distances) * scale,
        origin=origin,
        rotation=rotvec,
        rotation_mode=ROTATION_MODE_AXISANGLE,
        material_roughness=0.02,
        material_ior=1.78,
        use_bumpmap=True,
        bumpmap_dim=1000
    )
    crystal.to(device)

    # Rebuild the mesh
    crystal.build_mesh()
    # v2, f2 = crystal.build_mesh()
    # mesh = Trimesh(vertices=to_numpy(v2), faces=to_numpy(f2))
    # mesh_uv = mesh.unwrap()
    # mesh2.show()
    # exit()

    return crystal


def plot_scene():
    """
    Plot the scene with some interference patterns.
    """
    spp = 2**9

    # Bubble parameters
    n_bubbles = 50
    bubbles_min_x = -20
    bubbles_max_x = 20
    bubbles_min_y = -20
    bubbles_max_y = 20
    bubbles_min_z = 0
    bubbles_max_z = 20
    bubbles_min_scale = 0.001
    bubbles_max_scale = 1
    bubbles_min_roughness = 0.05
    bubbles_max_roughness = 0.2
    bubbles_min_ior = 1.1
    bubbles_max_ior = 1.8

    # Bumpmap defects
    n_defects = 20
    defect_max_z = 1
    defect_min_width = 0.001
    defect_max_width = 0.01

    # Create the crystal
    crystal = _generate_crystal(
        distances=[1.0, 0.5, 0.2],
        scale=10,
        origin=[0, 0, 20],
        rotvec=[np.pi / 5, 0, np.pi / 4],
    )

    # Create some bubbles
    if n_bubbles > 0:
        bubbles = make_bubbles(
            n_bubbles=n_bubbles,
            min_x=bubbles_min_x,
            max_x=bubbles_max_x,
            min_y=bubbles_min_y,
            max_y=bubbles_max_y,
            min_z=bubbles_min_z,
            max_z=bubbles_max_z,
            min_scale=bubbles_min_scale,
            max_scale=bubbles_max_scale,
            min_roughness=bubbles_min_roughness,
            max_roughness=bubbles_max_roughness,
            min_ior=bubbles_min_ior,
            max_ior=bubbles_max_ior,
            device=device,
        )
    else:
        bubbles = []

    # Generate a defect bumpmap
    crystal.bumpmap.data = generate_bumpmap(
        crystal=crystal,
        n_defects=n_defects,
        defect_min_width=defect_min_width,
        defect_max_width=defect_max_width,
        defect_max_z=defect_max_z,
    )

    # Render the scene
    scene = Scene(
        crystal=crystal,
        bubbles=bubbles,
        spp=spp,
        light_radiance=(0.6, 0.5, 0.3),
    )
    image = scene.render()

    # Plot it
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[1].imshow(to_numpy(crystal.bumpmap - crystal.uv_mask.to(torch.float32)), cmap='gray')
    axes[1].axis('off')
    fig.tight_layout()

    if save_plots:
        LOGS_PATH.mkdir(parents=True, exist_ok=True)
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_bubbles={n_bubbles}_defects={n_defects}_spp={spp}.png')
    if show_plots:
        plt.show()


if __name__ == '__main__':
    plot_scene()
