from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA
from crystalsizer3d.crystal import Crystal, ROTATION_MODE_AXISANGLE
from crystalsizer3d.crystal_renderer import Scene
from crystalsizer3d.csd_proxy import CSDProxy
from crystalsizer3d.scene_components.bubble import make_bubbles
from crystalsizer3d.scene_components.textures import NoiseTexture, NormalMapNoiseTexture, generate_crystal_bumpmap, \
    generate_noise_map
from crystalsizer3d.util.utils import to_numpy

if USE_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

save_plots = False
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
    csd_proxy = CSDProxy()
    cs = csd_proxy.load('LGLUAC11')
    miller_indices = [(1, 0, 1), (0, 2, 1), (0, 1, 0)]

    crystal = Crystal(
        lattice_unit_cell=cs.lattice_unit_cell,
        lattice_angles=cs.lattice_angles,
        miller_indices=miller_indices,
        point_group_symbol=cs.point_group_symbol,
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


def make_noise():
    for freq in [0.002]:  # , 0.02, 0.05]:
        for oct in [1, 2, 3, 4, 10]:
            # for oct in [4]:
            for ns in [0.01]:  # , 0.2, 0.3]:
                n = generate_noise_map(
                    perlin_freq=freq,
                    perlin_octaves=oct,
                    white_noise_scale=ns
                )
                plt.title(f'freq={freq}, oct={oct}, ns={ns}')
                plt.imshow(n, cmap='gray')
                plt.colorbar()
                plt.show()


def plot_scene():
    """
    Plot the scene with some interference patterns.
    """
    spp = 2**9

    # Bubble parameters
    n_bubbles = 0
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
    n_defects = 5
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
    crystal.bumpmap.data = generate_crystal_bumpmap(
        crystal=crystal,
        n_defects=n_defects,
        defect_min_width=defect_min_width,
        defect_max_width=defect_max_width,
        defect_max_z=defect_max_z,
    )

    # Generate a surface texture map
    cell_bumpmap = NormalMapNoiseTexture(
        dim=crystal.bumpmap_dim,
        perlin_freq=10.,
        perlin_octaves=5,
        white_noise_scale=0.1,
        max_amplitude=0.5,
        seed=0
    )

    # Generate a light radiance texture
    light_radiance_texture = NoiseTexture(
        dim=crystal.bumpmap_dim,
        channels=3,
        perlin_freq=5.,
        perlin_octaves=8,
        white_noise_scale=0.2,
        max_amplitude=0.5,
        zero_centred=True,
        shift=1,
        seed=0,
    )

    # Render the scene
    scene = Scene(
        crystal=crystal,
        bubbles=bubbles,
        spp=spp,
        light_radiance=(0.6, 0.5, 0.3),
        light_st_texture=light_radiance_texture,
        cell_bumpmap=cell_bumpmap
    )
    image = scene.render()

    # Plot it
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes[0, 0]
    ax.imshow(image)
    ax.axis('off')

    ax = axes[0, 1]
    ax.set_title('Crystal bumpmap')
    ax.imshow(to_numpy(crystal.bumpmap - crystal.uv_mask.to(torch.float32)), cmap='gray')

    ax = axes[1, 0]
    ax.set_title('Cell bumpmap')
    ax.imshow(to_numpy(cell_bumpmap.build()))
    ax.axis('off')

    ax = axes[1, 1]
    ax.set_title('Light radiance texture')
    ax.imshow(to_numpy(light_radiance_texture.build()))
    ax.axis('off')

    fig.tight_layout()

    if save_plots:
        LOGS_PATH.mkdir(parents=True, exist_ok=True)
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_bubbles={n_bubbles}_defects={n_defects}_spp={spp}.png')
    if show_plots:
        plt.show()


if __name__ == '__main__':
    plot_scene()
    # make_noise()
