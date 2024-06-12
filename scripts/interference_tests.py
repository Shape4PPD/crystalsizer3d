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
from crystalsizer3d.util.utils import set_seed, to_numpy

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
        align_to_floor: bool = True
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

    if align_to_floor:
        crystal.origin.data[2] -= crystal.vertices[:, 2].min()

    # Rebuild the mesh
    crystal.build_mesh()
    # v2, f2 = crystal.build_mesh()
    # mesh = Trimesh(vertices=to_numpy(v2), faces=to_numpy(f2))
    # mesh.show()

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
    set_seed(10)
    spp = 2**9

    # Bubble parameters
    n_bubbles = 50
    bubbles_min_scale = 0.001
    bubbles_max_scale = 0.1
    bubbles_min_roughness = 0.05
    bubbles_max_roughness = 0.2
    bubbles_min_ior = 1.3
    bubbles_max_ior = 2.3

    # Bumpmap defects
    n_defects = 5
    defect_max_z = 1
    defect_min_width = 0.001
    defect_max_width = 0.01

    # Create the crystal
    crystal = _generate_crystal(
        distances=[1.0, 0.5, 0.2],
        scale=1,
        origin=[0, 0, 0],
        rotvec=[-np.pi / 2, 0, np.pi / 2],
    )

    # Cuboid test
    # crystal = Crystal(
    #     lattice_unit_cell=[1, 1, 1],
    #     lattice_angles=[90, 90, 90],
    #     miller_indices=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    #     point_group_symbol='222',
    #     scale=5,
    #     origin=[0, 0, 1],
    #     distances=[1., 1., 0.1],
    #     rotation=[0, 0.4, 0.5],
    #     material_roughness=0.05,
    #     material_ior=2
    # )
    # crystal.origin.data[2] -= crystal.vertices[:, 2].min()
    # crystal.build_mesh()
    # crystal.to(device)
    # print(crystal.vertices)

    # Create some bubbles
    if n_bubbles > 0:
        bubbles = make_bubbles(
            n_bubbles=n_bubbles,
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
        # max_amplitude=10
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
        shift=1
    )

    # Render the scene
    scene = Scene(
        crystal=crystal,
        bubbles=bubbles,
        spp=spp,
        remesh_max_edge=None,
        # remesh_max_edge=0.05,

        camera_distance=32.,
        focus_distance=30.,
        # focal_length=29.27,
        camera_fov=10.2,
        aperture_radius=0.3,

        light_z_position=-5.1,
        light_scale=10.,
        light_radiance=(0.6, 0.5, 0.3),

        cell_z_positions=[-5, 0., 5., 10.],
        cell_surface_scale=5.7 / 2,
        cell_render_blanks=False,

        light_st_texture=light_radiance_texture,
        cell_bumpmap=cell_bumpmap
    )
    # scene.place_crystal(
    #     min_area=0.1,
    #     max_area=0.3,
    #     centre_crystal=False,
    # )
    print(crystal.vertices)
    scene.place_bubbles(
        min_scale=bubbles_min_scale,
        max_scale=bubbles_max_scale
    )
    image = scene.render()

    # Plot just the rendered image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')
    fig.tight_layout()
    if save_plots:
        LOGS_PATH.mkdir(parents=True, exist_ok=True)
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_bubbles={n_bubbles}_defects={n_defects}_spp={spp}.png')
    if show_plots:
        plt.show()

    # Plot all the components
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes[0, 0]
    ax.imshow(image)

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
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_bubbles={n_bubbles}_defects={n_defects}_spp={spp}_components.png')
    if show_plots:
        plt.show()


if __name__ == '__main__':
    plot_scene()
    # make_noise()
