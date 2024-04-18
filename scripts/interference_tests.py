from typing import List, Tuple

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
from PIL import Image, ImageDraw
from ccdc.io import EntryReader

from crystalsizer3d import USE_CUDA
from crystalsizer3d.crystal import Crystal, ROTATION_MODE_AXISANGLE
from crystalsizer3d.crystal_renderer_mitsuba import Bubble, build_mitsuba_mesh

if USE_CUDA:
    if 'cuda_ad_rgb' not in mi.variants():
        raise RuntimeError('No CUDA variant found.')
    mi.set_variant('cuda_ad_rgb')
    device = torch.device('cuda')
else:
    mi.set_variant('llvm_ad_rgb')
    device = torch.device('cpu')

from mitsuba import ScalarTransform4f as T

# dr.set_log_level(dr.LogLevel.Info)
# dr.set_thread_count(1)

# save_plots = True
# show_plots = False
save_plots = False
show_plots = True

SHAPE_NAME = 'crystal'
VERTEX_KEY = SHAPE_NAME + '.vertex_positions'
FACES_KEY = SHAPE_NAME + '.faces'
BSDF_KEY = SHAPE_NAME + '.bsdf'
COLOUR_KEY = BSDF_KEY + '.reflectance.value'


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


def create_scene(
        crystal: Crystal,
        bubbles: List[Bubble],
        radiance: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        spp: int = 256,
        res: int = 400
) -> mi.Scene:
    """
    Create a Mitsuba scene containing the given crystal.
    """
    scene_dict = {
        'type': 'scene',

        # Camera and rendering parameters
        'integrator': {
            'type': 'path',
        },
        'sensor': {
            'type': 'thinlens',
            'aperture_radius': 0.5,
            'focus_distance': 90,
            'fov': 25,
            'to_world': T.look_at(
                origin=[0, 0, 100],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'stratified',  # seems better than independent
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': res,
                'height': res,
                'filter': {'type': 'gaussian'},
                'sample_border': True,
            },
        },

        # Emitters
        'light': {
            'type': 'rectangle',
            'to_world': T.scale(50),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': radiance
                }
            },
        },

        # Shapes
        'surface': {
            'type': 'rectangle',
            'to_world': T.translate([0, 0, 1]) @ T.scale(25),
            'surface_material': {
                'type': 'dielectric',
                'int_ior': 1.,
                # 'specular_transmittance': {
                #     'type': 'bitmap',
                #     # 'bitmap': mi.Bitmap(dr.ones(mi.TensorXf, (12, 12, 3)))
                #     # 'bitmap': mi.Bitmap(surf),
                #     'filename': str(ROOT_PATH / 'tmp' / 'grid_1000x1000.png'),
                #     # 'type': 'rgb',
                #     # 'value': (1,0,0),
                #     'wrap_mode': 'clamp',
                # }
            },
        },
        SHAPE_NAME: build_mitsuba_mesh(crystal)
    }

    # Add bubbles
    for bubble in bubbles:
        scene_dict[bubble.SHAPE_NAME] = bubble.build_mesh()

    # Load the scene
    scene = mi.load_dict(scene_dict)

    return scene


def plot_scene():
    """
    Plot the scene with some interference patterns.
    """
    spp = 2**9

    # Bubble parameters
    n_bubbles = 0
    bubbles_min_x = -10
    bubbles_max_x = 10
    bubbles_min_y = -10
    bubbles_max_y = 10
    bubbles_min_z = 0
    bubbles_max_z = 10
    bubbles_min_scale = 0.001
    bubbles_max_scale = 1
    bubbles_min_alpha = 0.05
    bubbles_max_alpha = 0.2
    bubbles_min_ior = 1.1
    bubbles_max_ior = 1.8

    # Bumpmap defects
    n_defects = 200
    defect_max_z = 1
    defect_min_length = 0.001
    defect_max_length = 0.4
    defect_min_width = 0.0001
    defect_max_width = 0.001

    # Create the crystal
    crystal = _generate_crystal(
        distances=[1.0, 0.5, 0.2],
        scale=10,
        origin=[0, 0, 20],
        rotvec=[np.pi / 5, 0, np.pi / 4],
    )

    # Create some bubbles
    bubbles = []
    for i in range(n_bubbles):
        bubble = Bubble(
            shape_name=f'bubble_{i}',
            origin=[
                np.random.uniform(bubbles_min_x, bubbles_max_x),
                np.random.uniform(bubbles_min_y, bubbles_max_y),
                np.random.uniform(bubbles_min_z, bubbles_max_z),
            ],
            scale=np.random.uniform(bubbles_min_scale, bubbles_max_scale),
            roughness=np.random.uniform(bubbles_min_alpha, bubbles_max_alpha),
            ior=np.random.uniform(bubbles_min_ior, bubbles_max_ior),
        )
        bubble.to(device)
        bubbles.append(bubble)

    # Draw some line defects onto the bumpmap
    bumpmap_dim = crystal.bumpmap.shape[0]
    bumpmap = Image.new('L', (bumpmap_dim, bumpmap_dim), color=int(255 / 2))
    draw = ImageDraw.Draw(bumpmap)
    for i in range(n_defects):
        z = int(np.random.uniform() * 255)
        l = max(1, int(np.random.uniform(defect_min_length, defect_max_length) * bumpmap.width))
        w = max(1, int(np.random.uniform(defect_min_width, defect_max_width) * bumpmap.width))

        # Pick random start point
        x0 = np.random.randint(0, bumpmap.width)
        y0 = np.random.randint(0, bumpmap.height)

        # Pick a random angle and calculate end point
        angle = np.random.uniform(0, 2 * np.pi)
        x1 = x0 + int(l * np.cos(angle))
        y1 = y0 + int(l * np.sin(angle))

        # Draw line between points
        draw.line((x0, y0, x1, y1), fill=z, width=w)
    bumpmap = (np.array(bumpmap).astype(np.float32) / 255 - 0.5) * 2 * defect_max_z
    crystal.bumpmap.data = torch.from_numpy(bumpmap).to(device)

    scene = create_scene(
        crystal=crystal,
        bubbles=bubbles,
        radiance=(0.6, 0.5, 0.3),
        spp=spp
    )
    image = mi.render(scene)
    plt.imshow(image**(1.0 / 2.2))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    plot_scene()
    # optimise_scene()
    # track_losses()
