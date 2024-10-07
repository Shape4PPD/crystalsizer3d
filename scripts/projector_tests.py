import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.utils import tensor_to_image
from trimesh import Trimesh

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.projector import Projector
from crystalsizer3d.scene_components.scene import Scene
from crystalsizer3d.scene_components.utils import project_to_image
from crystalsizer3d.util.keypoints import generate_keypoints_heatmap
from crystalsizer3d.util.utils import init_tensor, to_numpy

if USE_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

TEST_CRYSTALS = {
    'cube': {
        'lattice_unit_cell': [1, 1, 1],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'point_group_symbol': '222',
        'scale': 1,
        'origin': [0.5, 0, 0],
        'distances': [1., 1., 1.],
        'rotation': [0.2, -2.2, 0.2],
        'material_ior': 1.2,
        'material_roughness': 0.01
    },
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
        'point_group_symbol': '222',
        'scale': 12,
        'material_ior': 1.7,
        # 'origin': [-2.2178, -0.9920, 5.7441],
        'origin': [0., 0., 0.],
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
        'scale': 2.5,
        'material_ior': 1.5,
        'material_roughness': 0.001
        # 'origin': [-2.2178, -0.9920,  5.7441],
        # 'rotation': [0.6168,  0.3305, -0.4568],
    },
    'alpha4': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 0, 0), (1, 1, 0), (0, 0, -1), (0, -1, -1),
                           (0, 1, -1), (0, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1),
                           (1, 1, -1), (-1, 0, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'distances': [0.24734821915626526, 0.9126611948013306, 0.7329514622688293, 0.828283965587616,
                      0.5967196822166443, 0.8582077026367188, 0.28573378920555115, 1.0, 0.9321545362472534,
                      0.9787896275520325, 0.8214888572692871, 0.7762473225593567, 0.8469175696372986,
                      0.7356109023094177, 0.7443752884864807, 0.6985854506492615, 0.6237171292304993,
                      0.9178063273429871, 0.8962163925170898, 0.9747581481933594],
        'point_group_symbol': '1',
        'scale': 1,
        'origin': [-2.8313868045806885, -0.6120783090591431, 18.232881367206573],
        # 'origin': [0, 0, 18.232881367206573],
        # 'rotation': [1.2772717475891113, 0.77781081199646, 1.0901552438735962],
        'rotation': [0, 0.1, 0.2],
        'material_ior': 1.8,
        'material_roughness': 0.01
    },
    'alpha5': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 0, 0), (1, 1, 0), (0, 0, -1), (0, -1, -1),
                           (0, 1, -1), (0, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1),
                           (1, 1, -1), (-1, 0, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'distances': [0.2410222887992859, 0.8811460137367249, 0.6749506592750549, 0.7014448046684265,
                      0.9037846326828003, 0.7973584532737732, 0.27292102575302124, 0.8122904300689697,
                      0.9151954650878906, 0.9412405490875244, 0.6824771761894226, 0.7488821744918823,
                      0.653540313243866, 0.7903946042060852, 0.6622034907341003, 0.6827501654624939,
                      1.0, 0.8102344870567322, 0.8926759362220764, 0.8574218153953552],
        'origin': [-0.3571832776069641, -0.19568444788455963, 0.6160652711987495],
        'scale': 5.1607864066598905,
        'rotation': [-0.0032918453216552734, -0.12640732526779175, -1.6554943323135376],
        'material_ior': 1.7000342640124446,
        'material_roughness': 0.13993626928782799
    },
    'alpha6': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 0, 0), (1, 1, 0), (0, 0, -1), (0, -1, -1),
                           (0, 1, -1), (0, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1),
                           (1, 1, -1), (-1, 0, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'distances': [0.19666633009910583, 0.6398002505302429, 0.4684765636920929, 0.8366933465003967,
                      0.741870641708374, 0.7422857880592346, 0.320103257894516, 0.5582546591758728, 0.5005581378936768,
                      0.6028826832771301, 0.5543457865715027, 0.6761952638626099, 0.5320407748222351,
                      0.8264796733856201, 0.7732805609703064, 0.7696529030799866, 0.7346971035003662,
                      0.7784842848777771, 0.8692794442176819, 0.8123345971107483],
        'scale': 4,
        'rotation': [-0.8004319465660955757, 0.9018085570773109794, 1.2971580028533936],
        "origin": [0.1671956479549408, 0.11368720978498459, 0.4015026092529297],
        'material_ior': 1.63,
        'material_roughness': 0.16
    },
    'alpha7': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 0, 0), (1, 1, 0), (0, 0, -1), (0, -1, -1),
                           (0, 1, -1), (0, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1),
                           (1, 1, -1), (-1, 0, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'distances': [0.4180285632610321, 0.912309467792511, 0.5652278065681458, 0.7703648805618286, 0.5062384605407715,
                      0.779175341129303, 0.5240940451622009, 0.8707500100135803, 1.0, 0.9103384613990784,
                      0.49669593572616577, 0.6077476739883423, 0.5837719440460205, 0.7623077630996704,
                      0.7793679237365723, 0.8473895788192749, 0.6140317916870117, 0.7662931680679321, 0.822736382484436,
                      0.6973626613616943],
        'scale': 4.84,
        'rotation': [0.8882678747177124, 0.4394600987434387, 0.09270056337118149],
        "origin": [1.147449254989624, 0.5360877513885498, 1.2139800786972046],
        'material_ior': 1.63,
        'material_roughness': 0.24
    },
    'alpha8': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 0, 0), (1, 1, 0), (0, 0, -1), (0, -1, -1),
                           (0, 1, -1), (0, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1),
                           (1, 1, -1), (-1, 0, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'distances': [
            0.25498175621032715, 0.5408605933189392, 0.7145980000495911, 0.8008067607879639, 0.46802985668182373,
            0.9101651906967163, 0.2759777903556824, 0.6600923538208008, 0.6596804261207581, 0.7460605502128601,
            0.8013473153114319, 0.8005070686340332, 0.8633545637130737, 0.7140575051307678, 0.863895058631897,
            0.8010476231575012, 0.46724194288253784, 1.0, 0.9095159769058228, 0.9993507862091064
        ],
        'scale': 4,
        'rotation': [3.4380996227264404,
                     0.4680838882923126,
                     4.040154933929443],
        "origin": [
            -0.29330942034721375,
            -0.23531806468963623,
            1.2931525707244873
        ],
        'material_ior': 1.83,
        'material_roughness': 0.1
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


def cube_test():
    """
    A cube with distances of 1 between origin and each face should entirely fill the image.
    """
    cube = Crystal(**TEST_CRYSTALS['cube'])
    cube.to(device)
    v, f = cube.build_mesh()
    m = Trimesh(vertices=to_numpy(v), faces=to_numpy(f))
    m.show()
    projector = Projector(cube, zoom=0.5)
    projector.image[:, projector.image.sum(dim=0) == 0] = 1
    plt.imshow(tensor_to_image(projector.image))
    plt.show()


def show_projected_image(which='alpha'):
    """
    Show the projected crystal wireframe.
    """
    # image_size = (256, 256)
    # image_size = (1000, 1000)
    image_size = (300, 300)
    assert which in TEST_CRYSTALS
    crystal = Crystal(**TEST_CRYSTALS[which])
    if which == 'beta':
        zoom = 0.001
    else:
        zoom = 0.1
    crystal.to(device)
    # v, f = crystal.build_mesh()
    # m = Trimesh(vertices=to_numpy(v), faces=to_numpy(f))
    # m.show()

    projector = Projector(crystal, external_ior=1., image_size=image_size, zoom=zoom, camera_axis=[0, 0, -1],
                          multi_line=True)
    projector.image[:, projector.image.sum(dim=0) == 0] = 1
    plt.imshow(tensor_to_image(projector.image))
    plt.show()


def show_vertices(which='alpha'):
    """
    Show the projected crystal wireframe along with locations of all the vertices and intersections.
    """
    image_size = (500, 500)
    assert which in TEST_CRYSTALS
    crystal = Crystal(**TEST_CRYSTALS[which])
    if which == 'beta':
        zoom = 0.001
    else:
        zoom = 0.2
    crystal.to(device)

    # Original method
    projector = Projector(crystal, external_ior=1., image_size=image_size, zoom=zoom, camera_axis=[0, 0, -1],
                          multi_line=False)
    projector.image[:, projector.image.sum(dim=0) == 0] = 1
    img_og = tensor_to_image(projector.image)

    # Multi-line method
    projector = Projector(crystal, external_ior=1., image_size=image_size, zoom=zoom, camera_axis=[0, 0, -1],
                          multi_line=True)
    projector.image[:, projector.image.sum(dim=0) == 0] = 1
    img_ml = tensor_to_image(projector.image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_og)
    axes[0].set_title('Original method')
    axes[1].imshow(img_ml)
    axes[1].set_title('Multi-line method')
    axes[1].scatter(*to_numpy(projector.keypoints).T, color='green', marker='o', s=100, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_vertex_heatmap(which='alpha'):
    """
    Generate a heatmap of all the vertices and intersections.
    """
    image_size = (500, 500)
    assert which in TEST_CRYSTALS
    crystal = Crystal(**TEST_CRYSTALS[which])
    zoom = 0.1
    crystal.to(device)
    projector = Projector(crystal, external_ior=1., image_size=image_size, zoom=zoom, camera_axis=[0, 0, -1],
                          multi_line=True)
    projector.image[:, projector.image.sum(dim=0) == 0] = 1

    heatmap = generate_keypoints_heatmap(
        keypoints=projector.keypoints,
        image_size=image_size[0],
        blob_height=1.0,
        blob_variance=20.0
    )

    heatmap = 1 - to_numpy(heatmap)
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, cmap='hot')
    plt.scatter(*to_numpy(projector.keypoints).T, color='green', marker='x', s=400, alpha=0.9)
    plt.tight_layout()
    plt.show()


def check_bounds():
    res = 400
    crystal = Crystal(**TEST_CRYSTALS['cube'])
    crystal.origin.data[2] -= crystal.vertices[:, 2].min()
    crystal.build_mesh()
    crystal.to(device)
    # print(crystal.vertices)

    # Create and render a scene
    scene = Scene(
        crystal=crystal,
        res=res,

        camera_distance=32.,
        focus_distance=30.,
        # focal_length=29.27,
        camera_fov=10.2,
        aperture_radius=0.3,

        light_z_position=-5.1,
        light_scale=5.,

        cell_z_positions=[-5, 0., 5., 10.],
        cell_surface_scale=3,
    )
    img = scene.render()

    # Check the scaling
    for i in range(4):
        z = scene.cell_z_positions[i]
        _, (min_y, max_y) = scene.get_xy_bounds(z)
        zoom = 2 / (max_y - min_y)
        logger.info(f'Estimated zoom factor for z={z}: {zoom:.3f}')
        pts = torch.tensor([[-1 / zoom, -1 / zoom, z], [1 / zoom, 1 / zoom, z]], device=device)
        uv_pts = project_to_image(scene.mi_scene, pts)
        assert torch.allclose(uv_pts[0], init_tensor([0, res], device=device), atol=1e-3)
        assert torch.allclose(uv_pts[1], init_tensor([res, 0], device=device), atol=1e-3)

    # Save the original image with projected overlay
    projector = Projector(
        crystal=crystal,
        external_ior=1.333,
        image_size=(res, res),
        zoom=zoom,
        background_image=img,
        camera_axis=[0, 0, -1]
    )
    img_overlay = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_overlay)
    # ax.imshow(img)
    # uv_crystal = to_numpy(scene.get_crystal_image_coords())
    # ax.scatter(uv_crystal[:, 0], uv_crystal[:, 1], marker='x', c='r', s=50)
    # uv_pts = to_numpy(uv_pts)
    # ax.scatter(uv_pts[:, 0], uv_pts[:, 1], marker='o', c='g', s=100)
    # uv_pts2 = to_numpy(uv_pts2)
    # ax.scatter(uv_pts2[:, 0], uv_pts2[:, 1], marker='o', c='purple', s=100)
    fig.tight_layout()
    plt.show()


def match_to_scene():
    res = 400
    crystal = Crystal(**TEST_CRYSTALS['alpha6'])
    crystal.scale.data = init_tensor(1.2, device=crystal.scale.device)
    crystal.origin.data[:2] = torch.tensor([0, 0], device=crystal.origin.device)
    crystal.origin.data[2] -= crystal.vertices[:, 2].min()
    v, f = crystal.build_mesh()
    crystal.to(device)
    # m = Trimesh(vertices=to_numpy(v), faces=to_numpy(f))
    # m.show()

    # Create and render a scene
    scene = Scene(
        crystal=crystal,
        res=res,
        spp=512,
        integrator_max_depth=3,

        camera_distance=32.,
        focus_distance=30.,
        # focal_length=29.27,
        camera_fov=10.2,
        aperture_radius=0.3,

        light_z_position=-5.1,
        # light_scale=5.,
        light_scale=10000.,
        light_radiance=.3,
        cell_z_positions=[-5, 0., 5., 10.],
        cell_surface_scale=3,
    )
    img = scene.render()

    # Get the unit scale factor
    z = crystal.vertices[:, 2].mean().item()
    _, (min_y, max_y) = scene.get_xy_bounds(z)
    zoom = 2 / (max_y - min_y)
    logger.info(f'Estimated zoom factor: {zoom:.3f}')
    pts2 = torch.tensor([[0, 1 / zoom, z], [0, -1 / zoom, z]], device=device)
    uv_pts2 = project_to_image(scene.mi_scene, pts2)  # these should appear at the top and bottom of the image

    # Save the original image with projected overlay
    projector = Projector(
        crystal=crystal,
        external_ior=1.333,
        image_size=(res, res),
        zoom=zoom,
        transparent_background=True,
        multi_line=True
    )
    img_overlay = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    img_overlay[:, :, 3] = (img_overlay[:, :, 3] * 0.5).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.imshow(img_overlay)
    # uv_crystal = to_numpy(scene.get_crystal_image_coords())
    # ax.scatter(uv_crystal[:, 0], uv_crystal[:, 1], marker='x', c='r', s=50)
    # uv_pts2 = to_numpy(uv_pts2)
    # ax.scatter(uv_pts2[:, 0], uv_pts2[:, 1], marker='o', c='purple', s=100)

    # uv_vertices = to_numpy(projector.vertices_2d)
    # ax.scatter(uv_vertices[:, 0], uv_vertices[:, 1], marker='o', c='g', s=70)

    fig.tight_layout()
    plt.show()


def make_rotation_video():
    """
    Create a video of a rotating crystal.
    """
    crystal = Crystal(**TEST_CRYSTALS['alpha'])
    crystal.to(device)
    w, h = 256, 256
    projector = Projector(crystal, image_size=(w, h), zoom=0.1)
    n_frames = 36
    duration = 5

    LOGS_PATH.mkdir(exist_ok=True)
    video_path = LOGS_PATH / f'{START_TIMESTAMP}_rotation_video.mp4'

    # Initialise ffmpeg process
    input_args = {
        'format': 'rawvideo',
        'pix_fmt': 'rgb24',
        's': f'{w}x{h}',
        'r': n_frames / duration,
    }
    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': 24,
    }
    process = (
        ffmpeg
        .input('pipe:', **input_args)
        .output(str(video_path), **output_args)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    logger.info('Rendering frames.')
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, n_frames)):
        if i > 0 and (i + 1) % 10 == 0:
            logger.info(f'Rendering frame {i + 1}/{n_frames}.')

        # Rotate the crystal, update the frame and write to stream
        crystal.rotation.data = init_tensor([angle, angle, 0.], device=device)
        crystal.build_mesh()
        image = projector.project()
        image = (tensor_to_image(image) * 255).astype(np.uint8)
        process.stdin.write(image.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()


def make_ior_video():
    """
    Create a video of a crystal with changing ior.
    """
    crystal = Crystal(**TEST_CRYSTALS['alpha4'])
    crystal.to(device)
    w, h = 512, 512
    projector = Projector(crystal, image_size=(w, h), zoom=0.05, external_ior=1.)
    n_frames = 36
    duration = 5

    LOGS_PATH.mkdir(exist_ok=True)
    video_path = LOGS_PATH / f'{START_TIMESTAMP}_ior_video.mp4'

    # Initialise ffmpeg process
    input_args = {
        'format': 'rawvideo',
        'pix_fmt': 'rgb24',
        's': f'{w}x{h}',
        'r': n_frames / duration,
    }
    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': 24,
    }
    process = (
        ffmpeg
        .input('pipe:', **input_args)
        .output(str(video_path), **output_args)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    logger.info('Rendering frames.')
    for i, ior in enumerate(np.linspace(1, 2, n_frames)):
        if i > 0 and (i + 1) % 10 == 0:
            logger.info(f'Rendering frame {i + 1}/{n_frames}.')

        # Rotate the crystal, update the frame and write to stream
        crystal.material_ior.data = init_tensor(ior, device=device)
        image = projector.project()
        image = (tensor_to_image(image) * 255).astype(np.uint8)
        process.stdin.write(image.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()


if __name__ == '__main__':
    # cube_test()
    # show_projected_image('alpha')
    # show_projected_image('beta')
    # show_projected_image('alpha2')
    # show_projected_image('alpha3')
    # show_projected_image('alpha4')
    # show_projected_image('alpha5')
    # show_projected_image('alpha6')
    show_vertices('alpha8')
    # show_vertex_heatmap('alpha6')
    # match_to_scene()
    # make_rotation_video()
    # make_ior_video()
