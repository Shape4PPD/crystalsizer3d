import torch
from torch import nn 
import numpy as np
from pathlib import Path
from crystalsizer3d.crystal import Crystal
from crystalsizer3d import LOGS_PATH, ROOT_PATH, START_TIMESTAMP, USE_CUDA, logger
from crystalsizer3d.util.utils import print_args, to_numpy, init_tensor
from crystalsizer3d.refiner.edge_matcher import EdgeMatcher
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from crystalsizer3d.util.plots import plot_image, plot_3d, plot_coutour_loss
#from plot_mesh import multiplot, overlay_plot, plot_sampled_points_with_intensity
import torch.optim as optim
from crystalsizer3d.scene_components.scene import Scene
import cv2
from crystalsizer3d.scene_components.utils import project_to_image
from crystalsizer3d.projector import Projector
from crystalsizer3d.nn.models.rcf import RCF
# from edge_detection import load_rcf
from scipy.ndimage import distance_transform_edt
from torchvision.transforms.functional import to_tensor
from scipy.ndimage import gaussian_filter
from kornia.utils import tensor_to_image
import json
from torch.utils.tensorboard import SummaryWriter
import io

# import matplotlib.pyplot as plt
from PIL import Image

if USE_CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
from PIL import Image

TEST_CRYSTALS = {
    'cube': {
        'lattice_unit_cell': [1, 1, 1],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'point_group_symbol': '222',
        'scale': 1,
        'origin': [0.5, 0, 0],
        'distances': [1., 1., 1.],
        'rotation': [0.2, 0.3, 0.3],
        'material_ior': 1.2,
        'material_roughness': 1.5#0.01
    },
    'cube_test': {
        'lattice_unit_cell': [1, 1, 1],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        'point_group_symbol': '222',
        'scale': 1,
        'origin': [0.5, 0, 0],
        'distances': [1.3, 1.0, 1.0],
        'rotation': [0.2, 0.3, 0.3],
        'material_ior': 1.2,
        'material_roughness': 1.5#0.01
    },
    'alpha6': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 0, 0), (1, 1, 0), (0, 0, -1), (0, -1, -1),
                           (0, 1, -1), (0, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1),
                           (1, 1, -1), (-1, 0, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],
        'distances': [0.3830717206001282, 0.8166847825050354, 0.8026739358901978, 0.9758344292640686,
                      0.9103631377220154, 1.0181487798690796, 0.3933243453502655, 0.7772741913795471,
                      0.8740742802619934, 0.7110176682472229, 0.6107826828956604, 0.9051218032836914,
                      0.908871591091156, 1.1111396551132202, 0.9634890556335449, 0.9997269511222839,
                      1.1894351243972778, 0.9173557758331299, 1.2018373012542725, 1.1176774501800537],
        'origin': [-0.3571832776069641, -0.19568444788455963, 0.6160652711987495],
        'scale': 5.1607864066598905,
        'rotation': [-0.1091805174946785,-0.001362028531730175,1.4652847051620483],
        'material_ior': 1.7000342640124446,
        'material_roughness': 0.13993626928782799
    },
    'alpha': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 0, -1), (1, 1, 1), (1, 1, -1), (0, 1, 1), (0, 1, -1), (1, 0, 0)],
        'distances': [0.53, 0.50, 1.13, 1.04, 1.22, 1.00, 1.30],
        # 'rotation': [0.3, 0.3, 0.3],
        'rotation': [0.0, 0.0, 0.3],
        'point_group_symbol': '222',
        'scale': 3.0,
    },
    'alpha_test': {
        'lattice_unit_cell': [7.068, 10.277, 8.755],
        'lattice_angles': [np.pi / 2, np.pi / 2, np.pi / 2],
        'miller_indices': [(0, 0, 1), (0, 0, -1), (1, 1, 1), (1, 1, -1), (0, 1, 1), (0, 1, -1), (1, 0, 0)],
        'distances': [0.6, 0.4, 1.0, 1.01, 1.28, 0.8, 1.00],
        # 'rotation': [0.3, 0.3, 0.3],
        'rotation': [0.0, 0.0, 0.0],
        'point_group_symbol': '222',
        'scale': 3.0,
    },
}

def log_plot_to_tensorboard(writer, tag, figure, global_step):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    writer.add_image(tag, image, global_step)
    buf.close()

def generate_synthetic_crystal(
        crystal,
        save_dir,
        res = 400,
    ):
    # first generate mesh with cube
    
    # crystal = Crystal(**TEST_CRYSTALS['cube'])
    # crystal.scale.data= init_tensor(1.2, device=crystal.scale.device)
    # crystal.origin.data[:2] = torch.tensor([0, 0], device=crystal.origin.device)
    # crystal.origin.data[2] -= crystal.vertices[:, 2].min()
    # v, f = crystal.build_mesh()
    # crystal.to(device)


    # Create and render a scene
    scene = Scene(
        crystal=crystal,
        res=res,
        spp=512,

        camera_distance=32.,
        focus_distance=30.,
        # focal_length=29.27,
        camera_fov=10.2,
        aperture_radius=0.3,

        light_z_position=-5.1,
        # light_scale=5.,
        light_scale=20000.,#10000.
        light_radiance=.3,
        integrator_max_depth=3,
        cell_z_positions=[-5, 0., 5., 10.],
        cell_surface_scale=3,
    )
    img = scene.render()
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(save_dir / 'rcf_featuremaps' / f'rendered_crystal.png')
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
        colour_facing_towards= [1.0,1.0,1.0],
        colour_facing_away = [1.0,1.0,1.0]
    )
    img_overlay = to_numpy(projector.image * 255).astype(np.uint8).squeeze().transpose(1, 2, 0)
    img_overlay[:, :, 3] = (img_overlay[:, :, 3] * 0.5).astype(np.uint8)
    # fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    # axs[0].imshow(img)
    img_og = img
    # axs[0].imshow(img_overlay)
    # fig.show()
    # get contour image
    rcf_path = Path(ROOT_PATH / 'tmp' / 'bsds500_pascal_model.pth')
    
    """
    Initialise the Richer Convolutional Features model for edge detection.
    """
    rcf = RCF()
    checkpoint = torch.load(rcf_path, weights_only=True)
    rcf.load_state_dict(checkpoint, strict=False)
    rcf.eval()
    rcf.to(device)


    img = to_tensor(img).to(device)[None, ...]
    feature_maps = rcf(img, apply_sigmoid=False)
    #third one seems best for now
    
    dist_maps_arr = []
    # Save the feature maps
    for i, feature_map in enumerate(feature_maps):
        feature_map = to_numpy(feature_map).squeeze()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        if i == len(feature_maps) - 1:
            name = 'fused'
        else:
            name = f'feature_map_{i + 1}'
        img = Image.fromarray((feature_map * 255).astype(np.uint8))
        img.save(save_dir / 'rcf_featuremaps' / f'{name}.png')

        # Save the distance transform
        # feature_map[feature_map < 0.2] = 0
        # img = img.resize((200, 200))
        img = np.array(img).astype(np.float32)/255
        img = (img - img.min()) / (img.max() - img.min())
        thresh = 0.5
        img[img < thresh] = 0
        img[img >= thresh] = 1
        dist = distance_transform_edt(1-img)  #, metric='taxicab')
        dist = dist.astype(np.float32)
        dist = (dist - dist.min()) / (dist.max() - dist.min())
        dist_maps_arr.append(to_tensor(dist))
        Image.fromarray((dist * 255).astype(np.uint8)).save(save_dir / 'rcf_featuremaps' / f'dists_{name}.png')
    
    dist_maps = torch.stack(dist_maps_arr)
    dist_map = dist_maps[5].unsqueeze(0)
    f_map = torch.abs(feature_maps[2])
    
    f_map_np = to_numpy(f_map).squeeze()
    f_map_np = (f_map_np - f_map_np.min()) / (f_map_np.max() - f_map_np.min())
    del rcf
    torch.cuda.empty_cache() 
    
    return f_map, dist_map, img_og, img_overlay, zoom

def generate_line_crystal(img_overlay,save_dir):
    #convert image overlay
    overlay_pil = Image.fromarray(img_overlay.astype(np.uint8))
    overlay_pil.save(save_dir / 'overlay_image.png')
    
    img_gray = np.dot(img_overlay[..., :3], [0.2989, 0.5870, 0.1140])

    # Step 2: Invert the grayscale image
    img_inverted = 255 - img_gray # dont invert #  

    image_pil = Image.fromarray(img_inverted.astype(np.uint8))
    image_pil.save(save_dir / 'inverted_image.png')

    # Save the distance transform
    # feature_map[feature_map < 0.2] = 0
    # img = img.resize((200, 200))
    img_inverted = np.array(img_inverted).astype(np.float32)/255
    img_inverted = gaussian_filter(img_inverted,sigma=2)
    img_inverted = (img_inverted - img_inverted.min()) / (img_inverted.max() - img_inverted.min())
    image_pil = Image.fromarray((img_inverted*255).astype(np.uint8))
    image_pil.save(save_dir / 'inverted__step_image.png')
    thresh = 0.95
    img_inverted[img_inverted < thresh] = 0
    img_inverted[img_inverted >= thresh] = 1
    dist = distance_transform_edt(1-img_inverted)  #, metric='taxicab')
    dist = dist.astype(np.float32)
    dist = (dist - dist.min()) / (dist.max() - dist.min())
    dist = 1-dist # invert again
    distance_map_tensor = to_tensor(dist).squeeze(0)
    Image.fromarray((dist * 255).astype(np.uint8)).save(save_dir / 'distance_map.png')
    dist_map = dist
    return distance_map_tensor

def run():
    
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}'#_{args.image_path.name}'
    rcf_dir = save_dir / 'rcf_featuremaps'
    crystal_dir = save_dir / 'crystals'
    save_dir.mkdir(parents=True, exist_ok=True)
    rcf_dir.mkdir(parents=True, exist_ok=True)
    crystal_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / 'tensorboard_logs'))

    # crystal_tar = Crystal(**TEST_CRYSTALS['cube'])
    crystal_tar = Crystal(**TEST_CRYSTALS['alpha'])
    crystal_tar.scale.data= init_tensor(1.2, device=crystal_tar.scale.device)
    crystal_tar.origin.data[:2] = torch.tensor([0, 0], device=crystal_tar.origin.device)
    crystal_tar.origin.data[2] -= crystal_tar.vertices[:, 2].min()
    v, f = crystal_tar.build_mesh()
    crystal_tar.to(device)

    # generate a synthetic crystal to compare too
    f_map, dist_map, img, img_overlay, zoom = generate_synthetic_crystal(
        crystal_tar,
        save_dir,
    )
    
    # from crystal lines
    img_tensor = generate_line_crystal(img_overlay,save_dir)
    # Normalize the tensor to the range [0, 1] if needed
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # from synthetic crystal
    f_map_inv = 1 - f_map
    img_tensor = f_map.to(device)
    # dist_inv = 1 - dist_map
    # img_tensor = dist_inv.to(device)
    # # dist_map = dist_map.to(device)
    
    projector_tar = Projector(crystal_tar,
                                external_ior=1.333,
                                zoom =zoom,
                                image_size=f_map.shape[-2:],
                                transparent_background=True)
    # projector_tar.to(device)
    projector_tar.project()

    # crystal_opt = Crystal(**TEST_CRYSTALS['cube_test'])
    # crystal_opt = Crystal(**TEST_CRYSTALS['alpha_test'])
    crystal_opt = crystal_tar.clone()
    crystal_opt.scale.data= init_tensor(1.2, device=crystal_opt.scale.device)
    crystal_opt.origin.data[:2] = torch.tensor([0, 0], device=crystal_opt.origin.device)
    crystal_opt.origin.data[2] -= crystal_opt.vertices[:, 2].min()
    crystal_tar_distances = crystal_tar.distances
    # Define the percentage range (e.g., ±5%)
    percentage = 0.05
    # Generate random values in the range [-percentage, +percentage]
    random_factors = torch.randn_like(crystal_opt.distances,device=crystal_opt.scale.device)  * percentage
    # Add the random amount to each value in the tensor
    modified_distances = crystal_tar_distances * (1 + random_factors)
    crystal_opt.distances.data = init_tensor(modified_distances, device=crystal_opt.scale.device)
    # crystal_opt.distances = modified_distances
    v, f = crystal_opt.build_mesh(distances=crystal_opt.distances)
    crystal_opt.to(device)

    projector_opt = Projector(crystal_opt,
                                external_ior=1.333,
                                zoom = zoom,
                                image_size=f_map.shape[-2:],
                                transparent_background=True)
    # projector_opt.to(device)
    projector_opt.project()
    # points_opt = projector_opt.edge_points
    params = {
            'distances': [crystal_opt.distances],
        }
    
    model = EdgeMatcher()
    model.to(device)

    #inital 
    img_int = img_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    fig, ax = plt.subplots()
    title = 'Inital conditions'
    plot_image(ax, title, img_int)
    img_ten = projector_opt.generate_image()
    plot_image(ax, title, tensor_to_image(img_ten))
    # plot_coutour_loss(ax,title,img_int,points_opt.squeeze().detach().cpu().numpy(),np.zeros(1),False)
    plt.savefig(save_dir / f'{title}.png')
    plt.close()
    optimizer = optim.Adam(params['distances'], lr=1e-2)
    target_dist = crystal_tar.distances
    # prev_dist = crystal_opt.distances
    # Training loop
    # with torch.autograd.detect_anomaly(False):
    for step in range(100):  # Run for 100 iterations
        print(f"Step {step}")
        # step.to(device)
        optimizer.zero_grad()  # Zero the gradients
        
        # Convert polar to Cartesian coordinates
        v, f = crystal_opt.build_mesh()
        
        projector_opt = Projector(crystal_opt,
                            zoom = zoom,
                            image_size=f_map.shape[-2:],
                            external_ior=1.333,
                            transparent_background=True)
        projector_opt.project()
        # points_opt = projector_opt.edge_points
        # normals_opt = projector_opt.edge_normals

        
        dist = crystal_opt.distances
        # a = points_opt.get_all_points_tensor()
        # print(f"points tensor {a}")
        # Forward pass: get the pixel value at the current point (x, y)
        loss, distances = model(projector_opt.edge_segments, img_tensor)  # Call model's forward method with Cartesian coordinates
        # Perform backpropagation (minimize the pixel value)
        
        loss.backward(retain_graph=True)

        # Check if the gradients for r and theta are non-zero
        print(f"Step {step}: {projector_opt.distances}")
        
        # Check if gradients are non-zero before optimizer step
        # if dist.grad.abs() < 1e-6:
        #     print(f"Warning: One of the gradients is very small at step {step}")
        
        # Update the radial parameters
        for group in optimizer.param_groups:
            print(group)
            
        if step % 1 == 0:
            fig, ax = plt.subplots()
            title = f'Step {str(step).zfill(3)}'
            plot_image(ax, title, img_int)
            img_ten = projector_opt.generate_image()
            plot_image(ax, title, tensor_to_image(img_ten))
            plt.savefig(save_dir / f'{title}.png')
            plt.savefig(save_dir / f'{title}.png')
            crystal_opt.to_json(crystal_dir / f"crystal_{str(step).zfill(3)}.json")
            plt.close()
            
        optimizer.step()

        # Log the loss value
        writer.add_scalar('Loss', loss.item(), step)
        # Print the updated polar coordinates and the current loss
        print(f"Step {step}: loss: {loss}")


if __name__ == "__main__":
    print(f"running test")
    run()
