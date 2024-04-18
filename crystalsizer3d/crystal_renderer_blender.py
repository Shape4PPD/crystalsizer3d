import json
import math
import os
import shutil
from multiprocessing import Lock, Manager, Pool
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from trimesh import Scene, Trimesh
from trimesh.exchange.obj import export_obj

from crystalsizer3d import BLENDER_PATH, PCS_PATH, logger
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs
from crystalsizer3d.args.renderer_args import RendererArgs
from crystalsizer3d.crystal_renderer import CrystalRenderer, RenderError, append_json, append_json_mp
from crystalsizer3d.util.utils import SEED


def blender_render(
        settings: 'CrystalWellSettings',
        attempts: int = 1,
        seed: Optional[int] = None,
        quiet: bool = False
):
    """
    Run blender to generate the renderings.
    """

    # Run containerised blender
    if str(BLENDER_PATH)[-3:] == 'sif':
        blender_cmd = 'apptainer exec '
        if settings.settings_dict['device'] == 'GPU':
            blender_cmd += '--nv '
        blender_cmd += (f'--bind {str(settings.path.parent.absolute())}:/pcs_output '
                        f'{BLENDER_PATH} blender --background '
                        f'--python /pcs/pcs/blender_addon/modules/blvcw/crystal_well_headless.py '
                        f'--settings_file /pcs_output/vcw_settings.json')

    # Run a locally-installed blender
    else:
        blender_cmd = f'{BLENDER_PATH} --background --python {PCS_PATH} --settings_file {str(settings.path.absolute())}'

    # Set random seed
    if seed is not None:
        blender_cmd += f' --seed {seed}'

    # Suppress output
    # logger.info('Rendering crystals, offloading to blender...')
    if quiet:
        blender_cmd += ' > /dev/null 2>&1'  # hide errors
        # blender_cmd += ' > /dev/null'  # show errors
    else:
        logger.info(f'Running command: {blender_cmd}')

    os.system(blender_cmd)

    # Check the output directory
    output_dir = Path(settings.settings_dict['output_path'])
    for i in range(settings.settings_dict['number_images']):
        if not (output_dir / f'{(i + 1):010d}.png').exists() or not (output_dir / f'{(i + 1):010d}.png.json').exists():
            if attempts > 1:
                logger.warning(f'Blender failed to render image {i + 1}/{settings.settings_dict["number_images"]}. '
                               f'Retrying {attempts - 1} more times.')
                for j in range(settings.settings_dict['number_images']):
                    img = output_dir / f'{(j + 1):010d}.png'
                    if img.exists():
                        img.unlink()
                    json_file = output_dir / f'{(j + 1):010d}.png.json'
                    if json_file.exists():
                        json_file.unlink()
                return blender_render(settings, attempts=attempts - 1, seed=seed, quiet=quiet)
            raise RenderError('Blender failed to render all images.', idx=i)


def render_from_parameters(
        mesh: Trimesh,
        settings_path: Path,
        r_params: dict,
        tmp_dir: Path = Path('/tmp/crystal_renders'),
        attempts: int = 3
) -> np.ndarray:
    """
    Render a crystal mesh from a dictionary of parameters.
    """
    # Ensure the temporary directory exists and is empty
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(exist_ok=True)

    # Write the mesh to file
    obj_path = tmp_dir / 'crystal.obj'
    scene = Scene()
    scene.add_geometry([mesh, ])
    with open(obj_path, 'w') as f:
        f.write(export_obj(
            scene,
            header=None,
            include_normals=True,
            include_color=False
        ))

    # Copy over the settings file and update the rendering parameters
    shutil.copy(settings_path, tmp_dir / 'vcw_settings.json')
    settings_path = tmp_dir / 'vcw_settings.json'
    vcw_settings = CrystalWellSettings()
    vcw_settings.from_json(settings_path)
    vcw_settings.path = settings_path
    vcw_settings.settings_dict['number_images'] = 1
    vcw_settings.settings_dict['crystal_import_path'] = str(obj_path.absolute())
    vcw_settings.settings_dict['output_path'] = str(tmp_dir.absolute())
    vcw_settings.settings_dict['crystal_location'] = r_params['location']
    vcw_settings.settings_dict['crystal_scale'] = r_params['scale']
    vcw_settings.settings_dict['crystal_rotation'] = r_params['rotation']
    vcw_settings.settings_dict['crystal_material_min_ior'] = r_params['material']['ior']
    vcw_settings.settings_dict['crystal_material_max_ior'] = r_params['material']['ior']
    vcw_settings.settings_dict['crystal_material_min_brightness'] = r_params['material']['brightness']
    vcw_settings.settings_dict['crystal_material_max_brightness'] = r_params['material']['brightness']
    vcw_settings.settings_dict['crystal_material_min_roughness'] = r_params['material']['roughness']
    vcw_settings.settings_dict['crystal_material_max_roughness'] = r_params['material']['roughness']
    vcw_settings.settings_dict['light_angle_min'] = np.rad2deg(r_params['light']['angle'])
    vcw_settings.settings_dict['light_angle_max'] = np.rad2deg(r_params['light']['angle'])
    vcw_settings.settings_dict['light_location'] = r_params['light']['location']
    vcw_settings.settings_dict['light_rotation'] = r_params['light']['rotation']
    vcw_settings.settings_dict['light_energy'] = r_params['light']['energy']
    vcw_settings.write_json()

    # Render the image
    blender_render(vcw_settings, attempts=attempts, quiet=True)

    # Load the image to a buffer
    img_path = tmp_dir / '0000000001.png'
    img = np.array(Image.open(img_path))

    # Clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return img


class CrystalWellSettings:
    """
    Clone (mostly) of the PCS class as we can't import it from the blender addon.
    VCW component to store every custom setting from the UI.
    Provided to CrystalWellSimulator that reads the values from the settings_dict.
    """

    def __init__(
            self,
            number_threads: int = 16,
            device: str = 'CPU',
            n_frames: int = 1,
            res_x: int = 1024,
            res_y: int = 1024,
            field_of_view: float = 1.5708 / 2,
            camera_distance: float = 15.0,
            cw_depth: float = -15.0,
            output_path: Path = Path('.'),
            number_crystals: int = 10,
            number_crystals_std_dev: int = 0,
            distributor: str = 'SIMPLE',
            center_crystals: bool = False,
            optimize_rotation: bool = True,
            crystal_location: Optional[Tuple[float]] = None,
            crystal_scale: Optional[float] = None,
            crystal_rotation: Optional[Tuple[float]] = None,
            total_crystal_area_min: float = 0.05,
            total_crystal_area_max: float = 0.5,
            crystal_area_min: int = 3**2,
            crystal_area_max: int = 128**2,
            crystal_edge_min: int = 3,
            crystal_edge_max: int = 384,
            crystal_aspect_ratio_max: float = 8 / 0.7,
            smooth_shading_distributor: bool = True,
            scaling_crystals_average: Tuple[float, float, float] = (1.0, 1.0, 1.0),
            scaling_crystals_std_dev: float = 0.0,
            rotation_crystals_average: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            rotation_crystals_std_dev: float = 0.0,
            crystal_object: str = '',
            crystal_import_path: Path = Path('.'),
            number_variants: int = 1,
            crystal_material_name: str = 'GLASS',
            custom_material_name: str = 'default',
            crystal_material_min_ior: float = 1.1,
            crystal_material_max_ior: float = 1.6,
            crystal_material_min_brightness: float = 0.75,
            crystal_material_max_brightness: float = 0.9,
            crystal_material_min_roughness: float = 0.0,
            crystal_material_max_roughness: float = 0.4,
            light_type: str = 'AREA',
            light_angle_min: int = 0,
            light_angle_max: int = 0,
            use_bottom_light: bool = True,
            light_location: Optional[Tuple[float]] = None,
            light_rotation: Optional[Tuple[float]] = None,
            light_energy: Optional[float] = None,
            number_images: int = 1,
            remesh_mode: str = 'VOXEL',
            remesh_octree_depth: int = 4,
            transmission_mode: bool = False,
            generate_blender: bool = False,
    ):
        self.settings_dict = {
            'number_threads': number_threads,
            'device': device,
            'n_frames': n_frames,
            'res_x': res_x,
            'res_y': res_y,
            'field_of_view': field_of_view,
            'camera_distance': camera_distance,
            'cw_depth': cw_depth,
            'number_crystals': number_crystals,
            'number_crystals_std_dev': number_crystals_std_dev,
            'distributor': distributor,
            'center_crystals': center_crystals,
            'optimize_rotation': optimize_rotation,
            'crystal_location': crystal_location,
            'crystal_scale': crystal_scale,
            'crystal_rotation': crystal_rotation,
            'total_crystal_area_min': total_crystal_area_min,
            'total_crystal_area_max': total_crystal_area_max,
            'crystal_area_min': crystal_area_min,
            'crystal_area_max': crystal_area_max,
            'crystal_edge_min': crystal_edge_min,
            'crystal_edge_max': crystal_edge_max,
            'crystal_aspect_ratio_max': crystal_aspect_ratio_max,
            'smooth_shading_distributor': smooth_shading_distributor,
            'scaling_crystals_average': scaling_crystals_average,
            'scaling_crystals_std_dev': scaling_crystals_std_dev,
            'rotation_crystals_average': rotation_crystals_average,
            'rotation_crystals_std_dev': rotation_crystals_std_dev,
            'crystal_material_name': crystal_material_name,
            'custom_material_name': custom_material_name,
            'crystal_material_min_ior': crystal_material_min_ior,
            'crystal_material_max_ior': crystal_material_max_ior,
            'crystal_material_min_brightness': crystal_material_min_brightness,
            'crystal_material_max_brightness': crystal_material_max_brightness,
            'crystal_material_min_roughness': crystal_material_min_roughness,
            'crystal_material_max_roughness': crystal_material_max_roughness,
            'light_type': light_type,
            'light_angle_min': light_angle_min,
            'light_angle_max': light_angle_max,
            'light_location': light_location,
            'light_rotation': light_rotation,
            'light_energy': light_energy,
            'use_bottom_light': use_bottom_light,
            'crystal_object': crystal_object,
            'crystal_import_path': str(crystal_import_path.absolute()),
            'number_variants': number_variants,
            'output_path': str(output_path.absolute()),
            'number_images': number_images,
            'remesh_mode': remesh_mode,
            'remesh_octree_depth': remesh_octree_depth,
            'transmission_mode': transmission_mode,
            'generate_blender': generate_blender
        }

        self.path = output_path.parent / 'vcw_settings.json'

    def print_settings(self):
        logger.info('*** VCW SETTINGS ***')
        logger.info(json.dumps(self.settings_dict, indent=4))

    def write_json(self):
        """
        Used to save the settings_dict as a json file.
        """
        with open(self.path, 'w') as file:
            json.dump(self.settings_dict, file, indent=4)

    def from_json(self, json_file_path):
        """
        Used for headless execution.
        """
        with open(json_file_path, 'r') as file:
            jd = json.load(file)
            self.settings_dict.update(jd)


def _render_batch(
        batch_idx: int,
        img_start_idx: int,
        obj_path: Path,
        n_imgs: int,
        settings_path: Path,
        tmp_dir: Path,
        images_dir: Path,
        lock: Lock,
        quiet_render: bool = False
):
    """
    Render a batch of crystals to images.
    """
    tmp_dir = tmp_dir / f'tmp_{batch_idx:010d}'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    assert images_dir.exists(), f'Images dir does not exist! ({images_dir})'

    # Copy over the settings file and update the parameters
    shutil.copy(settings_path, tmp_dir / 'vcw_settings.json')
    settings_path = tmp_dir / 'vcw_settings.json'
    settings = CrystalWellSettings()
    settings.from_json(settings_path)
    settings.path = settings_path
    settings.settings_dict['number_images'] = n_imgs
    settings.settings_dict['crystal_import_path'] = str(obj_path.absolute())
    settings.settings_dict['output_path'] = str(tmp_dir.absolute())
    settings.write_json()

    # Render
    logger.info(f'Batch {batch_idx}: Rendering {n_imgs} crystals from {obj_path} to {tmp_dir}.')
    try:
        blender_render(settings, attempts=1, quiet=quiet_render, seed=SEED + batch_idx)
    except RenderError as e:
        logger.warning(f'Rendering failed! {e}')
        logger.info(f'Adding idx={str(img_start_idx + e.idx)} details to {str(tmp_dir.parent / "errored.json")}')
        append_json_mp(
            tmp_dir.parent.parent / 'errored.json',
            {img_start_idx + e.idx: {
                'batch_idx': batch_idx,
                'obj_idx': e.idx,
                'obj_file': str(obj_path.name),
                'img_file': f'{img_start_idx + e.idx:010d}.png'
            }}, lock)
        logger.info(f'Added idx={str(img_start_idx + e.idx)} details to {str(tmp_dir.parent / "errored.json")}')
        raise e

    # Rename the images and json files to start at the correct index
    logger.info(f'Batch {batch_idx}: Collating results.')
    image_files = list(tmp_dir.glob('*.png'))
    image_files = sorted(image_files)
    json_files = list(tmp_dir.glob('*.png.json'))
    json_files = sorted(json_files)
    n_files = len(image_files)
    assert len(json_files) == n_files, \
        f'Batch {batch_idx}: Number of images and json files do not match! ({len(image_files)} vs {len(json_files)})'
    for i, (img, j) in enumerate(zip(image_files, json_files)):
        img.rename(images_dir / f'{img_start_idx + i:010d}.png')  # Move images into the images directory
        j.rename(tmp_dir / f'{img_start_idx + i:010d}.png.json2')

    # Combine the json parameter files
    json_files = list(tmp_dir.glob('*.json2'))
    json_files = sorted(json_files)
    assert len(json_files) == n_files
    segmentations = {}
    params = {}
    for j in json_files:
        with open(j) as f:
            data = json.load(f)
            assert j.stem not in segmentations
            assert j.stem not in params
            assert len(data['segmentation']) == 1
            assert len(data['crystals']['locations']) == 1
            assert len(data['crystals']['scales']) == 1
            assert len(data['crystals']['rotations']) == 1
            segmentations[j.stem] = data['segmentation'][0]
            params[j.stem] = {
                'location': data['crystals']['locations'][0],
                'scale': data['crystals']['scales'][0],
                'rotation': data['crystals']['rotations'][0],
                'material': {
                    'brightness': data['materials']['brightnesses'][0],
                    'ior': data['materials']['iors'][0],
                    'roughness': data['materials']['roughnesses'][0],
                },
                'light': data['light']
            }

    # Write the combined segmentations and parameters to json files
    append_json(tmp_dir.parent / f'segmentations_{batch_idx:010d}.json', segmentations)
    append_json(tmp_dir.parent / f'params_{batch_idx:010d}.json', params)

    # Move the blender files
    blend_files = list(tmp_dir.glob('*.blend'))
    if len(blend_files):
        blend_dir = images_dir.parent / 'blender'
        assert blend_dir.exists(), f'Blender files dir does not exist! ({blend_dir})'
        blend_files = sorted(blend_files)
        assert len(blend_files) == n_files
        for i, ble in enumerate(blend_files):
            ble.rename(blend_dir / f'{img_start_idx + i:010d}.blend')

    # Clean up
    shutil.rmtree(tmp_dir)


def _render_batch_wrapper(args):
    return _render_batch(*args)


class CrystalRendererBlender(CrystalRenderer):
    def __init__(
            self,
            obj_dir: Path,
            dataset_args: DatasetSyntheticArgs,
            renderer_args: RendererArgs,
            quiet_render: bool = False
    ):
        super().__init__(dataset_args, renderer_args, quiet_render)
        self.obj_dir = obj_dir
        self.blend_dir = self.obj_dir.parent / 'blender'
        self._init_settings()

    @property
    def images_dir(self) -> Path:
        return self.obj_dir.parent / 'images'

    def _init_settings(self):
        """
        Initialise the PCS settings and save them to file
        """
        self.settings = CrystalWellSettings(
            device=self.renderer_args.device,
            output_path=self.images_dir,
            res_x=self.dataset_args.image_size,
            res_y=self.dataset_args.image_size,
            number_crystals=1,
            number_crystals_std_dev=0,
            camera_distance=self.renderer_args.camera_distance,
            distributor='SIMPLE',
            center_crystals=self.dataset_args.centre_crystals,
            optimize_rotation=self.dataset_args.optimise_rotation,
            total_crystal_area_min=self.dataset_args.min_area,
            total_crystal_area_max=self.dataset_args.max_area,
            crystal_object='CUSTOM_SEQ',
            crystal_material_name=self.renderer_args.crystal_material_name,
            custom_material_name=self.renderer_args.custom_material_name,
            crystal_import_path=self.obj_dir,
            crystal_material_min_ior=self.renderer_args.min_ior,
            crystal_material_max_ior=self.renderer_args.max_ior,
            crystal_material_min_brightness=self.renderer_args.min_brightness,
            crystal_material_max_brightness=self.renderer_args.max_brightness,
            crystal_material_min_roughness=self.renderer_args.min_roughness,
            crystal_material_max_roughness=self.renderer_args.max_roughness,
            light_type=self.renderer_args.light_type,
            light_angle_min=self.renderer_args.light_angle_min,
            light_angle_max=self.renderer_args.light_angle_max,
            use_bottom_light=self.renderer_args.use_bottom_light,
            number_images=self.dataset_args.n_samples,
            remesh_mode=self.renderer_args.remesh_mode,
            remesh_octree_depth=self.renderer_args.remesh_octree_depth,
            transmission_mode=self.renderer_args.transmission_mode,
            generate_blender=self.dataset_args.generate_blender
        )
        self.settings.write_json()

    def render(self, obj_paths: Optional[list] = None):
        """
        Render all crystal objects to images.
        """
        if self.n_workers > 1:
            self._render_parallel(obj_paths)
            return

        if obj_paths is None:
            obj_paths = list(self.obj_dir.glob('*.obj'))
            obj_paths = sorted(obj_paths)
            logger.info(f'Found {len(obj_paths)} obj files.')
        Ns = self.dataset_args.n_samples
        No = self.dataset_args.batch_size
        Nb = math.ceil(Ns / No)

        # Create a temporary output directory
        output_dir = self.images_dir.parent / 'tmp_output'
        output_dir.mkdir(exist_ok=True)
        self.settings.settings_dict['output_path'] = str(output_dir.absolute())

        # Loop over object collections
        for i, obj_path in enumerate(obj_paths):
            logger.info(f'Rendering obj file {i + 1}/{len(obj_paths)}: "{obj_path.name}"')
            if obj_path.name == 'crystals.obj':
                batch_idx = 0
            else:
                batch_idx = int(obj_path.stem[-5:])
            start_idx = batch_idx * No

            # Adjust number of output samples to match the number of crystals in each file
            n_images = Ns - start_idx if batch_idx == Nb - 1 else No
            self.settings.settings_dict['number_images'] = n_images

            # Update the settings file with the paths
            self.settings.settings_dict['crystal_import_path'] = str(obj_path.absolute())
            self.settings.write_json()
            self._render_batch(output_dir, start_idx=start_idx, batch_idx=batch_idx, seed=SEED + i)

        # Remove the batch output directory
        output_dir.rmdir()

    def _render_parallel(self, obj_paths: Optional[list] = None):
        """
        Render all crystal objects to images using parallel processing.
        """
        if obj_paths is None:
            obj_paths = list(self.obj_dir.glob('*.obj'))
            obj_paths = sorted(obj_paths)
            logger.info(f'Found {len(obj_paths)} obj files.')
        Ns = self.dataset_args.n_samples
        No = self.dataset_args.batch_size
        Nb = math.ceil(Ns / No)

        # Create output directories
        tmp_dir = self.images_dir.parent / 'tmp_output'
        tmp_dir.mkdir(exist_ok=True)

        # Render the batches in parallel, each process collates its own results and parameters
        logger.info(f'Rendering crystals in parallel, worker pool size: {self.n_workers}')
        manager = Manager()
        lock = manager.Lock()
        shared_args = (self.settings.path, tmp_dir, self.images_dir, lock, True)
        args = []
        for i, obj_path in enumerate(obj_paths):
            batch_idx = int(obj_path.stem[-5:])
            start_idx = batch_idx * No
            n_images = Ns - start_idx if batch_idx == Nb - 1 else No
            args.append((batch_idx, start_idx, obj_path, n_images, *shared_args))
        with Pool(processes=self.n_workers) as pool:
            pool.map(_render_batch_wrapper, args)

        # Combine the batch json files
        logger.info('Combining batch results.')
        params = {}
        segmentations = {}
        for i in range(Nb):
            if (i + 1) % 50 == 0:
                logger.info(f'Combining batch results {i + 1}/{Nb}.')
            with open(tmp_dir / f'params_{i:010d}.json', 'r') as f:
                params.update(json.load(f))
            with open(tmp_dir / f'segmentations_{i:010d}.json', 'r') as f:
                segmentations.update(json.load(f))
        append_json(self.images_dir.parent / 'rendering_parameters.json', params)
        append_json(self.images_dir.parent / 'segmentations.json', segmentations)

        # Remove the tmp output directory
        shutil.rmtree(tmp_dir)

    def _render_batch(self, output_dir: Path, start_idx: int = 0, batch_idx: int = 0, seed: Optional[int] = None):
        """
        Render a batch of crystals to images.
        """
        assert not any(output_dir.iterdir()), f'Output dir not empty before batch render! ({output_dir})'

        try:
            blender_render(self.settings, attempts=1, quiet=self.quiet_render, seed=seed)
        except RenderError as e:
            append_json(
                output_dir.parent / 'errored.json',
                {start_idx + e.idx: {
                    'batch_idx': batch_idx,
                    'obj_idx': e.idx,
                    'obj_file': str(Path(self.settings.settings_dict['crystal_import_path']).name),
                    'img_file': f'{start_idx + e.idx:010d}.png'
                }})
            raise e

        # Rename the images and json files to start at the correct index
        logger.info('Renaming images and json files.')
        image_files = list(output_dir.glob('*.png'))
        image_files = sorted(image_files)
        json_files = list(output_dir.glob('*.json'))
        json_files = sorted(json_files)
        n_files = len(image_files)
        assert len(json_files) == n_files
        for i, (img, j) in enumerate(zip(image_files, json_files)):
            img.rename(self.images_dir / f'{start_idx + i:010d}.png')  # Move images into the images directory
            j.rename(output_dir / f'{start_idx + i:010d}.png.json2')

        # Rename the blender files
        if self.dataset_args.generate_blender:
            blender_files = list(output_dir.glob('*.blend'))
            blender_files = sorted(blender_files)
            assert len(blender_files) == n_files
            for i, ble in enumerate(blender_files):
                ble.rename(self.blend_dir / f'{start_idx + i:010d}.blend')

        # Combine the json parameter files
        logger.info('Combining json parameter files.')
        json_files = list(output_dir.glob('*.json2'))
        json_files = sorted(json_files)
        assert len(json_files) == n_files
        segmentations = {}
        params = {}
        for j in json_files:
            with open(j) as f:
                data = json.load(f)
                assert j.stem not in segmentations
                assert j.stem not in params
                assert len(data['segmentation']) == 1
                assert len(data['crystals']['locations']) == 1
                assert len(data['crystals']['scales']) == 1
                assert len(data['crystals']['rotations']) == 1
                segmentations[j.stem] = data['segmentation'][0]
                params[j.stem] = {
                    'location': data['crystals']['locations'][0],
                    'scale': data['crystals']['scales'][0],
                    'rotation': data['crystals']['rotations'][0],
                    'material': {
                        'brightness': data['materials']['brightnesses'][0],
                        'ior': data['materials']['iors'][0],
                        'roughness': data['materials']['roughnesses'][0],
                    },
                    'light': data['light']
                }

        # Write the combined segmentations and parameters to json files
        append_json(self.images_dir.parent / 'segmentations.json', segmentations)
        append_json(self.images_dir.parent / 'rendering_parameters.json', params)

        # Remove the individual json files
        for j in json_files:
            j.unlink()
        assert not any(output_dir.iterdir()), f'Output dir not properly emptied after batch render! ({output_dir})'

    def annotate_image(self, image_idx: int = 0):
        """
        Annotate the first image with the segmentations and save to disk
        """
        imgs = list(self.images_dir.glob('*.png'))
        imgs = sorted(imgs)
        img0_path = imgs[image_idx]
        img0 = np.array(Image.open(img0_path))
        with open(self.images_dir.parent / 'segmentations.json') as f:
            segmentations = json.load(f)
        seg = np.array(segmentations[img0_path.name])

        # Plot the image with segmentation overlay
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(np.flipud(img0))
        ax.scatter(seg[:, 0], seg[:, 1], marker='x', c='r', s=50)
        fig.tight_layout()
        plt.savefig(self.images_dir.parent / f'segmentation_example_{img0_path.stem}.png')
        plt.close(fig)
