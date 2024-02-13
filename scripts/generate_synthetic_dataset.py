import csv
import json
import math
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
from PIL import Image
from trimesh import Scene
from trimesh.exchange.obj import export_obj

from crystalsizer3d import LOGS_PATH, START_TIMESTAMP, logger
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs
from crystalsizer3d.args.renderer_args import RendererArgs
from crystalsizer3d.crystal_renderer import CrystalRenderer, CrystalWellSettings, blender_render
from crystalsizer3d.util.utils import print_args, set_seed, to_dict

PARAMETER_HEADERS = [
    'crystal_id',
    'idx',
    'image',
    'si',
    'il'
]


def parse_args(printout: bool = True) -> Tuple[DatasetSyntheticArgs, RendererArgs]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Generate a dataset of segmented objects found in videos.')
    DatasetSyntheticArgs.add_args(parser)
    RendererArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holder
    dataset_args = DatasetSyntheticArgs.from_args(args)
    renderer_args = RendererArgs.from_args(args)

    return dataset_args, renderer_args


def validate(
        output_dir: Path,
):
    """
    Render a few examples from the parameters to check that they match.
    """
    with open(output_dir / 'options.yml', 'r') as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)
        dataset_args = DatasetSyntheticArgs.from_args(spec['dataset_args'])

    n_examples = min(dataset_args.validate_n_samples, dataset_args.n_samples)
    if n_examples <= 0:
        logger.info('Skipping validation.')
        return

    val_dir = output_dir / 'validation'
    val_dir.mkdir(exist_ok=True)
    obj_dir = output_dir / 'crystals'
    obj_val_path = val_dir / 'crystal.obj'

    # Initialise synthetic crystal generator
    from crystalsizer3d.crystal_generator import CrystalGenerator
    generator = CrystalGenerator(
        crystal_id=dataset_args.crystal_id,
        ratio_means=dataset_args.ratio_means,
        ratio_stds=dataset_args.ratio_stds,
        zingg_bbox=dataset_args.zingg_bbox,
        constraints=dataset_args.distance_constraints,
    )

    # Copy over the rendering parameters
    settings_path = val_dir / 'vcw_settings.json'
    shutil.copy(output_dir / 'vcw_settings.json', settings_path)
    vcw_settings = CrystalWellSettings()
    vcw_settings.from_json(settings_path)
    vcw_settings.path = settings_path
    vcw_settings.settings_dict['number_images'] = 1
    vcw_settings.settings_dict['crystal_import_path'] = str(obj_val_path.absolute())
    vcw_settings.settings_dict['output_path'] = str(val_dir.absolute())

    # Load rendering parameters
    logger.info('Loading rendering parameters.')
    with open(output_dir / 'rendering_parameters.json', 'r') as f:
        rendering_parameters = json.load(f)

    # Load segmentations
    logger.info('Loading segmentations.')
    with open(output_dir / 'segmentations.json', 'r') as f:
        segmentations = json.load(f)

    # Load data
    logger.info('Loading parameters.')
    with open(output_dir / 'parameters.csv', 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = {}
        for i, row in enumerate(reader):
            if (i + 1) % 100 == 0:
                logger.info(f'Loaded {i + 1}/{dataset_args.n_samples} entries.')
            idx = int(row['idx'])
            assert i == idx, f'Missing row {i}!'
            item = {}
            for header in headers:
                if header == 'idx':
                    continue
                if header == 'crystal_id':
                    item[header] = row[header]
                elif header == 'image':
                    img_path = output_dir / 'images' / row['image']
                    assert img_path.exists(), f'Image path does not exist: {img_path}'
                    item['image'] = img_path
                elif header[0] == 'd':
                    item[header] = float(row[header])
                elif header in ['si', 'il']:
                    item[header] = float(row[header])

            # Include the rendering parameters and segmentations
            item['rendering_parameters'] = rendering_parameters[row['image']]
            item['segmentation'] = segmentations[row['image']]
            data[idx] = item

    # Pick some random indices to render
    idxs = np.random.choice(dataset_args.n_samples, size=n_examples, replace=False)
    idxs = np.sort(idxs)

    # Validate each random example
    failed_idxs = []
    for i, idx in enumerate(idxs):
        logger.info(f'Validating entry idx={idx} ({i + 1}/{n_examples}).')

        try:
            # Load the parameters for this idx
            example = data[idx]
            r_params = example['rendering_parameters']

            # Build the crystal
            logger.info('Re-generating crystal.')
            ref_idxs = [''.join(str(i) for i in k) for k in generator.distances.keys()]
            _, z, m = generator.generate_crystal(
                rel_rates=np.array([example[f'd{i}_{k}'] for i, k in enumerate(ref_idxs)]),
            )

            # Write the mesh to file
            scene = Scene()
            scene.add_geometry([m, ])
            with open(obj_val_path, 'w') as f:
                f.write(export_obj(
                    scene,
                    header=None,
                    include_normals=True,
                    include_color=False
                ))

            # Re-render the crystal using the same rendering parameters
            vcw_settings.settings_dict['crystal_location'] = r_params['location']
            vcw_settings.settings_dict['crystal_scale'] = r_params['scale']
            vcw_settings.settings_dict['crystal_rotation'] = r_params['rotation']
            vcw_settings.settings_dict['crystal_material_min_ior'] = r_params['material']['ior']
            vcw_settings.settings_dict['crystal_material_max_ior'] = r_params['material']['ior']
            vcw_settings.settings_dict['crystal_material_min_brightness'] = r_params['material']['brightness']
            vcw_settings.settings_dict['crystal_material_max_brightness'] = r_params['material']['brightness']
            vcw_settings.settings_dict['crystal_material_min_roughness'] = r_params['material']['roughness']
            vcw_settings.settings_dict['crystal_material_max_roughness'] = r_params['material']['roughness']
            if not vcw_settings.settings_dict['transmission_mode']:
                vcw_settings.settings_dict['light_angle_min'] = np.rad2deg(r_params['light']['angle'])
                vcw_settings.settings_dict['light_angle_max'] = np.rad2deg(r_params['light']['angle'])
                vcw_settings.settings_dict['light_location'] = r_params['light']['location']
                vcw_settings.settings_dict['light_rotation'] = r_params['light']['rotation']
            vcw_settings.settings_dict['light_energy'] = r_params['light']['energy']
            vcw_settings.write_json()
            blender_render(vcw_settings, attempts=3, quiet=True)

            # Save the images side by side for comparison
            img_path = val_dir / '0000000001.png'
            img_path_compare = val_dir / f'compare_{idx:05d}.png'
            img_og = Image.open(example['image'])
            img_new = Image.open(img_path)
            img_compare = Image.new('RGB', (img_og.width * 2, img_og.height))
            img_compare.paste(img_og, (0, 0))
            img_compare.paste(img_new, (img_og.width, 0))
            img_compare.save(img_path_compare)

            # Check the Zingg values match
            assert np.allclose(z[0], example['si']), f'Zingg SI values do not match: {z[0]} != {example["si"]}'
            assert np.allclose(z[1], example['il']), f'Zingg IL values do not match: {z[1]} != {example["il"]}'

            # Load the saved mesh from disk
            obj_path = obj_dir / f'crystals_{idx // dataset_args.n_objs_per_file:05d}.obj'
            vertices = []
            normals = []
            reading = False
            with open(obj_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f'o {generator.crystal_id}_{idx:06d}'):
                        reading = True
                        continue
                    if reading:
                        if line.startswith('v '):
                            vertices.append(np.array([float(v) for v in line.split(' ')[1:]]))
                        elif line.startswith('vn '):
                            normals.append(np.array([float(v) for v in line.split(' ')[1:]]))
                        elif line == '':
                            break

            # Check the vertices and normals match the re-generated mesh
            assert np.allclose(m.vertices, np.array(vertices)), 'Vertices do not match!'
            assert np.allclose(m.vertex_normals, np.array(normals)), 'Vertex normals do not match!'

            # Check that the rendering parameters that come out are the same that went in
            render_params_path = val_dir / '0000000001.png.json'
            with open(render_params_path) as f:
                r_params2 = json.load(f)
                assert np.allclose(r_params['location'], r_params2['crystals']['locations'][0]), \
                    'Location does not match!'
                assert np.allclose(r_params['scale'], r_params2['crystals']['scales'][0]), \
                    'Scale does not match!'
                assert np.allclose(r_params['rotation'], r_params2['crystals']['rotations'][0]), \
                    'Rotation does not match!'
                assert np.allclose(r_params['material']['ior'], r_params2['materials']['iors'][0]), \
                    'IOR does not match!'
                assert np.allclose(r_params['material']['brightness'], r_params2['materials']['brightnesses'][0]), \
                    'Brightness does not match!'
                if not vcw_settings.settings_dict['transmission_mode']:
                    assert np.allclose(r_params['light']['angle'], r_params2['light']['angle']), \
                        'Light angle does not match!'
                    assert np.allclose(r_params['light']['location'], r_params2['light']['location']), \
                        'Light location does not match!'
                    assert np.allclose(r_params['light']['rotation'], r_params2['light']['rotation']), \
                        'Light rotation does not match!'
                assert np.allclose(r_params['light']['energy'], r_params2['light']['energy']), \
                    'Light energy does not match!'
                s1, s2 = np.array(example['segmentation']), np.array(r_params2['segmentation'][0])
                if s1.shape == s2.shape:
                    max_s_err = np.max(np.abs(s1 - s2))
                    mean_s_err = np.mean(np.abs(s1 - s2))
                    err = f' Max err = {max_s_err:.3E}. Mean err = {mean_s_err:.3E}.'
                else:
                    err = f' Shapes do not match ({s1.shape} != {s2.shape}).'
                assert s1.shape == s2.shape and np.allclose(s1, s2, atol=0.1), \
                    f'Segmentations do not match! {err}.'

            # Assert that the images aren't too different
            img_og = np.array(img_og).astype(np.float32)
            img_new = np.array(img_new).astype(np.float32)
            mean_diff = np.mean(np.abs(img_og - img_new))
            max_diff = np.max(np.abs(img_og - img_new))
            assert max_diff < 15 and mean_diff < 0.01, \
                f'Images are too different! (Mean diff={mean_diff:.3E}, Max diff={max_diff:.1f})'

            # Clean up
            obj_val_path.unlink()
            render_params_path.unlink()
            img_path.unlink()
            logger.info('Validation passed.')

        except AssertionError as e:
            logger.warning(f'Validation failed: {e}')
            failed_idxs.append(idx)

    # Clean up
    settings_path.unlink()

    if len(failed_idxs) > 0:
        logger.warning(f'Validation failed for {len(failed_idxs)}/{n_examples} examples!')

        # Write the failed indices to a file
        with open(val_dir / 'failed_indices.txt', 'w') as f:
            for idx in failed_idxs:
                f.write(f'{idx}\n')
    else:
        logger.info('Validation complete. All examples passed!')


def main():
    """
    Generate a dataset of synthetic crystal images.
    """
    dataset_args, renderer_args = parse_args()

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Create a directory to save dataset and logs
    save_dir = LOGS_PATH / START_TIMESTAMP
    images_dir = LOGS_PATH / START_TIMESTAMP / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create a directory to save blender files
    if dataset_args.generate_blender:
        blender_dir = LOGS_PATH / START_TIMESTAMP / 'blender'
        blender_dir.mkdir(exist_ok=True)

    # Save arguments to json file
    with open(save_dir / 'options.yml', 'w') as f:
        spec = {
            'created': START_TIMESTAMP,
            'dataset_args': to_dict(dataset_args),
            'renderer_args': to_dict(renderer_args),
        }
        yaml.dump(spec, f)

    # Generate crystal shapes
    obj_dir = save_dir / 'crystals'
    obj_dir.mkdir(exist_ok=True)
    param_path = save_dir / 'parameters.csv'
    if dataset_args.obj_path is not None:
        # Copy the obj and parameters files to the save directory
        logger.info(f'Using pre-generated crystal shapes from {dataset_args.obj_path}.')
        if dataset_args.obj_path.is_dir():
            for f in dataset_args.obj_path.glob('*.obj'):
                shutil.copy(f, obj_dir)
        else:
            shutil.copy(dataset_args.obj_path, obj_dir)
        shutil.copy(dataset_args.param_path, param_path)
    else:
        # Initialise synthetic crystal generator
        from crystalsizer3d.crystal_generator import CrystalGenerator
        generator = CrystalGenerator(
            crystal_id=dataset_args.crystal_id,
            ratio_means=dataset_args.ratio_means,
            ratio_stds=dataset_args.ratio_stds,
            zingg_bbox=dataset_args.zingg_bbox,
            constraints=dataset_args.distance_constraints,
        )

        # Generate randomised crystals
        logger.info('Generating crystals.')
        crystals = generator.generate_crystals(num=dataset_args.n_samples)

        # Save all meshes to obj files, with limited number of objects per file
        logger.info(f'Saving meshes to {obj_dir}.')
        n_files = int(np.ceil(len(crystals) / dataset_args.n_objs_per_file))
        for i in range(n_files):
            start_idx = i * dataset_args.n_objs_per_file
            end_idx = (i + 1) * dataset_args.n_objs_per_file
            meshes = [c[2] for c in crystals[start_idx:end_idx]]
            scene = Scene()
            scene.add_geometry(meshes)
            obj_path = obj_dir / f'crystals_{i:05d}.obj'
            with open(obj_path, 'w') as f:
                f.write(export_obj(
                    scene,
                    header=None,
                    include_normals=True,
                    include_color=False
                ))

        # Save parameters to a csv file
        logger.info(f'Saving crystal parameters to {param_path}.')
        with open(param_path, 'w') as f:
            headers = PARAMETER_HEADERS.copy()
            ref_idxs = [''.join(str(i) for i in k) for k in generator.distances.keys()]
            for i, hkl in enumerate(ref_idxs):
                headers.append(f'd{i}_{hkl}')
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for i, (rel_rates, zingg_vals, _) in enumerate(crystals):
                entry = {
                    'crystal_id': generator.crystal_id,
                    'idx': i,
                    'image': f'{i:010d}.png',
                    'si': zingg_vals[0],
                    'il': zingg_vals[1],
                }
                for j, hkl in enumerate(ref_idxs):
                    entry[f'd{j}_{hkl}'] = rel_rates[j]
                writer.writerow(entry)

    # Render the crystals
    logger.info('Rendering crystals.')
    renderer = CrystalRenderer(
        obj_dir=obj_dir,
        dataset_args=dataset_args,
        renderer_args=renderer_args,
        quiet_render=False
    )
    renderer.render()

    # Annotate a single image for reference
    renderer.annotate_image()

    # Re-generate a few images from the parameters to check that they match
    validate(output_dir=save_dir)

    # Show how long this took, formatted nicely
    elapsed_time = time.time() - start_time
    logger.info(f'Finished in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s.')


def resume(
        save_dir: Path,
        revalidate: bool = True
):
    """
    Re-render any missing images in a dataset.
    Useful for fixing a broken run.
    """
    # Load arguments
    assert (save_dir / 'options.yml').exists(), f'Options file does not exist: {save_dir / "options.yml"}'
    with open(save_dir / 'options.yml', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        dataset_args = DatasetSyntheticArgs.from_args(args['dataset_args'])
        renderer_args = RendererArgs.from_args(args['renderer_args'])

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Check images dir exists
    images_dir = save_dir / 'images'
    assert images_dir.exists(), f'Images directory does not exist: {images_dir}'

    # Instantiate the renderer
    obj_dir = save_dir / 'crystals'
    renderer = CrystalRenderer(
        obj_dir=obj_dir,
        dataset_args=dataset_args,
        renderer_args=renderer_args,
        quiet_render=False
    )

    # If it was originally made in parallel, make sure the fix is in parallel too
    tmp_dir = save_dir / 'tmp_output'
    if renderer.n_workers > 1:
        if not tmp_dir.exists():
            logger.info('Temporary output directory does not exist. Can\'t fix/resume this.')
            return
        assert not (save_dir / 'segmentations.json').exists(), \
            'Segmentations file already exists! Try fixing in serial mode.'
        assert not (save_dir / 'rendering_parameters.json').exists(), \
            'Rendering parameters file already exists! Try fixing in serial mode.'

    # Check each obj file to see if we have all images and parameters present for it
    logger.info('Checking for missing images and parameters.')
    assert obj_dir.exists(), f'Crystals directory does not exist: {obj_dir}'
    Ns = dataset_args.n_samples
    No = dataset_args.n_objs_per_file
    Nb = math.ceil(Ns / No)
    bad_obj_files = []
    for i in range(Nb):
        if (i + 1) % 10 == 0:
            logger.info(f'Checking batch {i + 1}/{Nb}.')
        obj_path = obj_dir / f'crystals_{i:05d}.obj'
        assert obj_path.exists(), f'Object file does not exist: {obj_path}'
        n_objs_in_file = Ns - (i * No) if i == Nb - 1 else No
        img_idxs = np.arange(i * No, i * No + n_objs_in_file)
        for j in img_idxs:
            img_path = images_dir / f'{j:010d}.png'
            if not img_path.exists():
                bad_obj_files.append(obj_path)
                break

        # Check that the batch parameters and segmentations exist for all the good batches
        if obj_path in bad_obj_files:
            continue
        batch_params = tmp_dir / f'params_{i:010d}.json'
        batch_segmentations = tmp_dir / f'segmentations_{i:010d}.json'
        if not batch_params.exists() or not batch_segmentations.exists():
            bad_obj_files.append(obj_path)

    # If all images are present, nothing to do
    if len(bad_obj_files) == 0:
        logger.info('No missing images found.')
        elapsed_time = time.time() - start_time
        logger.info(f'Finished in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s.')
        return

    # Re-render bad batches
    logger.info(f'Re-rendering {len(bad_obj_files)} batches.')
    renderer.render(
        obj_paths=bad_obj_files,
    )

    # Annotate a single image for reference
    renderer.annotate_image()

    # Re-generate a few images from the parameters to check that they match
    if revalidate:
        validate(output_dir=save_dir)

    # Show how long this took, formatted nicely
    elapsed_time = time.time() - start_time
    logger.info(f'Finished in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s.')


if __name__ == '__main__':
    set_seed(1)
    main()

    # -- Use to re-validate a directory --
    # validate(
    #     output_dir=LOGS_PATH / '20231130_1608',
    # )

    # -- Use to fix a broken run --
    # resume(
    #     save_dir=LOGS_PATH / '20231109_1109',
    # )
