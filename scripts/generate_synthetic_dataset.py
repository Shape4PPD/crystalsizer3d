import csv
import json
import shutil
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

from crystalsizer3d import LOGS_PATH, N_WORKERS, START_TIMESTAMP, logger
from crystalsizer3d.args.dataset_synthetic_args import DatasetSyntheticArgs
from crystalsizer3d.crystal_generator import CrystalGenerator
from crystalsizer3d.crystal_renderer import CrystalRenderer
from crystalsizer3d.nn.dataset import PARAMETER_HEADERS
from crystalsizer3d.util.utils import print_args, set_seed, str2bool, to_dict, to_numpy


def parse_args(printout: bool = True) -> Tuple[DatasetSyntheticArgs, Namespace]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Generate a dataset of synthetic crystals.')
    DatasetSyntheticArgs.add_args(parser)
    parser.add_argument('--ds-name', type=str,
                        help='Set a dataset name to use for setting multiple workers on the same dataset.')
    parser.add_argument('--overwrite-existing', type=str2bool, default=False,
                        help='Overwrite an existing dataset if it exists, otherwise try resume then validate.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Set the random seed for the dataset generation.')
    parser.add_argument('--generate-only', type=str2bool, default=False,
                        help='Only generate the crystal parameters and images, do not render them.')
    parser.add_argument('--n-generator-workers', type=int, default=N_WORKERS,
                        help='Set the number of workers to use for generating crystals.')
    parser.add_argument('--n-renderer-workers', type=int, default=1,
                        help='Set the number of workers to use for rendering images.')
    parser.add_argument('--migrate-distances', type=str2bool, default=True,
                        help='Migrate the distances to the new format if resuming from the old format.'
                             'That is, positive minimum distances only and save face areas.')

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    dataset_args = DatasetSyntheticArgs.from_args(args)
    runtime_args = Namespace(
        ds_name=args.ds_name if args.ds_name is not None else START_TIMESTAMP,
        overwrite_existing=args.overwrite_existing,
        seed=args.seed,
        generate_only=args.generate_only,
        n_generator_workers=args.n_generator_workers,
        n_renderer_workers=args.n_renderer_workers,
        migrate_distances=args.migrate_distances
    )

    return dataset_args, runtime_args


def validate(
        output_dir: Path,
        runtime_args: Namespace
):
    """
    Render a few examples from the parameters to check that they match.
    """
    with open(output_dir / 'options.yml', 'r') as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)
        dsa = DatasetSyntheticArgs.from_args(spec['dataset_args'])

    n_examples = min(dsa.validate_n_samples, dsa.n_samples)
    if n_examples <= 0:
        logger.info('Skipping validation.')
        return

    val_dir = output_dir / 'validation'
    val_dir.mkdir(exist_ok=True)

    # Initialise synthetic crystal generator
    from crystalsizer3d.crystal_generator import CrystalGenerator
    generator = CrystalGenerator(
        crystal_id=dsa.crystal_id,
        miller_indices=dsa.miller_indices,
        ratio_means=dsa.ratio_means,
        ratio_stds=dsa.ratio_stds,
        zingg_bbox=dsa.zingg_bbox,
        constraints=dsa.distance_constraints,
        asymmetry=dsa.asymmetry,
        n_workers=runtime_args.n_generator_workers
    )

    # Initialise the crystal renderer
    renderer = CrystalRenderer(
        param_path=output_dir / 'parameters.csv',
        dataset_args=dsa,
        quiet_render=True,
        migrate_distances=False
    )

    # Load data
    logger.info('Loading parameters.')
    with open(renderer.param_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = {}
        for i, row in enumerate(reader):
            if (i + 1) % 100 == 0:
                logger.info(f'Loaded {i + 1}/{dsa.n_samples} entries.')
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

            # Include the rendering parameters, segmentations and vertices
            item['rendering_parameters'] = renderer.rendering_params[idx]
            item['segmentation'] = renderer.segmentations[idx]
            item['vertices'] = renderer.vertices[idx]

            # Add textures
            if dsa.crystal_bumpmap_dim > 0:
                tex_path = output_dir / 'crystal_bumpmaps' / f'{row["image"][:-4]}.npz'
                assert tex_path.exists(), f'Crystal bumpmap path does not exist: {tex_path}'
                item['rendering_parameters']['crystal']['bumpmap'] = tex_path

            # Add the clean image if it exists
            if dsa.generate_clean:
                clean_img_path = output_dir / 'images_clean' / row['image']
                assert clean_img_path.exists(), f'Clean image path does not exist: {clean_img_path}'
                item['clean_image'] = clean_img_path

            data[idx] = item

    # Pick some random indices to render
    idxs = np.random.choice(dsa.n_samples, size=n_examples, replace=False)
    idxs = np.sort(idxs)

    # Validate each random example
    failed_idxs = []
    for i, idx in enumerate(idxs):
        logger.info(f'Validating entry idx={idx} ({i + 1}/{n_examples}).')
        img_path = val_dir / '0000000001.png'
        img_path_clean = val_dir / '0000000001_clean.png'

        try:
            # Load the parameters for this idx
            example = data[idx]
            r_params = example['rendering_parameters']

            # Build the crystal
            logger.info('Re-generating crystal.')
            ref_idxs = [''.join(str(i) for i in k) for k in renderer.miller_indices]
            distances = np.array([example[f'd{i}_{k}'] for i, k in enumerate(ref_idxs)])
            _, _, z, m = generator.generate_crystal(distances=distances)

            # Render the noisy crystal image
            img, scene = renderer.render_from_parameters(r_params, return_scene=True)
            cv2.imwrite(str(img_path), img)

            # Save the images side by side for comparison
            img_path_compare = val_dir / f'compare_{idx:05d}.png'
            img_og = Image.open(example['image'])
            img_new = Image.open(img_path)
            img_compare = Image.new('RGB', (img_og.width * 2, img_og.height))
            img_compare.paste(img_og, (0, 0))
            img_compare.paste(img_new, (img_og.width, 0))
            img_compare.save(img_path_compare)

            if 'clean_image' in example:
                # Render the clean crystal image
                scene.clear_interference()
                img_clean = scene.render(seed=r_params['seed'])
                cv2.imwrite(str(img_path_clean), img_clean)

                # Save the images side by side for comparison
                img_path_compare_clean = val_dir / f'compare_{idx:05d}_clean.png'
                img_clean_og = Image.open(example['clean_image'])
                img_clean_new = Image.open(img_path_clean)
                img_clean_compare = Image.new('RGB', (img_clean_og.width * 2, img_clean_og.height))
                img_clean_compare.paste(img_clean_og, (0, 0))
                img_clean_compare.paste(img_clean_new, (img_clean_og.width, 0))
                img_clean_compare.save(img_path_compare_clean)

            # Assert that the images aren't too different
            img_og = np.array(img_og).astype(np.float32)
            img_new = np.array(img_new).astype(np.float32)
            mean_diff = np.mean(np.abs(img_og - img_new))
            max_diff = np.max(np.abs(img_og - img_new))
            assert max_diff < 15 and mean_diff < 0.01, \
                f'Images are too different! (Mean diff={mean_diff:.3E}, Max diff={max_diff:.1f})'

            # Assert that the clean images aren't too different
            img_clean_og = np.array(img_clean_og).astype(np.float32)
            img_clean_new = np.array(img_clean_new).astype(np.float32)
            mean_diff_clean = np.mean(np.abs(img_clean_og - img_clean_new))
            max_diff_clean = np.max(np.abs(img_clean_og - img_clean_new))
            assert max_diff_clean < 15 and mean_diff_clean < 0.01, \
                f'Clean images are too different! (Mean diff={mean_diff_clean:.3E}, Max diff={max_diff_clean:.1f})'

            # Check that the vertices match
            v1 = to_numpy(scene.crystal.vertices)
            v2 = np.array(example['vertices'])
            assert np.allclose(v1, v2, atol=1e-6), f'Vertices do not match!'

            # Check the Zingg values match
            assert np.allclose(z[0], example['si'], atol=0.05), \
                f'Zingg SI values do not match: {z[0]} != {example["si"]}'
            assert np.allclose(z[1], example['il'], atol=0.05), \
                f'Zingg IL values do not match: {z[1]} != {example["il"]}'

            logger.info('Validation passed.')

        except AssertionError as e:
            logger.warning(f'Validation failed: {e}')
            failed_idxs.append(idx)

        if img_path.exists():
            img_path.unlink()

    # Report any failed indices
    if len(failed_idxs) > 0:
        logger.warning(f'Validation failed for {len(failed_idxs)}/{n_examples} examples!')

        # Write the failed indices to a file
        with open(val_dir / 'failed_indices.txt', 'w') as f:
            for idx in failed_idxs:
                f.write(f'{idx}\n')
    else:
        logger.info('Validation complete. All examples passed!')


def generate_dataset():
    """
    Generate a dataset of synthetic crystal images.
    """
    dataset_args, runtime_args = parse_args()
    set_seed(runtime_args.seed)
    save_dir = LOGS_PATH / runtime_args.ds_name
    if save_dir.exists():
        if runtime_args.overwrite_existing:
            logger.warning(f'Overwriting existing dataset at {save_dir}.')
            shutil.rmtree(save_dir)
        else:
            # Load arguments
            assert (save_dir / 'options.yml').exists(), f'Options file does not exist: {save_dir / "options.yml"}'
            with open(save_dir / 'options.yml', 'r') as f:
                args = yaml.load(f, Loader=yaml.FullLoader)
                dataset_args = DatasetSyntheticArgs.from_args(args['dataset_args'])

            # If parameters and images dir exists then resume
            param_path = save_dir / 'parameters.csv'
            images_dir = save_dir / 'images'
            if param_path.exists() and images_dir.exists():
                if runtime_args.generate_only:
                    logger.info(f'Crystal parameters already exists at {save_dir}. Aborting since generate_only=True.')
                    return
                logger.info(f'Dataset already exists at {save_dir}. Resuming...')
                return resume(save_dir, runtime_args)

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Create a directory to save dataset and logs
    param_path = save_dir / 'parameters.csv'
    images_dir = save_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create a directory to save clean images
    if dataset_args.generate_clean:
        clean_images_dir = save_dir / 'images_clean'
        clean_images_dir.mkdir(exist_ok=True)

    # Save arguments to json file
    with open(save_dir / 'options.yml', 'w') as f:
        spec = {
            'created': START_TIMESTAMP,
            'dataset_args': to_dict(dataset_args),
        }
        yaml.dump(spec, f)

    # Initialise synthetic crystal generator
    generator = CrystalGenerator(
        crystal_id=dataset_args.crystal_id,
        miller_indices=dataset_args.miller_indices,
        ratio_means=dataset_args.ratio_means,
        ratio_stds=dataset_args.ratio_stds,
        zingg_bbox=dataset_args.zingg_bbox,
        constraints=dataset_args.distance_constraints,
        asymmetry=dataset_args.asymmetry,
        n_workers=runtime_args.n_generator_workers
    )

    # Generate randomised crystals
    logger.info('Generating crystals.')
    crystals = generator.generate_crystals(num=dataset_args.n_samples)

    # Save parameters to a csv file
    logger.info(f'Saving crystal parameters to {param_path}.')
    with open(param_path, 'w') as f:
        headers = PARAMETER_HEADERS.copy()
        if dataset_args.asymmetry is None:
            miller_idxs = dataset_args.miller_indices
        else:
            miller_idxs = generator.crystal.all_miller_indices.tolist()
        ref_idxs = [''.join(str(i) for i in k) for k in miller_idxs]
        for i, hkl in enumerate(ref_idxs):
            headers.append(f'd{i}_{hkl}')
        for i, hkl in enumerate(ref_idxs):
            headers.append(f'a{i}_{hkl}')
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for i, (distances, areas, zingg_vals, _) in enumerate(crystals):
            entry = {
                'crystal_id': generator.crystal_id,
                'idx': i,
                'image': f'{i:010d}.png',
                'si': zingg_vals[0],
                'il': zingg_vals[1],
            }
            for j, hkl in enumerate(ref_idxs):
                entry[f'd{j}_{hkl}'] = float(distances[j])
                entry[f'a{j}_{hkl}'] = float(areas[j])
            writer.writerow(entry)

    if runtime_args.generate_only:
        elapsed_time = time.time() - start_time
        logger.info(f'Finished in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s.')
        return

    # Render the crystals
    logger.info('Rendering crystals.')
    renderer = CrystalRenderer(
        param_path=param_path,
        dataset_args=dataset_args,
        quiet_render=False,
        n_workers=runtime_args.n_renderer_workers
    )
    renderer.render()

    # Check if there are any other workers still active
    if renderer.is_active():
        logger.warning('Found other workers still active. Skipping validation and annotation.')
        exit()

    # Annotate a single image for reference
    if hasattr(renderer, 'annotate_image'):
        renderer.annotate_image()

    # Re-generate a few images from the parameters to check that they match
    validate(output_dir=save_dir, runtime_args=runtime_args)

    # Show how long this took, formatted nicely
    elapsed_time = time.time() - start_time
    logger.info(f'Finished in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s.')


def resume(
        save_dir: Path,
        runtime_args: Namespace,
        revalidate: bool = True,
        attempt: int = 1,
        max_attempts: int = 10,
        sleep_time: int = 60
):
    """
    Re-render any missing images in a dataset.
    Useful for fixing a broken run.
    """
    try:
        # Load arguments
        assert (save_dir / 'options.yml').exists(), f'Options file does not exist: {save_dir / "options.yml"}'
        with open(save_dir / 'options.yml', 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
            dataset_args = DatasetSyntheticArgs.from_args(args['dataset_args'])

        # Check parameters and images dir exists
        param_path = save_dir / 'parameters.csv'
        assert param_path.exists(), f'Parameters file does not exist: {param_path}'
        images_dir = save_dir / 'images'
        assert images_dir.exists(), f'Images directory does not exist: {images_dir}'
    except Exception as e:
        if attempt < max_attempts:
            logger.warning(f'Dataset does not appear ready to be resumed. '
                           f'Sleeping for {sleep_time}s ({attempt}/{max_attempts}). '
                           f'Error: {e}')
            time.sleep(sleep_time)
            return resume(save_dir, runtime_args, revalidate, attempt + 1, max_attempts, sleep_time)
        raise RuntimeError(f'Failed to resume dataset: {e}')

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Check for any errored renders as we need to create new crystals shapes for these
    if (save_dir / 'errored.json').exists():
        with open(save_dir / 'errored.json', 'r') as f:
            errored = json.load(f)
        logger.info(f'Found {len(errored)} failed renders, rebuilding crystals '
                    f'and updating parameters for these entries.')

        # Generate some replacement crystals
        generator = CrystalGenerator(
            crystal_id=dataset_args.crystal_id,
            miller_indices=dataset_args.miller_indices,
            ratio_means=dataset_args.ratio_means,
            ratio_stds=dataset_args.ratio_stds,
            zingg_bbox=dataset_args.zingg_bbox,
            constraints=dataset_args.distance_constraints,
            asymmetry=dataset_args.asymmetry,
            n_workers=runtime_args.n_generator_workers
        )
        crystals = generator.generate_crystals(num=len(errored))
        hkls = [''.join(str(i) for i in k) for k in generator.distances.keys()]

        # Load the parameters
        param_bkup_path = save_dir / f'parameters.bkup_{START_TIMESTAMP}.csv'
        shutil.copy(param_path, param_bkup_path)
        logger.info(f'Backed up parameters to {param_bkup_path}.')
        with open(param_path, mode='r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        # Update the errored entries
        updated_idxs = []
        for i, (idx, dets) in enumerate(errored.items()):
            rel_rates, zingg_vals, mesh = crystals[i]

            # Update the parameters
            row_idx = next((ri for ri, row in enumerate(rows) if row['idx'] == str(idx)), None)
            if row_idx is None:
                logger.warning(f'Failed to find index {idx} in parameters file. This will need fixing...')
                continue
            rows[row_idx]['si'] = zingg_vals[0]
            rows[row_idx]['il'] = zingg_vals[1]
            for j, hkl in enumerate(hkls):
                rows[row_idx][f'd{j}_{hkl}'] = rel_rates[j]
            updated_idxs.append(idx)

        # Save updated parameters back to file
        logger.info(f'Saving updated crystal parameters to {param_path}.')
        with open(param_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Delete errored idxs
        for idx in updated_idxs:
            del errored[idx]
        if len(errored) == 0:
            logger.info('All errored entries have been re-generated.')
            (save_dir / 'errored.json').unlink()
        else:
            with open(save_dir / 'errored.json', 'w') as f:
                json.dump(errored, f)
            raise RuntimeError('Some errored entries could not be fixed.')

    # Now just re-run the render which should catch any missing images
    renderer = CrystalRenderer(
        param_path=save_dir / 'parameters.csv',
        dataset_args=dataset_args,
        quiet_render=True,
        n_workers=runtime_args.n_renderer_workers,
        remove_mismatched=True,
        migrate_distances=runtime_args.migrate_distances
    )
    renderer.render()

    # Check if there are any other workers still active
    if renderer.is_active():
        logger.warning('Found other workers still active. Skipping validation and annotation.')
        exit()

    # Annotate a single image for reference
    renderer.annotate_image()

    # Re-generate a few images from the parameters to check that they match
    if revalidate:
        validate(output_dir=save_dir, runtime_args=runtime_args)

    # Show how long this took, formatted nicely
    elapsed_time = time.time() - start_time
    logger.info(f'Finished in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s.')


if __name__ == '__main__':
    generate_dataset()

    # -- Use to re-validate a directory --
    # validate(
    #     output_dir=LOGS_PATH / '20231130_1608',
    # )
