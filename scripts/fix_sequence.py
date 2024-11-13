import json
import shutil
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
import yaml

from crystalsizer3d import START_TIMESTAMP, logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.crystal import Crystal
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.util.utils import FlexibleJSONEncoder, init_tensor, json_to_numpy, print_args, \
    str2bool

refiner: Refiner = None


def get_args(printout: bool = True) -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='CrystalSizer3D script to track crystal growth.')

    # Target run
    parser.add_argument('--run-dir', type=Path,
                        help='Directory containing the run data.')
    parser.add_argument('--overwrite', type=str2bool, default=False,
                        help='Overwrite existing images and videos.')

    # Parse the command line arguments
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Check arguments are valid
    assert args.run_dir.exists(), f'Run directory {args.run_dir} does not exist.'

    return args


@torch.no_grad()
def fix_sequence():
    """
    Load an existing run and fix the origin.
    """
    global refiner
    args = get_args()
    run_dir = args.run_dir

    # Set a timer going to record how long this takes
    start_time = time.time()

    # Load the args from the run dir
    with open(run_dir / 'args.yml', 'r') as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    args = Namespace(**args_dict)
    args.images_dir = Path(args.images_dir)
    save_dir_seq = Path(args.save_dir_seq)

    # Make save directory
    save_dir_fix = run_dir / f'fix_{START_TIMESTAMP}'
    if save_dir_fix.exists():
        shutil.rmtree(save_dir_fix)
    save_dir_fix.mkdir(parents=True, exist_ok=True)

    # Load the final parameters
    refiner_dir = save_dir_seq / 'refiner' / args.refiner_dir
    assert refiner_dir.exists(), f'Refiner output directory {refiner_dir} does not exist.'
    with open(refiner_dir / f'parameters_final.json', 'r') as f:
        data = json_to_numpy(json.load(f))

    # Instantiate a refiner
    refiner_args = RefinerArgs.from_args(args)
    refiner = Refiner(args=refiner_args, do_init=False)

    # Get a crystal object
    ds = refiner.manager.ds
    cs = ds.csd_proxy.load(ds.dataset_args.crystal_id)
    crystal = Crystal(
        lattice_unit_cell=cs.lattice_unit_cell,
        lattice_angles=cs.lattice_angles,
        miller_indices=ds.miller_indices,
        point_group_symbol=cs.point_group_symbol,
    )

    # Use the origin from the first frame for the other frames
    distances_old = data['distances']
    distances_new = np.zeros_like(distances_old)
    origin0 = init_tensor(data['origin'][0])
    for i in range(1, len(data['origin'])):
        crystal.distances.data = init_tensor(data['distances'][i])
        crystal.origin.data = init_tensor(data['origin'][i])
        crystal.adjust_origin(origin0)

    # Save the new parameters
    data['distances'] = distances_new
    data['origin'] = np.repeat([origin0], len(data['origin']), axis=0)
    with open(save_dir_fix / 'parameters_final.json', 'w') as f:
        json.dump(data, f, cls=FlexibleJSONEncoder)

    # Print how long this took - split into hours, minutes, seconds
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'Finished in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.')


if __name__ == '__main__':
    fix_sequence()
