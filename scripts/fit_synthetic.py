import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import yaml

from crystalsizer3d import logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.args.synthetic_fitter_args import SyntheticFitterArgs
from crystalsizer3d.synthetic.synthetic_fitter import SyntheticFitter
from crystalsizer3d.util.utils import print_args, str2bool


def get_args(printout: bool = True) -> Tuple[Namespace, SyntheticFitterArgs, RefinerArgs]:
    """
    Parse command line arguments.
    """
    rt_parser = ArgumentParser(description='CrystalSizer3D script to fit a synthetic dataset.')
    sf_parser = ArgumentParser()
    ref_parser = ArgumentParser()

    # Runtime args
    rt_parser.add_argument('--args-path', type=Path,
                           help='Load refiner arguments from this path, any arguments set on the command-line will take preference.')
    rt_parser.add_argument('--resume', type=str2bool, default=True,
                           help='Resume training from a previous checkpoint.')
    rt_parser.add_argument('--resume-from', type=Path,
                           help='Resume training from a different configuration, only really for debug.')
    rt_parser.add_argument('--save-renders-freq', type=int, default=5,
                           help='Save rendered images every n steps during training.')
    rt_parser.add_argument('--n-plotting-workers', type=int, default=8,
                           help='Number of plotting workers.')
    rt_parser.add_argument('--plot-queue-size', type=int, default=100,
                           help='Size of the plotting queue.')
    rt_parser.add_argument('--n-refiner-workers', type=int, default=2,
                           help='Number of refiner workers.')
    rt_parser.add_argument('--refiner-queue-size', type=int, default=100,
                           help='Size of the refiner queue.')

    # Synthetic fitter and refiner args
    SyntheticFitterArgs.add_args(sf_parser)
    RefinerArgs.add_args(ref_parser)

    # Cache the args for each parser
    actions = {}
    for parser_name, parser in zip(['rt', 'sf', 'ref'], [rt_parser, sf_parser, ref_parser]):
        actions[parser_name] = []
        for action in parser._actions:
            actions[parser_name].append(action.dest)

    # Load any args from file and set these as the defaults for any arguments that weren't set
    cli_args, _ = rt_parser.parse_known_args()
    if cli_args.args_path is not None:
        assert cli_args.args_path.exists(), f'Args path does not exist: {cli_args.args_path}'
        with open(cli_args.args_path, 'r') as f:
            args_yml = yaml.load(f, Loader=yaml.FullLoader)

        defaults = {parser_name: {} for parser_name in ['rt', 'sf', 'ref']}
        for k, v in args_yml.items():
            for parser_name, parser in zip(['rt', 'sf', 'ref'], [rt_parser, sf_parser, ref_parser]):
                if k in actions[parser_name]:
                    defaults[parser_name][k] = v
                    break
        for parser_name, parser in zip(['rt', 'sf', 'ref'], [rt_parser, sf_parser, ref_parser]):
            parser.set_defaults(**defaults[parser_name])

    # Parse the command line arguments again
    rt_args, args_remaining = rt_parser.parse_known_args()
    sf_args, args_remaining = sf_parser.parse_known_args(args_remaining)
    ref_args, _ = ref_parser.parse_known_args(args_remaining)
    if printout:
        combined_args = Namespace(**{**vars(ref_args), **vars(sf_args), **vars(rt_args)})
        print_args(combined_args)

    # Instantiate the parameter holders
    sf_args = SyntheticFitterArgs.from_args(sf_args)
    ref_args = RefinerArgs.from_args(ref_args)

    # Check that the dataset path exists
    assert sf_args.dataset_path.exists(), f'Dataset path {sf_args.dataset_path} does not exist.'

    # Remove the args_path argument as it is not needed
    delattr(rt_args, 'args_path')

    return rt_args, sf_args, ref_args


def fit_synthetic():
    """
    Make initial predictions and then refine the parameters for a synthetic dataset.
    """
    start_time = time.time()
    rt_args, sf_args, ref_args = get_args()

    fitter = SyntheticFitter(
        sf_args=sf_args,
        refiner_args=ref_args,
        runtime_args=rt_args,
    )
    fitter.fit()

    # Print how long this took - split into hours, minutes, seconds
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'Finished in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.')


if __name__ == '__main__':
    fit_synthetic()
