from argparse import ArgumentParser, Namespace
from pathlib import Path

import yaml

from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.util.utils import print_args


def parse_arguments(printout: bool = True) -> RefinerArgs:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Predict and refine crystal parameters from an image.')
    RefinerArgs.add_args(parser)
    parser.add_argument('--args-path', type=Path,
                        help='Load refiner arguments from this path, any arguments set on the command-line will take preference.')

    # Parse the command line arguments
    cli_args, _ = parser.parse_known_args()

    # Load any args from file and set these as the defaults for any arguments that weren't set
    if cli_args.args_path is not None:
        assert cli_args.args_path.exists(), f'Args path does not exist: {cli_args.args_path}'
        with open(cli_args.args_path, 'r') as f:
            args_yml = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**args_yml)

    # Parse the command line arguments again
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holder
    refiner_args = RefinerArgs.from_args(args)

    return refiner_args


def refine():
    """
    Predict and refine crystal parameters from an image.
    """
    refiner_args = parse_arguments()

    # Construct refiner
    refiner = Refiner(args=refiner_args)

    # Do the refining
    refiner.train()


if __name__ == '__main__':
    refine()
