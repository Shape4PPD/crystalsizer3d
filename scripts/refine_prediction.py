from argparse import ArgumentParser

from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.refiner.refiner import Refiner
from crystalsizer3d.util.utils import print_args


def parse_arguments(printout: bool = True) -> RefinerArgs:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Predict and refine crystal parameters from an image.')
    RefinerArgs.add_args(parser)

    # Do the parsing
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
