from argparse import ArgumentParser
from typing import Tuple

from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs
from crystalsizer3d.args.generator_args import GeneratorArgs
from crystalsizer3d.args.network_args import NetworkArgs
from crystalsizer3d.args.optimiser_args import OptimiserArgs
from crystalsizer3d.args.runtime_args import RuntimeArgs
from crystalsizer3d.args.transcoder_args import TranscoderArgs
from crystalsizer3d.nn.manager import Manager
from crystalsizer3d.util.utils import print_args


def parse_arguments(printout: bool = True) \
        -> Tuple[DatasetTrainingArgs, NetworkArgs, GeneratorArgs, TranscoderArgs, OptimiserArgs, RuntimeArgs]:
    """
    Parse command line arguments and build parameter holders.
    """
    parser = ArgumentParser(description='Train/Test crystal sizer network')

    DatasetTrainingArgs.add_args(parser)
    NetworkArgs.add_args(parser)
    GeneratorArgs.add_args(parser)
    TranscoderArgs.add_args(parser)
    OptimiserArgs.add_args(parser)
    RuntimeArgs.add_args(parser)

    # Do the parsing
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the parameter holders
    dataset_args = DatasetTrainingArgs.from_args(args)
    net_args = NetworkArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)
    transcoder_args = TranscoderArgs.from_args(args)
    optimiser_args = OptimiserArgs.from_args(args)
    runtime_args = RuntimeArgs.from_args(args)

    return dataset_args, net_args, generator_args, transcoder_args, optimiser_args, runtime_args


def train():
    """
    Trains a network to estimate crystal growth parameters from images.
    """
    dataset_args, net_args, generator_args, transcoder_args, optimiser_args, runtime_args = parse_arguments()

    # Construct manager
    manager = Manager(
        dataset_args=dataset_args,
        net_args=net_args,
        generator_args=generator_args,
        transcoder_args=transcoder_args,
        optimiser_args=optimiser_args,
        runtime_args=runtime_args
    )

    # Generate the neural network computation graph (view in tensorboard)
    # manager.log_graph()

    # Do some training
    manager.train(
        n_epochs=runtime_args.n_epochs
    )


if __name__ == '__main__':
    train()
