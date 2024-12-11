import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import yaml

from crystalsizer3d import logger
from crystalsizer3d.args.refiner_args import RefinerArgs
from crystalsizer3d.args.sequence_fitter_args import SequenceFitterArgs
from crystalsizer3d.sequence.sequence_fitter import SequenceFitter
from crystalsizer3d.util.utils import print_args, str2bool


def get_args(printout: bool = True) -> Tuple[Namespace, SequenceFitterArgs, RefinerArgs]:
    """
    Parse command line arguments.
    """
    rt_parser = ArgumentParser(description='CrystalSizer3D script to fit a crystal growth sequence.')
    sf_parser = ArgumentParser()
    ref_parser = ArgumentParser()

    # Runtime args
    rt_parser.add_argument('--args-path', type=Path,
                           help='Load refiner arguments from this path, any arguments set on the command-line will take preference.')
    rt_parser.add_argument('--resume', type=str2bool, default=True,
                           help='Resume training from a previous checkpoint.')
    rt_parser.add_argument('--resume-from', type=Path,
                           help='Resume training from a different configuration, only really for debug.')
    rt_parser.add_argument('--reset-lrs', type=str2bool, default=False,
                           help='When resuming, reset the learning rate and scheduler.')
    rt_parser.add_argument('--make-videos', type=str2bool, default=True,
                           help='Make video of the annotated masks/images (whatever was generated).')
    rt_parser.add_argument('--log-freq-pretrain', type=int, default=10,
                           help='Log every n batches during pretraining.')
    rt_parser.add_argument('--log-freq-train', type=int, default=5,
                           help='Log every n steps during training.')
    rt_parser.add_argument('--checkpoint-freq', type=int, default=5,
                           help='Checkpoint every n steps during training.')
    rt_parser.add_argument('--save-annotations-freq', type=int, default=5,
                           help='Save annotated images every n steps during training.')
    rt_parser.add_argument('--save-renders-freq', type=int, default=5,
                           help='Save rendered images every n steps during training.')
    rt_parser.add_argument('--plot-freq', type=int, default=20,
                           help='Make plots every n steps during training.')
    rt_parser.add_argument('--eval-freq', type=int, default=10,
                           help='Evaluate every n steps during training.')
    rt_parser.add_argument('--eval-annotate-freq', type=int, default=10,
                           help='Generate annotated images every n steps during training. (Must be a multiple of eval-freq).')
    rt_parser.add_argument('--eval-render-freq', type=int, default=10,
                           help='Render the evaluated parameters every n steps during training. (Must be a multiple of eval-freq).')
    rt_parser.add_argument('--eval-video-freq', type=int, default=10,
                           help='Make a video of the evaluated parameters every n steps during training. (Must be a multiple of eval-freq).')
    rt_parser.add_argument('--n-plotting-workers', type=int, default=8,
                           help='Number of plotting workers.')
    rt_parser.add_argument('--plot-queue-size', type=int, default=100,
                           help='Size of the plotting queue.')
    rt_parser.add_argument('--n-dataloader-workers', type=int, default=2,
                           help='Number of data loader workers.')
    rt_parser.add_argument('--prefetch-factor', type=int, default=1,
                           help='Data loader prefetch factor.')
    rt_parser.add_argument('--n-refiner-workers', type=int, default=2,
                           help='Number of refiner workers.')
    rt_parser.add_argument('--refiner-queue-size', type=int, default=100,
                           help='Size of the refiner queue.')
    rt_parser.add_argument('--measurements-dir', type=Path,
                           help='Path to a directory containing manual measurements.')

    # Sequence fitter and refiner args
    SequenceFitterArgs.add_args(sf_parser)
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
    sf_args = SequenceFitterArgs.from_args(sf_args)
    ref_args = RefinerArgs.from_args(ref_args)

    # Check arguments are valid
    assert sf_args.images_dir.exists(), f'Images directory {sf_args.images_dir} does not exist.'

    # Remove the args_path argument as it is not needed
    delattr(rt_args, 'args_path')

    return rt_args, sf_args, ref_args


def train_sequence():
    """
    Train a neural network to track a crystal growth sequence.
    """
    start_time = time.time()
    rt_args, sf_args, ref_args = get_args()

    sequence = SequenceFitter(
        sf_args=sf_args,
        refiner_args=ref_args,
        runtime_args=rt_args,
    )
    sequence.fit()

    # Print how long this took - split into hours, minutes, seconds
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'Finished in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.')


if __name__ == '__main__':
    train_sequence()
