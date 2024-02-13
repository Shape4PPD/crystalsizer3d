from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path
from typing import List, Union

from crystalsizer3d import USE_CUDA
from crystalsizer3d.args.base_args import BaseArgs
from crystalsizer3d.util.utils import str2bool


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            resume: bool = True,
            resume_from: Union[str, Path] = 'latest',
            resume_only: bool = False,
            use_gpu: bool = False,
            n_dataloader_workers: int = 4,
            batch_size: int = 32,
            n_epochs: int = 300,
            checkpoint_every_n_epochs: int = 1,
            checkpoint_every_n_batches: int = -1,
            max_checkpoints: int = 5,
            test_every_n_epochs: int = 1,
            log_every_n_batches: int = 1,
            plot_every_n_batches: int = -1,
            plot_n_examples: int = 4,
            save_plots: bool = True,
            save_plots_to_tb: bool = True,
            track_metrics: List[str] = [],
            **kwargs
    ):
        self.resume = resume
        if isinstance(resume_from, Path):
            resume_from = str(resume_from)
        self.resume_from = resume_from
        self.resume_only = resume_only
        self.use_gpu = use_gpu
        self.n_dataloader_workers = n_dataloader_workers
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.checkpoint_every_n_batches = checkpoint_every_n_batches
        self.max_checkpoints = max_checkpoints
        self.test_every_n_epochs = test_every_n_epochs
        self.log_every_n_batches = log_every_n_batches
        self.plot_every_n_batches = plot_every_n_batches
        self.plot_n_examples = plot_n_examples
        self.save_plots = save_plots
        self.save_plots_to_tb = save_plots_to_tb
        self.track_metrics = track_metrics

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Runtime Args')
        resume_parser = group.add_mutually_exclusive_group(required=False)
        resume_parser.add_argument('--resume', action='store_true',
                                   help='Resume from a previous checkpoint.')
        resume_parser.add_argument('--no-resume', action='store_false', dest='resume',
                                   help='Do not resume from a previous checkpoint.')
        resume_parser.set_defaults(resume=False)
        group.add_argument('--resume-from', type=str, default='latest',
                           help='Resume from a specific checkpoint id, or "latest" or "best". Default="latest".')
        group.add_argument('--resume-only', type=str2bool, default=False,
                           help='Abort if the checkpoint can\'t be loaded. Default=False.')
        parser.add_argument('--use-gpu', type=str2bool, default=USE_CUDA,
                            help='Use GPU. Defaults to environment setting.')
        group.add_argument('--n-dataloader-workers', type=int, default=4,
                           help='Number of dataloader worker processes.')
        group.add_argument('--batch-size', type=int, default=8,
                           help='Batch size to use for training and testing')
        group.add_argument('--n-epochs', type=int, default=300,
                           help='Number of epochs to run for.')
        group.add_argument('--checkpoint-every-n-epochs', type=int, default=1,
                           help='Save a checkpoint every n epochs, -1 turns this off.')
        group.add_argument('--checkpoint-every-n-batches', type=int, default=-1,
                           help='Save a checkpoint every n batches, -1 turns this off.')
        group.add_argument('--max-checkpoints', type=int, default=5,
                           help='Maximum number of checkpoints to keep.')
        group.add_argument('--test-every-n-epochs', type=int, default=1,
                           help='Test every n epochs, -1 turns this off.')
        group.add_argument('--log-every-n-batches', type=int, default=1,
                           help='Log metrics every n batches.')
        group.add_argument('--plot-every-n-batches', type=int, default=10,
                           help='Plot example inputs and outputs every n batches, -1 turns this off.')
        group.add_argument('--plot-n-examples', type=int, default=4,
                           help='Show this many random examples in a single plot.')
        group.add_argument('--save-plots', type=str2bool, default=True,
                           help='Save plot images to disk. Default = True.')
        group.add_argument('--save-plots-to-tb', type=str2bool, default=True,
                           help='Save plot images to TensorBoard. Default = True.')
        group.add_argument('--track-metrics', type=lambda s: [str(item) for item in s.split(',')], default=[],
                           help='Comma delimited list of metrics to track.')

        return group
