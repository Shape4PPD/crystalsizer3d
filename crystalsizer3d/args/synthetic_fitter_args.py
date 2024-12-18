from argparse import ArgumentParser, _ArgumentGroup
from pathlib import Path

from crystalsizer3d.args.base_args import BaseArgs


class SyntheticFitterArgs(BaseArgs):
    def __init__(
            self,

            # Target dataset
            dataset_path: Path | str,
            train_or_test: str = 'test',
            n_samples: int = 1000,

            initial_scene: Path | str | None = None,

            # Optimisation settings
            seed: int | None = None,
            checkpoint_freq: int = 50,

            **kwargs
    ):
        # Convert string paths to Path objects
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        if isinstance(initial_scene, str):
            initial_scene = Path(initial_scene)

        # Target dataset
        self.dataset_path = dataset_path
        self.train_or_test = train_or_test
        self.n_samples = n_samples

        if initial_scene is not None:
            assert initial_scene.exists(), f'Initial scene file {initial_scene} does not exist.'
        self.initial_scene = initial_scene

        # Optimisation settings
        self.seed = seed
        self.checkpoint_freq = checkpoint_freq

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Synthetic Dataset Fitter Args')

        # Target sequence
        group.add_argument('--dataset-path', type=Path,
                           help='Path to the dataset.')
        group.add_argument('--train-or-test', type=str, default='test', choices=['train', 'test'],
                           help='Whether to use the training or test set.')
        group.add_argument('--n-samples', type=int, default=1000,
                           help='Number of samples to use from the dataset.')

        parser.add_argument('--initial-scene', type=Path,
                            help='Path to the initial scene file. Will be used for any fixed parameters.')

        # Optimisation settings
        group.add_argument('--seed', type=int,
                           help='Seed for the random number generator.')
        group.add_argument('--checkpoint-freq', type=int, default=50,
                           help='Checkpoint the losses and stats every n number of steps.')

        return group
