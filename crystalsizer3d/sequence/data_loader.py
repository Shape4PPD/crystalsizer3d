from pathlib import Path
from typing import List, TYPE_CHECKING, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from crystalsizer3d.sequence.adaptive_sampler import AdaptiveSampler

if TYPE_CHECKING:
    from crystalsizer3d.sequence.sequence_fitter import SequenceFitter


class Dataset(IterableDataset):
    def __init__(
            self,
            X_paths: List[Path],
            X_dn_paths: List[Path],
            X_wis: List[Tensor],
            X_dn_wis: List[Tensor],
            keypoints: List[Tensor] | None,
            edges: List[Path] | None,
            adaptive_sampler: AdaptiveSampler,
            batch_size: int = 1,
    ):
        assert len(X_paths) == len(X_wis) == len(X_dn_paths) == len(X_dn_wis)
        if keypoints is not None:
            assert len(keypoints) == len(X_paths)
        if edges is not None:
            assert len(edges) == len(X_paths)
        self.X_paths = X_paths
        self.X_dn_paths = X_dn_paths
        self.X_wis = X_wis
        self.X_dn_wis = X_dn_wis
        self.keypoints = keypoints
        self.edges = edges
        self.adaptive_sampler = adaptive_sampler
        self.batch_size = batch_size

    def __getitem__(self, index: int) -> tuple[Path, Path, Tensor, Tensor, Tensor | None, Tensor | None]:
        return (self.X_paths[index],
                self.X_dn_paths[index],
                self.X_wis[index],
                self.X_dn_wis[index],
                self.keypoints[index]['keypoints'] if self.keypoints is not None else None,
                self.edges[index] if self.edges is not None else None)

    def __len__(self) -> int:
        return len(self.X_paths)

    def __iter__(self):
        while True:
            idxs = self.adaptive_sampler.sample_frames(self.batch_size)
            yield idxs, [self[idx] for idx in idxs]


def collate_fn(batch) -> Tuple[Tensor, List[Tensor | None]]:
    """
    Collate function for the data loader.
    """
    indices, Xs = batch[0][0], batch[0][1]
    X_paths = [X[0] for X in Xs]
    X_dn_paths = [X[1] for X in Xs]
    X_wis = torch.stack([X[2] for X in Xs]).permute(0, 2, 3, 1)
    X_dn_wis = torch.stack([X[3] for X in Xs]).permute(0, 2, 3, 1)
    keypoints = [X[4] for X in Xs] if Xs[0][4] is not None else [None for _ in Xs]
    edges = [X[5] for X in Xs] if Xs[0][5] is not None else [None for _ in Xs]
    targets = [X_paths, X_dn_paths, X_wis, X_dn_wis, keypoints, edges]
    return indices, targets


def get_data_loader(
        sequence_fitter: 'SequenceFitter',
        adaptive_sampler: AdaptiveSampler,
        batch_size: int,
        n_workers: int,
        prefetch_factor: int = 1,
) -> DataLoader:
    """
    Get a data loader.
    """
    dataset = Dataset(
        X_paths=sequence_fitter.X_targets,
        X_dn_paths=sequence_fitter.X_targets_denoised,
        X_wis=sequence_fitter.X_wis['og'],
        X_dn_wis=sequence_fitter.X_wis['dn'],
        keypoints=sequence_fitter.keypoints,
        edges=sequence_fitter.edges,
        adaptive_sampler=adaptive_sampler,
        batch_size=batch_size
    )

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        num_workers=n_workers,
        prefetch_factor=prefetch_factor if n_workers > 0 else None
    )

    return loader
