from abc import ABC
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as DatasetTorch, default_collate
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from crystalsizer3d.nn.dataset import Dataset


def get_affine_transforms() -> transforms.Compose:
    raise NotImplementedError('Affine transforms not implemented.')
    return transforms.Compose([
        transforms.RandomOrder([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
    ])


def get_image_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=9, sigma=(0.01, 5)),
        ])
    ])


class DatasetLoader(DatasetTorch, ABC):
    def __init__(
            self,
            ds: Dataset,
            augment: bool,
            train_or_test: str,
    ):
        assert train_or_test in ['train', 'test']
        self.ds = ds
        self.idxs = list(getattr(ds, train_or_test + '_idxs'))
        self.train_or_test = train_or_test
        self.augment = augment
        if self.augment:
            self.image_transforms = get_image_transforms()
            # todo: get_affine_transforms() - labels need changing too with this
            self.affine_transforms = None
        else:
            self.image_transforms = None
            self.affine_transforms = None

    def __getitem__(self, index: int) -> Tuple[dict, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        ds_idx = self.idxs[index]
        item, image, params = self.ds.load_item(ds_idx)
        image = to_tensor(image)
        params = {
            k: torch.from_numpy(v).to(torch.float32) if isinstance(v, np.ndarray) else v
            for k, v in params.items()
        }

        if self.augment:
            # image = self.affine_transforms(image)
            image_aug = self.image_transforms(image)
        else:
            image_aug = image.clone()

        return item, image, image_aug, params

    def __len__(self) -> int:
        return self.ds.get_size(self.train_or_test)


def get_data_loader(
        ds: Dataset,
        augment: bool,
        train_or_test: str,
        batch_size: int,
        n_workers: int
) -> DataLoader:
    """
    Get a data loader.
    """
    assert train_or_test in ['train', 'test']

    def collate_fn(batch):
        transposed = list(zip(*batch))
        ret = []
        ret.append(transposed[0])  # meta
        ret.append(default_collate(transposed[1]))  # image
        ret.append(default_collate(transposed[2]))  # image_aug

        def collate_targets(targets):
            data = {}
            for k in targets[0].keys():
                if k in ['sym_rotations', 'vertices']:
                    data[k] = [target[k] for target in targets]
                else:
                    data[k] = default_collate([target[k] for target in targets])
            return data

        ret.append(collate_targets(transposed[3]))  # params
        return ret

    dataset_loader = DatasetLoader(
        ds=ds,
        augment=augment,
        train_or_test=train_or_test,
    )

    loader = torch.utils.data.DataLoader(
        dataset_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=1 if n_workers > 0 else None
    )

    return loader
