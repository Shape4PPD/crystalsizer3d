from abc import ABC
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset as DatasetTorch, default_collate
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs
from crystalsizer3d.nn.dataset import Dataset


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 1.0):
        super().__init__()
        assert 0 <= sigma_min <= sigma_max, 'sigma_min should be smaller than sigma_max'
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor) -> Tensor:
        sigma = self.get_params(self.sigma_min, self.sigma_max)
        return img + sigma * torch.randn_like(img)

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}(sigma_min={self.sigma_min}, sigma_max={self.sigma_max})'
        return s


class DatasetLoader(DatasetTorch, ABC):
    def __init__(
            self,
            ds: Dataset,
            dst_args: DatasetTrainingArgs,
            train_or_test: str,
    ):
        assert train_or_test in ['train', 'test']
        self.ds = ds
        self.idxs = list(getattr(ds, train_or_test + '_idxs'))
        self.train_or_test = train_or_test
        self.dst_args = dst_args
        if train_or_test == 'train' and self.dst_args.augment:
            self.image_transforms = self._init_image_transforms()
        else:
            self.image_transforms = None

    def _init_image_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=9,
                        sigma=(
                            self.dst_args.augment_blur_min_sigma,
                            self.dst_args.augment_blur_max_sigma
                        )),
                ], p=self.dst_args.augment_blur_prob),
                transforms.RandomApply([
                    GaussianNoise(
                        sigma_min=self.dst_args.augment_noise_min_sigma,
                        sigma_max=self.dst_args.augment_noise_max_sigma
                    ),
                ], p=self.dst_args.augment_noise_prob),
            ]),
        ])

    def __getitem__(self, index: int) \
            -> Tuple[dict, Tensor, Tensor, Optional[Tensor], Dict[str, Tensor]]:
        ds_idx = self.idxs[index]
        item, image, image_clean, params = self.ds.load_item(ds_idx)
        image = to_tensor(image)
        if image_clean is not None:
            image_clean = to_tensor(image_clean)
        params = {
            k: torch.from_numpy(v).to(torch.float32) if isinstance(v, np.ndarray) else v
            for k, v in params.items()
        }

        if self.image_transforms is not None:
            image_aug = self.image_transforms(image)
        else:
            image_aug = image.clone()

        return item, image, image_aug, image_clean, params

    def __len__(self) -> int:
        return self.ds.get_size(self.train_or_test)


def get_data_loader(
        ds: Dataset,
        dst_args: DatasetTrainingArgs,
        train_or_test: str,
        batch_size: int,
        n_workers: int,
        prefetch_factor: int = 1,
) -> DataLoader:
    """
    Get a data loader.
    """
    assert train_or_test in ['train', 'test']

    def collate_fn(batch):
        transposed = list(zip(*batch))

        def collate_targets(targets):
            data = {}
            for k in targets[0].keys():
                if k in ['sym_rotations', 'vertices']:
                    data[k] = [target[k] for target in targets]
                else:
                    data[k] = default_collate([target[k] for target in targets])
            return data

        ret = [
            transposed[0],  # meta
            default_collate(transposed[1]),  # image
            default_collate(transposed[2]),  # image_aug
            default_collate(transposed[3]) if transposed[3][0] is not None else None,  # image_clean
            collate_targets(transposed[4]),  # params
        ]

        return ret

    dataset_loader = DatasetLoader(
        ds=ds,
        dst_args=dst_args,
        train_or_test=train_or_test,
    )

    loader = torch.utils.data.DataLoader(
        dataset_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor if n_workers > 0 else None
    )

    return loader
