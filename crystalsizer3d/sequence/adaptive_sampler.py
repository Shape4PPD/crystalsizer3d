from typing import List

import torch
from torch import Tensor

from crystalsizer3d.util.ema import EMA


class AdaptiveSampler:
    def __init__(self, sequence_length: int, ema_decay: float = 0.9, ema_init: float = 1.0):
        self.sequence_length = sequence_length
        self.emas = [EMA(val=ema_init, decay=ema_decay) for _ in range(sequence_length)]

    @property
    def errors(self):
        return torch.tensor([ema.val for ema in self.emas])

    def update_errors(self, frame_indices: List[int], errors: Tensor):
        """
        Update exponential moving averages.
        """
        for i, frame_idx in enumerate(frame_indices):
            self.emas[frame_idx](errors[i].item())

    def get_sampling_probabilities(self):
        """
        Convert errors to probabilities by normalising to sum to 1.
        """
        errors = self.errors
        probabilities = errors / errors.sum()
        return probabilities

    def sample_frames(self, batch_size: int):
        """
        Sample frames according to the adaptive probabilities.
        """
        probabilities = self.get_sampling_probabilities()
        frame_indices = torch.multinomial(probabilities, batch_size, replacement=False)
        return frame_indices
