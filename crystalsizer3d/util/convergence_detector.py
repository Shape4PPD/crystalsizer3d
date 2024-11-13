from typing import Tuple

import torch
import torch.nn as nn


class ConvergenceDetector(nn.Module):
    tau_fast: torch.Tensor
    tau_slow: torch.Tensor
    val: torch.Tensor
    mu_fast: torch.Tensor
    mu_slow: torch.Tensor
    T_upper: torch.Tensor
    T_lower: torch.Tensor
    convergence_count: torch.Tensor
    converged: torch.Tensor
    threshold: float
    patience: int

    continuous_state_vars = [
        'tau_fast',
        'tau_slow',
        'val',
        'mu_fast',
        'mu_slow',
        'T_upper',
        'T_lower',
        'convergence_count'
    ]
    binary_state_vars = ['converged']

    def __init__(
            self,
            shape: Tuple[int, ...],
            tau_fast: int = 10,
            tau_slow: int = 100,
            threshold: float = 0.1,
            min_absolute_threshold: float = 1e-6,
            patience: int = 100
    ):
        super().__init__()
        for x_name in self.continuous_state_vars:
            self.register_buffer(x_name, torch.zeros(shape, dtype=torch.float32))
        for x_name in self.binary_state_vars:
            self.register_buffer(x_name, torch.zeros(shape, dtype=torch.bool))

        self.tau_fast.fill_(tau_fast)
        self.tau_slow.fill_(tau_slow)
        self.threshold = threshold
        self.min_absolute_threshold = min_absolute_threshold
        self.patience = patience

    def forward(self, val: torch.Tensor, first_val: bool = False):
        """
        Update the moving averages and determine convergence.
        """
        self.val = val.detach().clone()
        if first_val:
            self.mu_fast = self.val.clone()
            self.mu_slow = self.val.clone()
        self._update_state()
        self._update_estimates()

    @torch.jit.export
    def reset_counters(self):
        """
        Reset the convergence counters
        """
        self.convergence_count.zero_()

    def _update_state(self):
        """
        If the slow estimate is within the bounds then increment a counter.
        If the counter exceeds the patience then we say it has converged.
        Otherwise, reset the counter.
        """
        in_bounds = self._check_bounds()
        self.convergence_count = torch.where(
            in_bounds,
            self.convergence_count + 1,
            torch.zeros_like(self.convergence_count)
        )
        self.converged = self.convergence_count >= self.patience

    def _update_estimates(self):
        """
        Update the fast and slow moving average estimates.
        """

        # Update fast estimate
        alpha_fast = 1 / self.tau_fast
        diff_fast = self.val - self.mu_fast
        self.mu_fast += alpha_fast * diff_fast

        # Update slow estimate
        alpha_slow = 1 / self.tau_slow
        diff_slow = self.val - self.mu_slow
        self.mu_slow += alpha_slow * diff_slow

        # Update threshold bounds
        buffer = torch.clamp_min(self.threshold * torch.abs(self.mu_slow), self.min_absolute_threshold)
        self.T_upper = self.mu_slow + buffer
        self.T_lower = self.mu_slow - buffer

    def _check_bounds(self) -> torch.Tensor:
        return (self.mu_fast > self.T_lower) & (self.mu_fast < self.T_upper)
