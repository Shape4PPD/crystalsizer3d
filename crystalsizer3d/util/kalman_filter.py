import torch
import torch.nn as nn
from torch import Tensor


class KalmanFilter(nn.Module):
    state: Tensor
    F: Tensor
    H: Tensor
    P: Tensor
    Q: Tensor
    R: Tensor

    def __init__(
            self,
            param_dim: int,
            process_variance: float = 1e-3,
            measurement_variance: float = 1e-1
    ):
        """
        Initialise the Kalman filter.
        """
        super().__init__()
        self.param_dim = param_dim

        # Initialise the state vector [position; velocity; acceleration]
        self.register_buffer('state', torch.zeros(3 * param_dim, requires_grad=False))

        # State transition matrix for constant acceleration model
        F = torch.eye(3 * param_dim)
        F[:param_dim, param_dim:2 * param_dim] = torch.eye(param_dim)
        F[:param_dim, 2 * param_dim:] = 0.5 * torch.eye(param_dim)
        F[param_dim:2 * param_dim, 2 * param_dim:] = torch.eye(param_dim)
        self.register_buffer('F', F)

        # Measurement matrix (we only observe position, not velocity or acceleration)
        H = torch.zeros(param_dim, 3 * param_dim)
        H[:, :param_dim] = torch.eye(param_dim)
        self.register_buffer('H', H)

        # Covariance matrices
        self.register_buffer('P', torch.eye(3 * param_dim))  # Initial state covariance
        self.register_buffer('Q', process_variance * torch.eye(3 * param_dim))  # Process noise
        self.register_buffer('R', measurement_variance * torch.eye(param_dim))  # Measurement noise

        # Step counters
        self.predict_count = 0
        self.update_count = 0

    def predict(self) -> Tensor:
        """
        Predict the next state and covariance.
        """
        if self.predict_count > self.update_count:
            return self.state[:self.param_dim]
        self.predict_count += 1

        # Predict the state vector
        self.state = self.F @ self.state

        # Predict the error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Return the predicted position (first `param_dim` elements)
        return self.state[:self.param_dim]

    def update(self, measurement: Tensor) -> None:
        """
        Update the state with a new measurement.
        """
        if self.update_count >= self.predict_count:
            self.predict()
            return self.update(measurement)
        self.update_count += 1
        if measurement.dtype != torch.float32:
            measurement = measurement.to(torch.float32)

        # Measurement residual
        y = measurement - (self.H @ self.state)

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.inverse(S)

        # Update the state vector
        self.state = self.state + K @ y

        # Update the error covariance
        I = torch.eye(self.P.size(0), device=self.P.device)
        self.P = (I - K @ self.H) @ self.P
