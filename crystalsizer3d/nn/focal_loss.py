import torch
from torch import Tensor, nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCELoss(reduction='none')  # Binary Cross-Entropy

    def forward(self, inputs: Tensor, targets: Tensor):
        # BCE loss per pixel
        bce_loss = self.bce_loss(inputs, targets)
        # Calculate pt (probability of correct class)
        pt = torch.exp(-bce_loss)
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()
