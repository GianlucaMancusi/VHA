import torch
from torch import Tensor


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, predict: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return torch.sum(((predict - target) * mask) ** 2.0) / (torch.sum(mask) + 1e-10)
