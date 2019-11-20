import torch
from torch.nn.modules.module import Module


class HingeLoss(Module):
    def __init__(self, C: float = 1.0):
        super().__init__()
        self._C = C

    def forward(self, pred, target, w):
        loss = torch.nn.ReLU()(1 - pred * target).squeeze()
        return torch.sum(loss, dim=0) + (0.5 * self._C) * w.norm(2)
