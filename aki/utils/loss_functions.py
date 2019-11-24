import torch
from torch.nn.modules.module import Module


class HingeLoss(Module):
    def __init__(self, c: float = 1.0):
        super().__init__()
        self._c = 0.5 * c
        self._clamp = torch.nn.ReLU()

    def forward(self, pred, target, w):
        loss = self._clamp(1 - pred * target)
        return torch.sum(loss, dim=0) + self._c * w.norm(2)
