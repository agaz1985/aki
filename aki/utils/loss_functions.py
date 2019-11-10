import torch
from torch.nn.modules.module import Module


class HingeLoss(Module):
    def __init__(self, C):
        super().__init__()
        self._C = C

    def forward(self, prediction, target, weights):
        loss = (1.0 - prediction.T * target).clamp(min=0)
        return torch.mean(self._C * (weights ** 2)) + torch.mean(loss)
