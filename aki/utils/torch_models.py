import torch
from torch.nn import Module


class SVMLinear(Module):
    def __init__(self, n_input_features: int, n_classes: int):
        super().__init__()
        self._linear = torch.nn.Linear(in_features=n_input_features, out_features=n_classes, bias=True)
        torch.nn.init.xavier_uniform_(self._linear.weight)
        self._linear.bias.data.fill_(0.0)

    def forward(self, x):
        x = self._linear(x)
        return x
