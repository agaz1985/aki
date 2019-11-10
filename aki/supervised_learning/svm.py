import h5py
import torch

from torch.nn import Module

from aki.utils.loss_functions import HingeLoss
from aki.utils.model import ModelBase


class SVMLinear(Module):
    def __init__(self, n_input_features: int, n_classes: int):
        super().__init__()
        self._linear = torch.nn.Linear(in_features=n_input_features, out_features=n_classes, bias=True)

    def forward(self, x):
        x = self._linear(x)
        return x


class SVM(ModelBase):
    def __init__(self, device: torch.device):
        super().__init__(device, "svm")

    def _init_model(self, input_shape, n_classes):
        self._model = SVMLinear(n_input_features=input_shape[1], n_classes=n_classes)
        self._C = 1
        self._loss = HingeLoss(self._C, )
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-3)

    def _predict_implementation(self, *argparams, **kwparams):
        pass

    def _save_implementation(self, state_object: h5py.File):
        pass

    def _load_implementation(self, state_object: h5py.File):
        pass

    def _fit_implementation(self, x: torch.Tensor, y: torch.Tensor, n_classes: int, max_number_iterations: int,
                            eps: float):
        self._init_model(x.shape, n_classes)

        self._model.train()
        last_loss = None
        for epoch in range(max_number_iterations):
            self._optimizer.zero_grad()
            pred = self._model(x)
            loss = self._loss(pred, y, self._model._linear.weight)
            loss.backward()
            self._optimizer.step()
            if last_loss is not None and (last_loss - loss.item()) < eps:
                break
            last_loss = loss.item()

        self._model.eval()
        with torch.no_grad():
            return self._model(x), self._model._linear.weight[0].detach().cpu().numpy(), self._model._linear.bias[
                0].detach().cpu().numpy()
