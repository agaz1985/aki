import h5py
import torch

from aki.utils.loss_functions import HingeLoss
from aki.utils.model import ModelBase
from aki.utils.torch_models import SVMLinear


class SVM(ModelBase):
    def __init__(self, device: torch.device):
        super().__init__(device, "svm")

    def _init_model(self, input_shape: tuple, n_classes: int, learning_rate: float = 1e-3, C: float = 1.0):
        self._model = SVMLinear(n_input_features=input_shape[1], n_classes=n_classes).to(self._device)
        self._C = C
        self._loss = HingeLoss(self._C)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate)
        self._input_shape = input_shape
        self._n_classes = n_classes
        self._learning_rate = learning_rate

    def _predict_implementation(self, x: torch.Tensor):
        return self._model(x), self._model._linear.weight, self._model._linear.bias

    def _save_implementation(self, state_object: h5py.File):
        state_object.create_dataset('input_shape', data=self._input_shape)
        state_object.create_dataset('n_classes', data=self._n_classes)
        state_object.create_dataset('learning_rate', data=self._learning_rate)
        state_object.create_dataset('weight', data=self._model._linear.weight.detach().cpu().numpy())
        state_object.create_dataset('bias', data=self._model._linear.bias.detach().cpu().numpy())

    def _load_implementation(self, state_object: h5py.File):
        self._init_model(state_object['input_shape'][()],
                         state_object['n_classes'][()],
                         state_object['learning_rate'][()])
        self._model._linear.weight = torch.nn.Parameter(torch.from_numpy(state_object['weight'][()]))
        self._model._linear.weight.requires_grad = True
        self._model._linear.bias = torch.nn.Parameter(torch.from_numpy(state_object['bias'][()]))
        self._model._linear.bias.requires_grad = True

    def _fit_implementation(self, x: torch.Tensor, y: torch.Tensor, n_classes: int, max_number_iterations: int,
                            eps: float, lr: float, C: float):
        self._init_model(x.shape, n_classes, lr, C)
        y[y == 0] = -1

        self._model.train()
        last_loss = None
        for epoch in range(max_number_iterations):
            self._optimizer.zero_grad()
            pred = self._model(x)
            loss = self._loss(pred, y, self._model._linear.weight)
            loss.backward()
            self._optimizer.step()
            if last_loss is not None and abs(last_loss - loss.item()) < eps:
                break
            last_loss = loss.item()

        self._model.eval()
        with torch.no_grad():
            return self._model(x), self._model._linear.weight, self._model._linear.bias
