import logging

import aki.utils.reproducibility
import torch


class ModelBase:
    """
    Abstract base class for the models.
    """

    def __init__(self, device: torch.device):
        self._logger = logging.getLogger('aki_logger')
        self._device = device
        self._logger.debug(f"Using device: {device}.")
        self._is_fit = False

    def _predict_implementation(self, *argparams, **kwparams):
        raise NotImplementedError

    def fit(self, *argparams, **kwparams):
        # Remember to set _is_fit in the actual implementation.
        raise NotImplementedError

    def predict(self, *argparams, **kwparams):
        if not self._is_fit:
            raise RuntimeError("Model has not been trained yet. Call fit and re-try.")
        self._predict_implementation(*argparams, **kwparams)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
