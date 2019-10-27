import logging

import aki.utils.reproducibility
import torch
import h5py

from pathlib import Path


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

    def _save_implementation(self, state_object: h5py.File):
        raise NotImplementedError

    def _load_implementation(self, state_object: h5py.File):
        raise NotImplementedError

    def _fit_implementation(self, *argparams, **kwparams):
        raise NotImplementedError

    def fit(self, *argparams, **kwparams):
        results = self._fit_implementation(*argparams, **kwparams)
        self._is_fit = True
        return results

    def predict(self, *argparams, **kwparams):
        if not self._is_fit:
            raise RuntimeError("Model has not been trained yet. Call fit and re-try.")
        return self._predict_implementation(*argparams, **kwparams)

    def save(self, model_filepath: Path):
        with h5py.File(str(model_filepath), 'w') as state_object:
            self._save_implementation(state_object)
            state_object.create_dataset('is_fit', data=self._is_fit)

    def load(self, model_filepath: Path):
        if not model_filepath.exists():
            raise FileNotFoundError(f"Model file {model_filepath} does not exist.")

        with h5py.File(str(model_filepath), 'r') as state_object:
            self._load_implementation(state_object)
            self._is_fit = state_object['is_fit'][()]
