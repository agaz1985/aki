import logging

import aki.utils.reproducibility


class ModelBase:
    """
    Abstract base class for the models.
    """

    def __init__(self):
        self._logger = logging.getLogger('aki_logger')

    def fit(self, *argparams, **kwparams):
        raise NotImplementedError

    def predict(self, *argparams, **kwparams):
        raise NotImplementedError
