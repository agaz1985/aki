import unittest

import torch
import numpy as np

from pathlib import Path
from os import remove

from sklearn.datasets import make_blobs

from aki.supervised_learning.svm import SVM
from aki.utils.pyotrch import get_device


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self._device = get_device()
        self._model = SVM(self._device)
        x, y = make_blobs(n_samples=100, centers=1, random_state=0, cluster_std=0.30)
        self._x = torch.from_numpy(x).to(self._device).float()
        self._y = torch.from_numpy(y).to(self._device).float()
        self._y = self._y.unsqueeze(dim=-1)
        self._model_filename = Path("./checkpoint.h5")

    def test_name(self):
        self.assertEqual("svm", self._model.get_name())

    def test_save_load(self):
        # Fit and save a model.
        pred, w, b = self._model.fit(self._x, self._y, n_classes=1, eps=1e-5, lr=1e-3,
                                     max_number_iterations=1000, C=1.0)

        self._model.save(self._model_filename)
        self.assertTrue(self._model_filename.exists())

        # Create a new empty model.
        new_model = SVM(self._device)
        with self.assertRaises(RuntimeError) as _:
            new_model.predict(self._x)

        # Load the saved checkpoint.
        new_model.load(self._model_filename)

        # Run the inference and compare the results.
        pred_new, w_new, b_new = new_model.predict(self._x)

        self.assertTrue(np.all(np.equal(pred.detach().cpu().numpy(), pred_new.detach().cpu().numpy())))
        self.assertTrue(np.all(np.equal(w.detach().cpu().numpy(), w_new.detach().cpu().numpy())))
        self.assertTrue(np.all(np.equal(b.detach().cpu().numpy(), b_new.detach().cpu().numpy())))

        # Clean-up the test.
        remove(str(self._model_filename))
        self.assertFalse(self._model_filename.exists())


if __name__ == '__main__':
    unittest.main()
