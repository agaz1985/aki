import unittest

import torch
import numpy as np

from pathlib import Path
from os import remove

from aki.unsupervised_learning.cluster_analysis.fuzzy_c_means import FuzzyCMeans
from aki.utils.data import generate_normal_data
from aki.utils.pyotrch import get_device


class TestFuzzyCMeans(unittest.TestCase):
    def setUp(self):
        self._device = get_device()
        self._model = FuzzyCMeans(self._device)
        data = generate_normal_data(n_points=100,
                                    mean_list=[[4, 2], [1, 7]],
                                    std_list=[[0.8, 0.3], [0.3, 0.5]])
        self._data = torch.from_numpy(np.vstack(data).T).unsqueeze(dim=0).float().to(self._device)
        self._model_filename = Path("./checkpoint.h5")

    def test_save_load(self):
        # Fit and save a model.
        centroids, membership = self._model.fit(self._data,
                                                n_clusters=2,
                                                fuzziness=0.2,
                                                max_iterations=100,
                                                eps=1e-5)

        self._model.save(self._model_filename)
        self.assertTrue(self._model_filename.exists())

        # Create a new empty model.
        new_model = FuzzyCMeans(self._device)
        with self.assertRaises(RuntimeError) as _:
            new_model.predict(self._data)

        # Load the saved checkpoint.
        new_model.load(self._model_filename)

        # Run the inference and compare the results.
        centroids_new, membership_new = new_model.predict(self._data)

        self.assertTrue(np.all(np.equal(centroids.detach().cpu().numpy(), centroids_new.detach().cpu().numpy())))
        self.assertTrue(np.all(np.equal(membership.detach().cpu().numpy(), membership_new.detach().cpu().numpy())))

        # Clean-up the test.
        remove(str(self._model_filename))
        self.assertFalse(self._model_filename.exists())


if __name__ == '__main__':
    unittest.main()
