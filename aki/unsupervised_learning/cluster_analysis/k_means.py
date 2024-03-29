import h5py
import torch

from pathlib import Path

from aki.utils.model import ModelBase


class KMeans(ModelBase):
    """
    PyTorch implementation of the k-means clustering algorithm.
    """

    def __init__(self, device: torch.device):
        super().__init__(device, "k-means")

    def _init_parameters(self, input_data: torch.Tensor, n_clusters: int):
        self._n_clusters = n_clusters
        # Use Forgy initialization method for the k centroids.
        indexes = torch.multinomial(torch.ones(input_data.T.shape), self._n_clusters, False).T.to(self._device)
        self._c = torch.gather(input_data, 0, indexes)

    def _expectation_step(self, x):
        repeated_input = x.unsqueeze(dim=1).repeat(1, self._c.shape[0], 1)
        self._distance_map = torch.norm(repeated_input - self._c.unsqueeze(dim=0), p=2, dim=2) ** 2

    def _maximization_step(self, x):
        _, membership = self._distance_map.min(axis=-1)
        for index in range(self._n_clusters):
            selected = torch.nonzero(membership == index).squeeze()
            if selected.nelement() != 0:
                selected = torch.index_select(x, 0, selected)
                self._c[index, :] = selected.mean(dim=0)

    def _predict_implementation(self, x):
        n_batches, n_samples, n_features = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape([n_batches * n_samples, n_features])
        self._expectation_step(x)
        return self._c, self._distance_map.view([n_batches, n_samples, self._n_clusters])

    def _save_implementation(self, state_object: h5py.File):
        state_object.create_dataset('centroids', data=self._c.detach().cpu().numpy())
        state_object.create_dataset('n_clusters', data=self._n_clusters)
        state_object.create_dataset('distance_map', data=self._distance_map.detach().cpu().numpy())

    def _load_implementation(self, state_object: h5py.File):
        self._c = torch.from_numpy(state_object['centroids'][()]).to(self._device)
        self._n_clusters = state_object['n_clusters'][()]
        self._distance_map = torch.from_numpy(state_object['distance_map'][()]).to(self._device)

    def _fit_implementation(self, x: torch.Tensor, n_clusters: int, max_iterations: int = 1000,
                            eps: float = 5e-3):
        """
        Run the k-means cluster analysis fit on the input data.
        :param x: input data as a Tensor with dimensions [batch, n_samples, n_features].
        :param n_clusters: number of clusters.
        :param max_iterations: maximum number of iterations.
        :param eps: convergence error.
        :return: clusters centroids and distance map.
        """
        n_batches, n_samples, n_features = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape([n_batches * n_samples, n_features])
        self._init_parameters(x, n_clusters)

        iteration = 0
        fit_converged = False
        last_c = None
        while not fit_converged and iteration < max_iterations:
            # Assign observations to the clusters.
            self._expectation_step(x)
            # Compute the cluster centroids.
            self._maximization_step(x)
            # Check the termination condition.
            error = 0 if last_c is None else torch.dist(last_c, self._c)
            fit_converged = False if last_c is None else error < eps
            last_c = self._c.detach().clone()
            self._logger.debug(f"Iteration {iteration}, mean error: {error}")
            iteration += 1

        return self._c, self._distance_map.view([n_batches, n_samples, n_clusters])
