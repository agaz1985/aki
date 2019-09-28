import numpy as np
import torch

from torch.distributions import Dirichlet

from aki.utils.model import ModelBase


class FuzzyCMeans(ModelBase):
    def __init__(self, device: torch.device):
        super().__init__()
        self._device = device
        self._logger.debug(f"Using device: {device}.")

    def _init(self, n_clusters: int, input_shape: tuple, fuzziness: float):
        self._n_clusters = n_clusters
        self._p = fuzziness
        self._w = Dirichlet(torch.ones(self._n_clusters)).sample([input_shape[0]]).to(self._device)
        self._c = torch.zeros([n_clusters, input_shape[-1]]).to(self._device)
        self._last_c = None
        self._fit_converged = False

    @staticmethod
    def _preprocess_input(x: torch.Tensor):
        return x.squeeze(dim=0)

    def _compute_centroids(self, x):
        fuzzy_w = self._w ** self._p
        self._c.data = (torch.mm(fuzzy_w.T, x) / torch.sum(fuzzy_w, dim=0).unsqueeze(dim=1))

    def _update_membership_matrix(self, x):
        exponent = 1.0 / (self._p - 1.0)
        dist = 1 / torch.pow(
            torch.norm(x.unsqueeze(dim=1).repeat(1, self._c.shape[0], 1) - self._c.unsqueeze(dim=0), 2, dim=2), 2)
        num = torch.pow(dist, exponent)
        den = torch.sum(num, dim=1)
        self._w.data = num / torch.unsqueeze(den, dim=1)

    def fit(self, x: torch.Tensor, n_clusters: int, fuzziness: float = 2.0, max_iterations: int = 1000,
            eps: float = 5e-3):
        x = self._preprocess_input(x)
        self._init(n_clusters, x.shape, fuzziness)

        iteration = 0
        while not self._fit_converged and iteration < max_iterations:
            # Compute the cluster centroids.
            self._compute_centroids(x)
            # Update the fuzzy partition.
            self._update_membership_matrix(x)
            # Check the termination condition
            error = 0 if self._last_c is None else np.abs(self._last_c - self._c.detach().cpu().numpy())
            self._fit_converged = False if self._last_c is None else np.all(error < eps)
            self._last_c = self._c.detach().cpu().numpy()
            self._logger.debug(f"Iteration {iteration}, mean error: {np.mean(error)}")
            iteration += 1

        return self._c.detach().cpu().numpy(), self._w.detach().cpu().numpy(), np.argmax(self._w.detach().cpu().numpy(),
                                                                                         axis=1)

    def predict(self, x):
        x = self._preprocess_input(x)
        self._update_membership_matrix(x)
        return self._c.detach(), self._w.detach(), np.argmax(self._w.detach(), axis=1)
