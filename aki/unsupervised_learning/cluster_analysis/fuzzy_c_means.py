import torch

from torch.distributions import Dirichlet

from aki.utils.model import ModelBase


class FuzzyCMeans(ModelBase):
    """
    PyTorch implementation of the fuzzy c-means clustering algorithm.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)

    def _init_parameters(self, n_clusters: int, n_samples: int, n_features: int, fuzziness: float):
        self._n_clusters = n_clusters
        self._p = fuzziness
        self._w = Dirichlet(torch.ones(self._n_clusters)).sample([n_samples]).to(self._device)
        self._c = torch.zeros([n_clusters, n_features]).to(self._device)

    def _compute_centroids(self, x):
        fuzzy_w = self._w ** self._p
        self._c.data = (torch.mm(fuzzy_w.T, x) / torch.sum(fuzzy_w, dim=0).unsqueeze(dim=1))

    def _update_membership_matrix(self, x):
        exponent = 1.0 / (self._p - 1.0)
        distance = torch.norm(x.unsqueeze(dim=1).repeat(1, self._c.shape[0], 1) - self._c.unsqueeze(dim=0), 2,
                              dim=2) ** 2
        numerator = (1 / (distance ** exponent))
        self._w.data = numerator / torch.sum(numerator, dim=1).unsqueeze(dim=1)

    def fit(self, x: torch.Tensor, n_clusters: int, fuzziness: float = 2.0, max_iterations: int = 1000,
            eps: float = 5e-3):
        """
        Run the fuzzy c-means cluster analysis fit on the input data.
        :param x: input data as a Tensor with dimensions [batch, n_samples, n_features].
        :param n_clusters: number of clusters.
        :param fuzziness: fuzzy level [1.25 - 2.0].
        :param max_iterations: maximum number of iterations.
        :param eps: convergence error.
        :return: clusters centroids and membership matrix.
        """
        n_batches, n_samples, n_features = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape([n_batches * n_samples, n_features])
        self._init_parameters(n_clusters, n_samples * n_batches, n_features, fuzziness)

        iteration = 0
        fit_converged = False
        last_c = None
        while not fit_converged and iteration < max_iterations:
            # Compute the cluster centroids.
            self._compute_centroids(x)
            # Update the fuzzy partition.
            self._update_membership_matrix(x)
            # Check the termination condition.
            error = 0 if last_c is None else torch.dist(last_c, self._c)
            fit_converged = False if last_c is None else error < eps
            last_c = self._c.detach()
            self._logger.debug(f"Iteration {iteration}, mean error: {error}")
            iteration += 1

        self._is_fit = True
        return self._c, self._w.unsqueeze(dim=0).view([n_batches, n_samples, self._n_clusters])

    def _predict_implementation(self, x):
        n_batches, n_samples, n_features = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape([n_batches * n_samples, n_features])
        self._update_membership_matrix(x)
        return self._c, self._w.unsqueeze(dim=0).view([n_batches, n_samples, self._n_clusters])
