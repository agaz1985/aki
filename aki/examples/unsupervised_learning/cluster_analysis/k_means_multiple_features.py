from __future__ import division, print_function

import logging
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import torch

from aki.unsupervised_learning.cluster_analysis.k_means import KMeans
from aki.utils.data import generate_normal_data
from aki.utils.pyotrch import get_device

logger = logging.getLogger('aki_logger')


def main(parameters):
    # Generate the example data.
    logger.info("Generate the input data...")
    n_points = parameters['n_points']
    x_points, y_points, labels = parameters['input_data']
    cluster_data = np.vstack([x_points, y_points])

    # Run the sci-kit k-means implementation.
    logger.info("Running sci-kit k-means implementation...")
    start = time.time()
    sk_centroids = cluster.KMeans(n_clusters=parameters['n_clusters'],
                                  max_iter=parameters['max_iterations'],
                                  tol=parameters['error'],
                                  init='random').fit(cluster_data.T).cluster_centers_
    end = time.time()
    sk_time = round((end - start) * 1e3, 2)

    # Run the fcm pytorch implementation.
    logger.info("Running PyTorch k-means implementation...")
    aki_fcm = KMeans(parameters['device'])
    tensor_data = torch.from_numpy(cluster_data.T).unsqueeze(dim=0).float().to(parameters['device'])
    start = time.time()
    aki_centroids, _ = aki_fcm.fit(tensor_data,
                                   n_clusters=parameters['n_clusters'],
                                   max_iterations=parameters['max_iterations'],
                                   eps=parameters['error'])
    end = time.time()
    pytorch_time = round((end - start) * 1e3, 2)
    aki_centroids = aki_centroids.detach().cpu().numpy()

    # Visualize the test data.
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

    for idx, label in enumerate(range(3)):
        ax[0].plot(x_points[labels == label], y_points[labels == label], '.', color=colors[label])
        ax[1].plot(x_points[labels == label], y_points[labels == label], '.', color=colors[label])

    ax[0].plot(sk_centroids[0, 0], sk_centroids[0, 1], 'X', color='red')
    ax[0].plot(sk_centroids[1, 0], sk_centroids[1, 1], 'X', color='red')
    ax[0].plot(sk_centroids[2, 0], sk_centroids[2, 1], 'X', color='red')

    ax[1].plot(aki_centroids[0, 0], aki_centroids[0, 1], 'X', color='red')
    ax[1].plot(aki_centroids[1, 0], aki_centroids[1, 1], 'X', color='red')
    ax[1].plot(aki_centroids[2, 0], aki_centroids[2, 1], 'X', color='red')

    ax[0].set_title(f"Test data: {n_points} points x {parameters['n_clusters']} clusters. [sklearn ~{sk_time} ms.]")
    ax[1].set_title(f"Test data: {n_points} points x {parameters['n_clusters']} clusters. [aki ~{pytorch_time} ms.]")

    plt.show()


if __name__ == "__main__":
    # Generate three clusters 2D data given means and standard deviations.
    n_samples = int(1e5)
    mean_list = [[4, 2],
                 [1, 7],
                 [5, 6]]
    std_list = [[0.8, 0.3],
                [0.3, 0.5],
                [1.1, 0.7]]

    # Define the example parameters.
    example_parameters = {'n_clusters': 3,
                          'max_iterations': 100,
                          'error': 5e-3,
                          'device': get_device(),
                          'input_data': generate_normal_data(n_samples, mean_list, std_list),
                          'n_points': n_samples}

    main(example_parameters)
