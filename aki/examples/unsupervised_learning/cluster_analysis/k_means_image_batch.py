from __future__ import division, print_function

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from pathlib import Path

from sklearn import cluster

from aki.unsupervised_learning.cluster_analysis.k_means import KMeans
from aki.utils.filesystem import load_image
from aki.utils.pyotrch import get_device

logger = logging.getLogger('aki_logger')


def main(parameters: dict):
    running_time = {'scikit': [], 'aki': []}

    # Create the sub-plots.
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    image_filepath = parameters['input_image']

    # Load the image.
    logger.info("Loading the input image...")
    image = load_image(Path(__file__).parent / image_filepath)
    input_data = np.expand_dims(image.flatten(), axis=0)

    # Create a batch using the same image.
    input_batch = np.stack([input_data] * parameters['batch_size'], axis=0)

    # Run the k-means sci-kit implementation.
    logger.info("Running k-means sci-kit implementation...")
    sk_input_data = input_batch.reshape([1, parameters['batch_size'] * np.prod(input_data.shape)])
    start = time.time()
    sk_labels = cluster.KMeans(n_clusters=parameters['n_clusters'],
                               max_iter=parameters['max_iterations'],
                               tol=parameters['error'],
                               init='random').fit(sk_input_data.T).labels_
    end = time.time()
    running_time['scikit'].append(round((end - start), 2))

    # Run the k-means PyTorch implementation.
    logger.info("Running k-means PyTorch implementation...")
    aki_km = KMeans(parameters['device'])
    input_data_tensor = torch.from_numpy(input_batch.swapaxes(1, 2)).float().to(parameters['device'])
    start = time.time()
    _, aki_distance_map = aki_km.fit(input_data_tensor,
                                     n_clusters=parameters['n_clusters'],
                                     max_iterations=parameters['max_iterations'],
                                     eps=parameters['error'])
    end = time.time()
    running_time['aki'].append(round((end - start), 2))

    output_shape = [parameters['batch_size'], image.shape[0], image.shape[1]]
    for idx in range(3):
        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(sk_labels.reshape(output_shape)[0, :, :])
        ax[2].imshow(np.argmin(aki_distance_map.detach().cpu().numpy(), axis=2).reshape(output_shape)[0, :, :])

    ax[0].set_title(f"input image with batch size {parameters['batch_size']}")
    ax[1].set_title(
        f"{parameters['n_clusters']} classes clustered. [scikit ~{running_time['scikit'][0]} s.]")
    ax[2].set_title(
        f"{parameters['n_clusters']} classes clustered [aki ~{running_time['aki'][0]} s.]")

    plt.show()


if __name__ == "__main__":
    # Define the example parameters.
    example_parameters = {'n_clusters': 10,
                          'max_iterations': 100,
                          'error': 1e-3,
                          'device': get_device(),
                          'batch_size': 5,
                          'input_image': "../../data/blue_orange_small.jpg"}

    main(example_parameters)
