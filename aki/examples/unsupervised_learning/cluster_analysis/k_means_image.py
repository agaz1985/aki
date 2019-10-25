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
    image_size = ['small', 'big']

    # Create the sub-plots.
    fig, ax = plt.subplots(2, 3, figsize=(15, 15))

    for image_index, image_filepath in enumerate(parameters['input_list']):
        # Load the image.
        logger.info("Loading the input image...")
        image = load_image(Path(__file__).parent / image_filepath)
        input_data = np.expand_dims(image.flatten(), axis=0).T

        # Run the kmeans sci-kit implementation.
        logger.info("Running k-means sci-kit implementation...")
        start = time.time()
        sk_labels = cluster.KMeans(n_clusters=parameters['n_clusters'],
                                   max_iter=parameters['max_iterations'],
                                   tol=parameters['error'],
                                   init='random').fit(input_data).labels_
        end = time.time()
        running_time['scikit'].append(round((end - start), 2))

        # Run the fcm PyTorch implementation.
        logger.info("Running k-means PyTorch implementation...")
        aki_km = KMeans(parameters['device'])
        input_data_tensor = torch.from_numpy(input_data).unsqueeze(dim=0).float().to(parameters['device'])
        start = time.time()
        _, aki_distance_map = aki_km.fit(input_data_tensor,
                                          n_clusters=parameters['n_clusters'],
                                          max_iterations=parameters['max_iterations'],
                                          eps=parameters['error'])
        end = time.time()
        running_time['aki'].append(round((end - start), 2))

        for idx in range(3):
            ax[image_index][0].imshow(image, cmap='gray')
            ax[image_index][1].imshow(sk_labels.reshape(image.shape))
            ax[image_index][2].imshow(np.argmin(aki_distance_map.detach().cpu().numpy(), axis=2).reshape(image.shape))

        ax[image_index][0].set_title(f"input image - {image_size[image_index]}")
        ax[image_index][1].set_title(
            f"{parameters['n_clusters']} classes clustered. [scikit ~{running_time['scikit'][image_index]} s.]")
        ax[image_index][2].set_title(
            f"{parameters['n_clusters']} classes clustered [aki ~{running_time['aki'][image_index]} s.]")

    plt.show()


if __name__ == "__main__":
    # Define the example parameters.
    example_parameters = {'n_clusters': 10,
                          'max_iterations': 100,
                          'error': 1e-3,
                          'device': get_device(),
                          'input_list': ["../../data/blue_orange_small.jpg", "../../data/blue_orange_big.jpg"]}

    main(example_parameters)
