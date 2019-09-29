from __future__ import division, print_function

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import torch

from pathlib import Path

from aki.unsupervised_learning.cluster_analysis.fuzzy_c_means import FuzzyCMeans
from aki.utils.filesystem import load_image
from aki.utils.pyotrch import get_device

logger = logging.getLogger('aki_logger')


def main(parameters: dict):
    running_time = {'skfuzzy': [], 'aki': []}
    image_size = ['small', 'big']

    # Create the sub-plots.
    fig, ax = plt.subplots(2, 3, figsize=(15, 15))

    for image_index, image_filepath in enumerate(parameters['input_list']):
        # Load the image.
        logger.info("Loading the input image...")
        image = load_image(Path(__file__).parent / image_filepath)
        input_data = np.expand_dims(image.flatten(), axis=0)

        # Run the fcm sci-kit fuzzy implementation.
        logger.info("Running FCM sci-kit fuzzy implementation...")
        start = time.time()
        _, sk_membership, _, _, _, _, _ = fuzz.cluster.cmeans(input_data,
                                                              parameters['n_clusters'],
                                                              parameters['fuzziness'],
                                                              error=parameters['error'],
                                                              maxiter=parameters['max_iterations'],
                                                              init=None)
        end = time.time()
        running_time['skfuzzy'].append(round((end - start), 2))

        # Run the fcm pytorch implementation.
        logger.info("Running FCM PyTorch implementation...")
        aki_fcm = FuzzyCMeans(parameters['device'])
        input_data_tensor = torch.from_numpy(input_data.T).unsqueeze(dim=0).float().to(parameters['device'])
        start = time.time()
        _, aki_membership = aki_fcm.fit(input_data_tensor,
                                        n_clusters=parameters['n_clusters'],
                                        fuzziness=parameters['fuzziness'],
                                        max_iterations=parameters['max_iterations'],
                                        eps=parameters['error'])
        end = time.time()
        running_time['aki'].append(round((end - start), 2))

        for idx in range(3):
            ax[image_index][0].imshow(image, cmap='gray')
            ax[image_index][1].imshow(np.argmax(sk_membership, axis=0).reshape(image.shape))
            ax[image_index][2].imshow(np.argmax(aki_membership.detach().cpu().numpy(), axis=2).reshape(image.shape))

        ax[image_index][0].set_title(f"input image - {image_size[image_index]}")
        ax[image_index][1].set_title(
            f"{parameters['n_clusters']} classes clustered. [skfuzzy ~{running_time['skfuzzy'][image_index]} s.]")
        ax[image_index][2].set_title(
            f"{parameters['n_clusters']} classes clustered [aki ~{running_time['aki'][image_index]} s.]")

    plt.show()


if __name__ == "__main__":
    # Define the example parameters.
    example_parameters = {'n_clusters': 10,
                          'max_iterations': 100,
                          'fuzziness': 2.0,
                          'error': 1e-3,
                          'device': get_device(),
                          'input_list': ["../../data/blue_orange_small.jpg", "../../data/blue_orange_big.jpg"]}

    main(example_parameters)
