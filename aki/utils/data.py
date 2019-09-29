import numpy as np


def generate_normal_data(n_points: int, mean_list: list, std_list: list):
    x_points = np.zeros(1)
    y_points = np.zeros(1)
    labels = np.zeros(1)
    for i, ((x_mean, y_mean), (x_std, y_std)) in enumerate(zip(mean_list, std_list)):
        x_points = np.hstack((x_points, np.random.standard_normal(n_points) * x_std + x_mean))
        y_points = np.hstack((y_points, np.random.standard_normal(n_points) * y_std + y_mean))
        labels = np.hstack((labels, np.ones(n_points) * i))
    return x_points, y_points, labels
