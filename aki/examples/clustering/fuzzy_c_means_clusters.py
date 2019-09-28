from __future__ import division, print_function

import time

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import torch

from clustering.fuzzy_cmeans.fcm_numpy import FCMNumpy
from clustering.fuzzy_cmeans.fcm_pytorch import FCMPytorch

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
n_points = 10000

# Define three cluster centers
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(n_points) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(n_points) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(n_points) * i))

alldata = np.vstack((xpts, ypts))

# Algorithms.

start = time.time()
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, 3, 2, error=5e-3, maxiter=1000, init=None)
end = time.time()
skfuzzy_time = round((end - start) * 1e3, 2)

numpy_fcm = FCMNumpy()
start = time.time()
c, mem, seg = numpy_fcm.fit(alldata.T, n_clusters=3, fuzziness=2.0, max_iterations=1000, eps=5e-3)
end = time.time()
numpy_time = round((end - start) * 1e3, 2)

pytorch_fcm = FCMPytorch().to('cuda:0')
tensor_data = torch.from_numpy(alldata.T).unsqueeze(dim=0).float().to('cuda:0')
start = time.time()
cpt, mempt, segpt = pytorch_fcm.fit(tensor_data, n_clusters=3, fuzziness=2.0, max_iterations=1000, eps=5e-3)
end = time.time()
pytorch_time = round((end - start) * 1e3, 2)

# Visualize the test data.
fig, ax = plt.subplots(1, 3)

for idx, label in enumerate(range(3)):
    ax[0].plot(xpts[labels == label], ypts[labels == label], '.', color=colors[label])
    ax[1].plot(xpts[labels == label], ypts[labels == label], '.', color=colors[label])
    ax[2].plot(xpts[labels == label], ypts[labels == label], '.', color=colors[label])

ax[0].plot(cntr[0, 0], cntr[0, 1], 'X', color='red')
ax[0].plot(cntr[1, 0], cntr[1, 1], 'X', color='red')
ax[0].plot(cntr[2, 0], cntr[2, 1], 'X', color='red')

ax[1].plot(c[0, 0], c[0, 1], 'X', color='red')
ax[1].plot(c[1, 0], c[1, 1], 'X', color='red')
ax[1].plot(c[2, 0], c[2, 1], 'X', color='red')

ax[2].plot(cpt[0, 0], cpt[0, 1], 'X', color='red')
ax[2].plot(cpt[1, 0], cpt[1, 1], 'X', color='red')
ax[2].plot(cpt[2, 0], cpt[2, 1], 'X', color='red')

ax[0].set_title(f'Test data: {n_points} points x3 clusters. [skfuzzy ~{skfuzzy_time} ms.]')
ax[1].set_title(f'Test data: {n_points} points x3 clusters. [numpy ~{numpy_time} ms.]')
ax[2].set_title(f'Test data: {n_points} points x3 clusters. [pytorch ~{pytorch_time} ms.]')

plt.show()

print(f'Test data: {n_points} points x3 clusters. [skfuzzy ~{skfuzzy_time} ms.]')
print(f'Test data: {n_points} points x3 clusters. [numpy ~{numpy_time} ms.]')
print(f'Test data: {n_points} points x3 clusters. [pytorch ~{pytorch_time} ms.]')
