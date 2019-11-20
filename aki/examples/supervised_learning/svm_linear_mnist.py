import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

from aki.supervised_learning.svm import SVM
from aki.utils.pyotrch import get_device

from sklearn.datasets.samples_generator import make_blobs

# creating datasets X containing n_samples
# Y containing two classes
n_classes = 2
X, y = make_blobs(n_samples=10, centers=n_classes,
                  random_state=0, cluster_std=0.30)

clf = LinearSVC(random_state=0, tol=1e-5)

start = time.time()
clf.fit(X, y)
end = time.time() - start
print(f"scikit: {end}")

yp = clf.predict(X)

print(f"{np.sum(yp == y) / len(y) * 100} %")

aki_svm = SVM(get_device())

x_tensor = torch.from_numpy(X).to(get_device()).float()
y_tensor = torch.from_numpy(y).to(get_device()).float()

y_tensor = y_tensor.unsqueeze(dim=-1)

start = time.time()
prob, w, b = aki_svm.fit(x_tensor, y_tensor, n_classes=1, eps=1e-5, lr=3e-4, max_number_iterations=1000, C=1.0)
end = time.time() - start
print(f"aki: {end}")

prob = prob.detach().cpu().numpy()
w = w.detach().cpu().numpy().squeeze()
b = b.detach().cpu().numpy().squeeze()

prob = (prob > 0.0).astype(int).flatten()

print(f"{np.sum(prob == y) / len(y) * 100} %")

plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b')

y_vals = -b / w[1] - w[0] / w[1] * X[:, 0]
plt.plot(X[:, 0], y_vals, c='g')
plt.show()
