import time

import torch
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

from aki.supervised_learning.svm import SVM
from aki.utils.pyotrch import get_device

from sklearn.datasets.samples_generator import make_blobs

# creating datasets X containing n_samples
# Y containing two classes
X, y = make_blobs(n_samples=5, centers=2,
                  random_state=0, cluster_std=0.40)
print(y)
y[y == 0] = -1

clf = LinearSVC(random_state=0, tol=1e-5)

start = time.time()
clf.fit(X, y)
end = time.time() - start
print(f"scikit: {end}")

y_prob = clf.decision_function(X)
yp = clf.predict(X)

print(yp)

print(f"{np.sum(yp == y) / len(y) * 100} %")

aki_svm = SVM(get_device())

x_tensor = torch.from_numpy(X).to(get_device()).float()
y_tensor = torch.from_numpy(y).to(get_device()).float()

start = time.time()
prob, w, b = aki_svm.fit(x_tensor, y_tensor, n_classes=1, eps=1e-5, max_number_iterations=1000)
end = time.time() - start
print(f"aki: {end}")

prob = prob.detach().cpu().numpy().reshape(y.shape)
print(prob)

pred = (prob > 0.0).astype(np.int)
pred[pred == 0] = -1
print(f"{np.sum(pred == y) / len(y) * 100} %")

plt.scatter(X[:, 0], X[:, 1])

slope = (w[0] - w[1]) / b
y_vals = b + slope * X[:, 0]

plt.plot(X[:, 0], y_vals, c='r')
plt.show()
