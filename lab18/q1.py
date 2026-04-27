#Consider the following dataset. Implement the RBF kernel.
# Check if RBF kernel separates the data well and compare it with the Polynomial Kernel.

import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC

# -----------------------------
# Data
# -----------------------------
X = np.array([[6, 5], [6, 9], [8, 6], [8, 8], [8, 10],[9, 2], [9, 5], [10, 10], [10, 13], [11, 5],
              [11, 8], [12, 6], [12, 11], [13, 4], [14, 8]])
y = np.array([-1, -1, 1, 1, 1,-1, 1, 1, -1, 1,1, 1, -1, -1, -1])


# -----------------------------
# Custom RBF kernel
# -----------------------------
def rbf_kernel(a, b, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(a - b) ** 2)


# -----------------------------
# Train models
# -----------------------------
rbf_model = SVC(kernel='rbf', gamma=0.1)
poly_model = SVC(kernel='poly', degree=2)

rbf_model.fit(X, y)
poly_model.fit(X, y)


# -----------------------------
# Plot decision boundary
# -----------------------------
def plot_decision(model, title):
    plt.figure(figsize=(6, 5))

    # mesh grid
    xx, yy = np.meshgrid(np.linspace(5, 15, 200),
                         np.linspace(0, 15, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)

    # plot points
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label='Blue')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Red', marker='^')

    plt.title(title)
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


# -----------------------------
# Compare
# -----------------------------
plot_decision(poly_model, "Polynomial Kernel (degree=2)")
plot_decision(rbf_model, "RBF Kernel")
