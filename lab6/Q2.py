#Data normalization - scale the values between 0 and 1. Implement code from scratch.

X=[[1,2],
   [3,4],
   [5,6],
   [7,8]]
import numpy as np


def min_max_normalize(X):
    X = np.array(X, dtype=float)

    # Compute min and max for each column (feature)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    # Avoid division by zero
    denominator = X_max - X_min
    denominator[denominator == 0] = 1

    # Apply formula
    X_norm = (X - X_min) / denominator

    return X_norm
print(min_max_normalize(X))