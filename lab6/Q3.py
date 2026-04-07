#Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.

import numpy as np

X=[[1,2],[3,4],[5,6]]

def standardize(X):
    X=np.array(X, dtype=np.float64)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X = (X - mean) / std
    return X
print(standardize(X))