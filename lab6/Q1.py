#K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.
import numpy as np
from sklearn.model_selection import StratifiedKFold

X = np.array([
    [1.2, 3.4],
    [2.1, 1.5],
    [0.5, 2.2],
    [3.1, 0.9],
    [2.2, 2.8]])

y = np.array([1, 0, 1, 0, 1])
def k_fold_split(X, y):
 kf = StratifiedKFold(n_splits=5, shuffle=True)
 return kf.split(X, y)
print(k_fold_split(X, y))
