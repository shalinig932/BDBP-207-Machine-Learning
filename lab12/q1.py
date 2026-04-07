#Implement a decision regression tree algorithm without using scikit-learn using the diabetes dataset.
# Fetch the dataset from scikit-learn library.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
def variance(y):
    if len(y) == 0:
        return 0
    mean = np.mean(y)
    return np.mean((y - mean) ** 2)
def variance_reduction(parent, left, right):
    if len(left) == 0 or len(right) == 0:
        return 0

    n = len(parent)

    var_parent = variance(parent)
    var_left = variance(left)
    var_right = variance(right)

    weighted = (len(left)/n)*var_left + (len(right)/n)*var_right

    return var_parent - weighted

def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gain = -1

    n_features = X.shape[1]

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])

        for t in thresholds:
            left_idx = X[:, feature] <= t
            right_idx = X[:, feature] > t

            gain = variance_reduction(y, y[left_idx], y[right_idx])

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=5):
    # Stop conditions
    if len(y) <= 1:
        return Node(value=np.mean(y))

    if depth >= max_depth:
        return Node(value=np.mean(y))

    feature, threshold = best_split(X, y)

    if feature is None:
        return Node(value=np.mean(y))

    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold

    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth)

    return Node(feature, threshold, left, right)

def predict_one(x, tree):
    if tree.value is not None:
        return tree.value

    if x[tree.feature] <= tree.threshold:
        return predict_one(x, tree.left)
    else:
        return predict_one(x, tree.right)


def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

# Train
tree = build_tree(X_train, y_train, max_depth=5)

# Predict
y_pred = predict(X_test, tree)

# Evaluation (MSE)
mse = np.mean((y_test - y_pred) ** 2)

print("Mean Squared Error:", mse)

