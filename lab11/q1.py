#decision tree classifier

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))

def information_gain(parent, left, right):
    if len(left) == 0 or len(right) == 0:
        return 0

    n = len(parent)
    H_parent = entropy(parent)

    H_left = entropy(left)
    H_right = entropy(right)

    weighted = (len(left)/n)*H_left + (len(right)/n)*H_right

    return H_parent - weighted

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

            gain = information_gain(y, y[left_idx], y[right_idx])

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
    if len(set(y)) == 1:
        return Node(value=y[0])

    if depth >= max_depth:
        return Node(value=most_common_label(y))

    feature, threshold = best_split(X, y)

    if feature is None:
        return Node(value=most_common_label(y))

    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold

    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth)

    return Node(feature, threshold, left, right)

def most_common_label(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]

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

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)