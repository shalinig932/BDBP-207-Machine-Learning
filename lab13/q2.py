#Implement bagging regressor without using scikit-learn

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

class SimpleTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # stopping condition
        if depth >= self.max_depth or len(y) <= 2:
            return np.mean(y)

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return np.mean(y)

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        left = self._build_tree(X[left_idx], y[left_idx], depth+1)
        right = self._build_tree(X[right_idx], y[right_idx], depth+1)

        return (best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                gain = self._variance_reduction(y, left, right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold

    def _variance(self, y):
        return np.var(y)

    def _variance_reduction(self, parent, left, right):
        n = len(parent)
        return np.var(parent) - (
            len(left)/n * np.var(left) + len(right)/n * np.var(right)
        )

    def predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left, right = tree

        if x[feature] <= threshold:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])
class BaggingRegressor:
    def __init__(self, n_estimators=10, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def _bootstrap_sample(self, X, y):
        n = len(X)
        indices = np.random.choice(n, n, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.models = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            model = SimpleTree(max_depth=self.max_depth)
            model.fit(X_sample, y_sample)

            self.models.append(model)

    def predict(self, X):
        predictions = []

        for model in self.models:
            predictions.append(model.predict(X))

        predictions = np.array(predictions)

        # Average predictions
        return np.mean(predictions, axis=0)

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Bagging Regressor
model = BaggingRegressor(n_estimators=10, max_depth=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate (MSE)
mse = np.mean((y_test - y_pred) ** 2)

print("MSE:", mse)

