 #Implement Adaboost classifier without using scikit-learn. Use the Iris dataset.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
data = load_iris()
X = data.data
y = data.target

# Convert to binary (-1, +1)
y = np.where(y == 0, 1, -1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for t in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X[:, feature] < t] = -1
                    else:
                        predictions[X[:, feature] > t] = -1

                    # Weighted error
                    error = np.sum(weights[y != predictions])

                    if error < min_error:
                        min_error = error
                        self.feature = feature
                        self.threshold = t
                        self.polarity = polarity

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X[:, self.feature] < self.threshold] = -1
        else:
            predictions[X[:, self.feature] > self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        # Initialize weights
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)

            predictions = stump.predict(X)

            # Compute error
            error = np.sum(weights[y != predictions])

            # Avoid division by zero
            error = max(error, 1e-10)

            # Compute alpha
            alpha = 0.5 * np.log((1 - error) / error)

            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            # Save model
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):
            final_pred += alpha * model.predict(X)

        return np.sign(final_pred)


# Train model
model = AdaBoost(n_estimators=10)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
