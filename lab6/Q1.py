#K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

accuracies = []

# --------------------------------------------------
def load_data():
    df = pd.read_csv('/home/ibab/Machine_learning/SGML1/ML-lab1/lab6/breast_cancer_dataset.csv')
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis'].map({'M': 0, 'B': 1})
    return X, y

# --------------------------------------------------
def k_fold_split(X, y, k=10, shuffle=True):
    n = len(X)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = min((i + 1) * fold_size, n)

        test_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))

        folds.append((train_idx, test_idx))

    return folds

# --------------------------------------------------
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# --------------------------------------------------

# Load dataset
X, y = load_data()

# Create folds
folds = k_fold_split(X, y, k=10)

# Perform 10-fold cross validation
for train_idx, test_idx in folds:

    # Split data
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Train model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Print results
print("Accuracy for each fold:", accuracies)
print("Average Accuracy:", np.mean(accuracies))








