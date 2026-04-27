import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    df = pd.read_csv("Heart.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


# -----------------------------
# Split Data
# -----------------------------
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# Train Model
# -----------------------------
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Predict Probabilities
# -----------------------------
def get_probabilities(model, X_test):
    return model.predict_proba(X_test)[:, 1]


# -----------------------------
# Compute Metrics
# -----------------------------
def compute_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

    return accuracy, precision, sensitivity, specificity, f1


# -----------------------------
# Evaluate at different thresholds
# -----------------------------
def evaluate_thresholds(y_true, y_prob, thresholds):
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        acc, prec, sens, spec, f1 = compute_metrics(y_true, y_pred)

        print(f"\nThreshold = {t}")
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Sensitivity:", sens)
        print("Specificity:", spec)
        print("F1-score:", f1)


# -----------------------------
# Plot ROC Curve
# -----------------------------
def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.show()

    print("AUC:", roc_auc)


# -----------------------------
# Main Function
# -----------------------------
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    y_prob = get_probabilities(model, X_test)

    thresholds = [0.3, 0.5, 0.7]
    evaluate_thresholds(y_test.values, y_prob, thresholds)

    plot_roc(y_test, y_prob)


if __name__ == "__main__":
    main()