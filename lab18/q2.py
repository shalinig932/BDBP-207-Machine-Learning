import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_binary_iris():
    iris = load_iris()
    X = iris.data[:, :2]          # first 2 features
    y = iris.target

    # keep only classes 1 and 2
    mask = y > 0
    X = X[mask]
    y = y[mask]

    # convert labels from {1,2} to {0,1}
    y = y - 1
    return X, y


def stratified_manual_split(X, y, test_ratio=0.1, random_state=42):
    rng = np.random.RandomState(random_state)

    train_idx = []
    test_idx = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        n_test = max(1, int(round(len(cls_idx) * test_ratio)))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale"):
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["class 1", "class 2"]))

    return y_pred


def plot_decision_boundary(model, X_train, y_train, X_test, y_test, title):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.25)

    # training points
    plt.scatter(
        X_train[y_train == 0, 0], X_train[y_train == 0, 1],
        label="Train class 1", marker="o"
    )
    plt.scatter(
        X_train[y_train == 1, 0], X_train[y_train == 1, 1],
        label="Train class 2", marker="^"
    )

    # test points
    plt.scatter(
        X_test[y_test == 0, 0], X_test[y_test == 0, 1],
        label="Test class 1", marker="o", edgecolors="k", s=100
    )
    plt.scatter(
        X_test[y_test == 1, 0], X_test[y_test == 1, 1],
        label="Test class 2", marker="^", edgecolors="k", s=100
    )

    # support vectors
    plt.scatter(
        model.support_vectors_[:, 0], model.support_vectors_[:, 1],
        s=160, facecolors="none", edgecolors="black", label="Support vectors"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    X, y = load_binary_iris()

    X_train, X_test, y_train, y_test = stratified_manual_split(
        X, y, test_ratio=0.1, random_state=42
    )

    model = train_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale")

    evaluate_model(model, X_test, y_test)

    plot_decision_boundary(
        model, X_train, y_train, X_test, y_test,
        title="SVM on Iris Classes 1 and 2 (first 2 features)"
    )


if __name__ == "__main__":
    main()