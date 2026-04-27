import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import seaborn as snp
import matplotlib.pyplot as plt


def load_data():
    dataset = load_iris()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    return X, y


def EDA(X):
    print(X.info())
    print(X.head())
    print(X.describe().T)
    print(X.isnull().sum())
    print("missing value % is:", X.isnull().sum() * 100 / len(X))

    plt.figure(figsize=(8, 6))
    snp.heatmap(X.corr(), annot=True, fmt='.2f')
    plt.show()


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocessing(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # ✅ FIXED
    X_test = scaler.transform(X_test)
    return X_train, X_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


def main():
    X, y = load_data()

    EDA(X)

    # Correct order
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test = preprocessing(X_train, X_test)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    main()