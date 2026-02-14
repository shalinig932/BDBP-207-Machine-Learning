#Implement logistic regression using scikit-learn for the breast cancer dataset -
#sigmoid function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data():
 df = pd.read_csv('/home/ibab/Machine_learning/SGML1/ML-lab1/lab5/breast_cancer_dataset.csv')
 df = df.dropna() # Remove rows with any missing values
 x = df.drop('diagnosis', axis=1)
 y = df['diagnosis'].map({'M': 0, 'B': 1})
 return x, y

def split_data(X, y):
    # Added stratify=y to maintain balance between M and B classes
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# def train_model(X_train, X_test, y_train, y_test):


def main():
    # 1. Pipeline
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # 2. Train
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # 3. Evaluate
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()




