#Compute SONAR classification results with and without data pre-processing (data normalization).
#Perform data pre-processing with your implementation and with scikit-learn methods and compare the results.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#-------------load_data-------------------
def load_data():
    df = pd.read_csv('sonar.all-data.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = y.map({'M':0,'R':1})
    return X, y

#---------logistic regression--------------
def log_reg(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('score(no processing):', score)
    return model

#---------manual normalization--------------
def manual_normalize(X_train, X_test):
    X_train_min = X_train.min()
    X_train_max = X_train.max()

    X_train_norm = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_test_norm = (X_test - X_train_min) / (X_train_max - X_train_min)

    return X_train_norm, X_test_norm

#---------scikit normalization--------------
def sklearn_normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

#---------main-------------------------------
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # No preprocessing
    model = log_reg(X_train, X_test, y_train, y_test)

    # Manual normalization
    X_train_norm, X_test_norm = manual_normalize(X_train, X_test)
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    print('score(manual processing):', accuracy_score(y_test, y_pred))

    # Scikit-learn normalization
    X_train_scaled, X_test_scaled = sklearn_normalize(X_train, X_test)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print('score(sklearn processing):', accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    main()
