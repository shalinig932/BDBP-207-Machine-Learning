#1.)implement a linear regression model using scikit-learn for
# the simulated dataset-simulated_data_multiple_linear_regression_for_ML.csv
# to predict the "disease_score" from multiple clinical parameters.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    data=pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X=data.drop("disease_score",axis=1)
    y=data["disease_score"]
    # print(X.head())
    # print(X.info())
    return X,y


def split_data(X, y):
        return train_test_split(X, y, test_size=0.3, random_state=99)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
    # print(X_train.shape())
    # print(X_test.shape())
    # print(y_train.shape())
    # print(y_test.shape())

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def main():
    X,y=load_data()
    X_train,X_test,y_train,y_test=split_data(X,y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    mse,r2 = test_model(model,X_test_scaled, y_test)
    print("mse:",mse)
    print("r2:",r2)
if __name__=="__main__":
    main()