#Implement a regression decision tree algorithm using scikit-learn for the simulated dataset.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

#--------load_data---------------------
def load_data():
    df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    return X, Y

#--------split_data-----------------------
def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.2, random_state=42)

#-------bagging (Random Forest)---------
def bagging(X_train, X_test, Y_train, Y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    print("Random Forest R^2:", r2)
    print("Random Forest MSE:", mse)
    return r2, mse

#-----------main-------------------------
def main():
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    bagging(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
    main()