#tree regression using scikit
#diabetes dataset, iris dataset
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# load dataset
def load_data():
    data = load_diabetes()
    X = data.data
    y = data.target
    return X, y


# train_test_split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# create a bagging regressor
def train_model(X_train, y_train):
    bagging_reg = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=100,
        random_state=42,
    )
    bagging_reg.fit(X_train, y_train)
    return bagging_reg


# prediction
def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


# evaluate model
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    y_pred = predict(model, X_test)

    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    main()




#--------------------------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#load data
def load_data():
 iris=load_iris()
 X = iris.data #.data takes the input as features
 y = iris.target #.target takes the target(y)
 return X, y


def split_data(X,y):
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 return X_train, X_test, y_train, y_test

def scaler(X_train,X_test):
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    return X_train,X_test


def bagging(X_train,y_train):
 bagging1 = BaggingClassifier(
     estimator=DecisionTreeClassifier(),
     n_estimators=100,
     random_state=42)
 bagging1.fit(X_train,y_train)
 return bagging1


def pred(bagging1,X_test):
    y_pred=bagging1.predict(X_test)
    return y_pred

def evaluate_model(y_test, y_pred):
    acc=accuracy_score(y_test, y_pred)
    return acc
def main():
    X,y=load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train,X_test=scaler(X_train,X_test)
    bagging2=bagging(X_train,y_train)
    y_pred=pred(bagging2,X_test)
    evaluate_model(y_test,y_pred)
    print("Accuracy", evaluate_model(y_test, y_pred))

if __name__ == "__main__":
    main()