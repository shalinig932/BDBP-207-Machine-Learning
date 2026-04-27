#2.Implement XGBoost classifier and regressor using scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#----load_data-------
def load_data():
    df = pd.read_csv('Boston.csv')
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    return X,y

#------EDA------------
def EDA(X):
    print(X.shape)
    print(X.isnull().sum())
    print('missing values in each col:',X.isnull())

#------split_data-------------
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#------preprocessing----------
def preprocessing(X_train,X_test):
    scaler = StandardScaler()
    ft=scaler.fit_transform(X_train)
    tt=scaler.transform(X_test)
    return ft,tt

#-------XG_regressor-----------------
def XG(X_train,X_test,y_train,y_test):
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("R2 Score:", r2_score(y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


#------main---------------
def main():
    X,y = load_data()
    EDA(X)
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train,X_test=preprocessing(X_train,X_test)
    XG(X_train,X_test,y_train,y_test)
if __name__ == "__main__":
    main()