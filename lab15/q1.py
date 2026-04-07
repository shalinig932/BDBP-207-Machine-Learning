#implement Gradient boosting with scikit.Implement Gradient Boost Regression and
# Classification using scikit-learn. Use the Boston housing dataset from the ISLP package
# for the regression problem and weekly dataset from the ISLP package and use Direction as
# the target variable for the classification.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

#-------load_data-----------------------
def load_data():
    df=pd.read_csv('Boston.csv')
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    return X,Y
#---------EDA----------------------------
def EDA(X):
    print('X head:',X.head())
    print('X info:',X.info())
    X.drop(X.columns[X.columns.str.contains('^Unnamed')], axis=1, inplace=True)#as unnamed 0 is not useful so we can just drop it.
    print('X info:',X.info())
    print('X corr:',X.corr())
    plt.figure(figsize=(10,8))
    sns.heatmap(X.corr(), annot=True, fmt='.2f')
    plt.show()

#-------split_data-------------------------
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


#------preprocessing----------------------
def preprocessing(X_train, X_test):
 scaler = StandardScaler()
 ft=scaler.fit_transform(X_train)
 tt=scaler.transform(X_test)
 return ft,tt

#------GB-----------------------------------
def GBM(X_train,y_train,X_test,y_test):
    gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    print('R2 score:',gbm.score(X_test,y_test))
    print("predictions:",y_pred)
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    print('gbm Score:',gbm.score(X_test, y_test))


#--------------------------------------------
def main():
    X,Y = load_data()
    EDA(X)
    X_train, X_test, y_train, y_test = split_data(X,Y)
    X_train,X_test=preprocessing(X_train, X_test)
    GBM(X_train,y_train,X_test,y_test)
if __name__ == '__main__':
    main()
