#weekly dataset - classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics

#------load_data----------------
def load_data():
    df = pd.read_csv('Weekly.csv')

    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    return X,y

#----------EDA----------------
def EDA(X):
    print('X shape:',X.shape)
    print('input data head:',X.head())
    print('X describe:',X.describe())
    print('X corr',X.corr())
    plt.figure(figsize=(10,8))
    sns.heatmap(X.corr(),annot=True,cmap='coolwarm',fmt='.2f')
    print('missing values in each col:',X.isnull().sum())
    plt.show()

#---------split_data------------
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
#---------preprocessing----------
def preprocessing(X_train,X_test):
    scaler = StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
#--------GBM---------------------
def GBM(X_train,y_train,X_test,y_test):
    gbm=GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    gbm.fit(X_train, y_train)
    y_pred=gbm.predict(X_test)
    print('R2 score:',gbm.score(X_test,y_test))
    print("predictions:",y_pred)
    print('accuracy:',metrics.accuracy_score(y_test,y_pred))
    print('gbm Score:', gbm.score(X_test, y_test))
    print('classification report:',classification_report(y_test, y_pred))
#--------main-----------------
def main():
    X,y = load_data()
    EDA(X)
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train,X_test=preprocessing(X_train,X_test)
    GBM(X_train,y_train,X_test,y_test)


if __name__ == "__main__":
    main()

