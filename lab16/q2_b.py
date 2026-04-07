#weekly.csv - classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report
from sklearn import metrics
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#------load_data---------------------------------
def load_data():
    df=pd.read_csv('Weekly.csv')
    # Encode target
    df['Direction'] = df['Direction'].map({'Up': 1, 'Down': 0})
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    return X,y

#-----EDA--------------------------
def EDA(X):
    print(X.shape)
    print('missing values in col:',X.isnull().sum())
    print('corr:',X.corr())
    plt.figure(figsize=(5,5))
    sns.heatmap(X.corr(),annot=True,cmap='coolwarm')
    plt.show()

#--------split_data-----------------
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#-------preprocessing--------------
def preprocessing(X_train,X_test):
    scaler = StandardScaler()
    ft=scaler.fit_transform(X_train)
    tt=scaler.transform(X_test)
    return ft,tt

#------XG--------------------------
def XG(X_train,X_test,y_train,y_test):
    model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print('accuracy_score:',accuracy_score(y_test,y_pred))
    print('roc_auc_score:',roc_auc_score(y_test,y_prob))
    print('classification_report:',classification_report(y_test,y_pred))


#---------------main-----------------
def main():
    X,y=load_data()
    EDA(X)
    X_train,X_test,y_train,y_test = split_data(X,y)
    X_train,X_test= preprocessing(X_train,X_test)
    XG(X_train,X_test,y_train,y_test)
if __name__ == '__main__':
    main()








