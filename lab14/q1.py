#Implement Adaboost classifier using scikit-learn. Use the Iris dataset.

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier # so for adaboost using scikit we use this model
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


#-----load-data------------------------
def load_data():
 df=pd.read_csv('Iris.csv')
 X=df.drop('Species',axis=1)
 y=df['Species']
 return X,y


#------EDA-----------------------------
def EDA(X):
    print('X describe:',X.describe())
    print('X null',X.isnull().sum())
    print('X shape:',X.shape)
    print('X.info:',X.info())


#--------train_test_split---------------
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


#----------preprocessing------------------
def preprocessing(y_train,y_test):
    le=LabelEncoder()
    y_train=le.fit(y_train)
    y_test=le.transform(y_test)
    return y_train,y_test

#----------adaboosting--------------------
def AdaBoost(X_train,y_train,X_test):
    model=AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), #estimator is a weak learner that adaboost trains repeatedly
        n_estimators=100,
        learning_rate=0.1,
        random_state=0
    )
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    return model,y_pred
#-------Accuracy_score----------------------
def accuracy(y_test,y_pred):
    print('Accuracy:',accuracy_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)
#-----------not needed from here---------------------------
#--------Gradient-boosting-----------------------

def GBM(X_train,y_train,X_test,y_test):
 gb = GradientBoostingClassifier()
 gb.fit(X_train, y_train)
 print("GB Accuracy:not needed for adaboosting(just for comparision)", gb.score(X_test, y_test))

#as we got accuracy score=1.0 for both adaboost and gradient boost classifier
#we will check it using cross_val_scores
#------cvs----------------------------------------
def cross_val(model,X,y):
    score=cross_val_score(model,X,y,cv=5)
    print('cvs:not needed for adaboosting, used here just to check if its overfitting',score)
#as the scores of cv are 1,0.83 so it's not overfitting
#conclusion:the dataset is easy
#------not needed till here------------------------
#-------main---------------------------------------
def main():
    X,y=load_data()
    EDA(X)
    X_train, X_test, y_train, y_test = split_data(X,y)
    preprocessing(y_train,y_test)
    model, y_pred=AdaBoost(X_train,y_train,X_test)
    accuracy(y_test,y_pred)
    GBM(X_train,y_train,X_test,y_test)
    cross_val(model,X,y)
if __name__=='__main__':
    main()

