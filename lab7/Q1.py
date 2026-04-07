#Perform 10-fold cross validation for SONAR dataset in scikit-learn using logistic regression.
# SONAR dataset is a binary classification problem with target variables as Metal or Rock. i.e. signals are from metal or rock.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

#load data
def load_data():
     df=pd.read_csv("/home/ibab/Machine_learning/SGML1/ML-lab1/lab7/sonar.all-data.csv",header=None)
     X=df.iloc[:,:-1]
     y=df.iloc[:,-1]
     return X,y
print(load_data())
# X,y=load_data()
# print(X.shape,y.shape)

def K_fold(k=5,shuffle='true'):
    X,y=load_data()
    n=len(X)
    indices=np.arange(n)
    if shuffle=='true':
        np.random.shuffle(indices)
    fold_size=n//k
    fold=[]
    for i in range(k):
        start=i*fold_size
        end=((i+1)*fold_size,n)
        test_X=X[indices[start:end]]
        train_X=np.concatenate((indices[:start],indices[end:]))
        fold.append([train_X,test_X])
    return fold
#print(K_fold(k=5))
def normalize(X):
 test_X,train_X=K_fold()
 cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Compute min & max on training fold
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1

        # Normalize
    X_train_norm = (X_train - X_min) / X_range
    X_test_norm = (X_test - X_min) / X_range

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_norm, y_train)

    y_pred = model.predict(X_test_norm)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print("Fold Accuracies:", accuracies)
print("Mean Accuracy:", np.mean(accuracies))






