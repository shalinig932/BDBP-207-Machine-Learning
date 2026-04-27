#Use validation set to do feature and model selection.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
#load data
df=pd.read_csv("/home/ibab/Machine_learning/SGML1/ML-lab1/lab6/breast_cancer_dataset.csv")
X=df.drop("diagnosis",axis=1)
y=df["diagnosis"].map({'M':0 , 'B':1})

#split the data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)

# Split Temp into Validation (20%) and Test (20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
accuracy_score=y_val,y_test

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_val=scaler.transform(X_val)
X_test=scaler.transform(X_test)



model = LogisticRegression(max_iter=5000)

scores = cross_val_score(model, X, y, cv=10)

print("Validation Scores:", scores)
print("Average Validation Accuracy:", scores.mean())