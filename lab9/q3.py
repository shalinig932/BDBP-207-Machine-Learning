#Implement a classification decision tree algorithm using scikit-learn for the sonar  dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#-------load_data----------------------------------
def load_data():
    df = pd.read_csv('sonar.all-data.csv')
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    # Convert labels to numeric
    Y = Y.map({'M': 0, 'R': 1})
    return X, Y
#--------split_data--------------------------------
def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.2, random_state=42)

#--------classifier--------------------------------
def RFC(X_train, X_test, Y_train, Y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(Y_test, y_pred)
    print("Accuracy:", acc_score)
    return acc_score

#----------main------------------------------------
def main():
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    RFC(X_train, X_test, Y_train, Y_test)

if __name__ == '__main__':
    main()