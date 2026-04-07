#Build a classification model for wisconsin dataset using Ridge and Lasso classifier using scikit-learn

from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# L2 penalty (Ridge)
ridge_clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=1000)
ridge_clf.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_clf.predict(X_test_scaled)
print("Ridge Accuracy:", accuracy_score(y_test, y_pred_ridge))
print("Ridge Classification Report:\n", classification_report(y_test, y_pred_ridge))