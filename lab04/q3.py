#Implement normal equations method from scratch and compare your results
# on a simulated dataset (disease score fluctuation as target) and the admissions
# dataset (https://www.kaggle.com/code/erkanhatipoglu/linear-regression-using-the-normal-equation ).
# You can compare the results with scikit-learn and your own gradient descent implementation.



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


filename = "/home/Machine_learning/SGML1/lab04/simulated_data_multiple_linear_regression_for_ML.csv"
target_col = "disease_score_fluct"

# Load Data
df = pd.read_csv(filename)
X = df.drop(target_col, axis=1).values
y = df[target_col].values.reshape(-1, 1)

# Add bias column (for Normal Eq)

ones = np.ones((X.shape[0], 1))
X_bias = np.hstack((ones, X))


# 1.Normal equation

Xt = X_bias.T
theta_normal = np.linalg.inv(Xt.dot(X_bias)).dot(Xt).dot(y)
y_pred_normal = X_bias.dot(theta_normal)

ss_res = np.sum((y - y_pred_normal) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_normal = 1 - ss_res / ss_tot


# 2.Gradient descent

# Normalize features
X_gd = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_gd = np.hstack((np.ones((X_gd.shape[0], 1)), X_gd))

theta = np.zeros((X_gd.shape[1], 1))
alpha = 0.01
epochs = 1000
m = len(y)

for i in range(epochs):
    y_pred = X_gd.dot(theta)
    gradient = (1/m) * X_gd.T.dot(y_pred - y)
    theta = theta - alpha * gradient

theta_gd = theta
y_pred_gd = X_gd.dot(theta_gd)

ss_res = np.sum((y - y_pred_gd) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2_gd = 1 - ss_res / ss_tot


# 3. SCIKIT-LEARN

model = LinearRegression()
model.fit(X, y)
y_pred_sklearn = model.predict(X)
r2_sklearn = r2_score(y, y_pred_sklearn)


# RESULTS

print("R2 (Normal Equation):   ", r2_normal)
print("R2 (Gradient Descent):  ", r2_gd)
print("R2 (scikit-learn):      ", r2_sklearn)

print("Theta (Normal Equation):")
print(theta_normal)

print("Theta (Gradient Descent):")
print(theta_gd)

print("Coefficients (sklearn):")
print("Intercept:", model.intercept_)
print("Weights:", model.coef_)
# X_T = X.T
# # print(X_T)
# #this above gives transpose of X
#
# X_inv=np.linalg.inv(X.T @ X)
# # print(X_inv)
# #this gives the product of transpose of X and X and inversing it using inv.
#
# X_product=X_inv @ X_T
# # print(X_product)
# #this gives the product of X inverse and transpose of X
#
# theta = X_product @ Y
# print(theta)
# #this gives the final product that is theta


