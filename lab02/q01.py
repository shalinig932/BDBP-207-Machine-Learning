


import numpy as np

# Convert X to NumPy array
X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])

# Column-wise mean (mean centering)
mean = np.mean(X, axis=0)
print("Mean:", mean)
#calculating mean, axis =0 refers to direction along which the calculation is done.
# Mean-centered data
x_centered = X - mean
#here its along column.ex:1+0+2+1+0/5=0.8
 #here mean 0.8 is subtracted from the column then
                    #1.0 from column 2
                    #1.0 from column 3
# Number of samples
n = X.shape[0]

# Manual covariance matrix computation
cov_matrix_manual = (x_centered.T @ x_centered) / (n - 1)
print("Manual Covariance Matrix:\n", cov_matrix_manual)

# NumPy covariance matrix
cov_matrix_numpy = np.cov(X, rowvar=False)
print("NumPy Covariance Matrix:\n", cov_matrix_numpy)


# def mean_center(X, mean):
#     n = len(X)
#     m = len(X[0])
#     X_centered = []
#
#     for i in range(n):
#         row = []
#         for j in range(m):
#             row.append(X[i][j] - mean[j])
#         X_centered.append(row)
#
#     return X_centered
# mean = X_centered(X)
# X_centered = mean_center(X, mean)
#
# for row in X_centered:
#     print(row)