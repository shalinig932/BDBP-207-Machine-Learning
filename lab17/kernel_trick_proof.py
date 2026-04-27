#Implement a polynomial kernel K(a,b) =  a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2 .
# Apply this kernel function and evaluate the output for the same x1 and x2 values.
# Notice that the result is the same in both scenarios demonstrating the power of kernel trick.

import numpy as np

x1 = np.array([3, 6])
x2 = np.array([10, 10])

#  Polynomial kernel directly
def kernel(a, b):
    return (a[0]**2)*(b[0]**2) + 2*a[0]*b[0]*a[1]*b[1] + (a[1]**2)*(b[1]**2)

# Feature mapping (correct one)
def transform(x):
    return np.array([
        x[0]**2,
        np.sqrt(2)*x[0]*x[1],
        x[1]**2
    ])

# Compute kernel directly
k_val = kernel(x1, x2)

# Compute via higher-dimensional dot product
u = transform(x1)
v = transform(x2)
dot_val = np.dot(u, v)

print("Kernel value =", k_val)
print("Dot product in higher dimension =", dot_val)
