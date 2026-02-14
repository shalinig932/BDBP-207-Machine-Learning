#Implement sigmoid function in python and visualize it.
import numpy as np
z=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
print(sigmoid(z))