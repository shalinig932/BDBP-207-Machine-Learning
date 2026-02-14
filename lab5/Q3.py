#3.)Compute the derivative of a sigmoid function and visualize it
import matplotlib.pyplot as plt
import numpy as np
z = []
for i in range(-100,101):
    z.append(i/10)
def sigmoid(z):
    sig_results = []
    for z in z:
        s = 1 / (1 + np.exp(-z))
        sig_results.append(s)
    return sig_results
print("sigmoid values are",sigmoid(z))

def sigmoid_derivative(z):
    H=np.array((sigmoid(z)))
    derive= H * (1-H)
    return derive
print("the values of derivative of sigmoid function are",sigmoid_derivative(z))

plt.plot(z)
plt.title("sigmoid function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()






