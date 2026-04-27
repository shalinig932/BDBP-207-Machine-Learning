#Let x1 = [3, 6], x2 = [10, 10].  Use the above “Transform” function to transform these vectors to a higher dimension
# and  compute the dot product in a higher dimension. Print the value.

import numpy as np

x1=np.array([3,6])
x2=np.array([10,10])

def transform(x):
    return np.array([x[0],x[1],x[0]*x[1]])
u=transform(x1)
v=transform(x2)
print(u)
print(v)


