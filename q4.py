#Implement Gaussian PDF - mean = 0, sigma = 15 in the range[start=-100, stop=100, num=100]
import math
import matplotlib.pyplot as plt
mean=0
sigma=15
#range given as
start=-100
stop=100
num=100
step=(start-stop)/(num-1)
x_values=[]
for i in range(num):
    x_values.append(start+i*step)
y_values=[]
for x in x_values:
 coefficient=1/(sigma*math.sqrt(2*math.pi))
 exponent=math.exp(-((x-mean)**2)/(2*sigma**2))
 y_values.append(coefficient*exponent)
print(x_values,y_values)
plt.plot(x_values,y_values)
plt.title("Gaussian Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()