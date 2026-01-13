#5.Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].
# Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
# What is the value of x1 at which the function value (y) is zero. What do you infer from this?
import math
import matplotlib.pyplot as plt
start=-10
stop=10
num=100
step=(start-stop)/(num-1)
# x1=[]
# for i in range(num):
#     x1.append(start+1*step)
# y=[]
# for x in range(num):
#     y.append(x1**2 for x1 in x1)
x1=[ start+i*step for i in range(num) ]
y= [x**2 for x in x1]
z=[-5,-3,0,3,5]
for i in range(z):
 y0=2*i
 slope=2*y0


# for x in [-5,-3,0,3,5]:
#  derivative= 2*x
#  print(x,derivative)
plt.plot(x1,y)
plt.title("x1^2")
plt.grid(True)
plt.show()
