#Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=--10, stop=10, num=100]
import matplotlib.pyplot as plt
start=-10
stop=10
num=70
step=(stop-start)/(num-1)
X1=[]
for i in range(num):
    X1.append(start+i*step)
y= [2* (X1**2) +3* X1+4 for X1 in X1]

plt.plot(X1,y)
plt.title("implementing y=2x1^2 + 3x1 + 4")
plt.xlabel("x1")
plt.ylabel("y")
plt.grid(True)
plt.show()