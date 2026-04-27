#2)Implement y = 2x1 + 3 and plot x1, y [start=-100, stop=100, num=100]

import matplotlib.pyplot as plt
start=-100
stop=100
num=100
step=(start-stop)/(num - 1)
X1=[start+i * step for i in range(num)]
y=[2* X1+3 for X1 in X1]
print(X1)
print(y)

plt.plot(X1,y)
plt.title("implementing y=2x1+3")
plt.xlabel('X1')
plt.ylabel('Y')
plt.grid(True)
plt.show()
