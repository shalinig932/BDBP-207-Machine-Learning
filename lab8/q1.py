#Implement L2-norm and L1-norm from scratch
import math

import numpy as np
X=[3,-5,6]
def L1(X):
    total=0
    for i in range(len(X)):
        total += abs(X[i])
        return total

def L2(X):
    total=0
    for i in range(len(X)):
        total += abs(X[i])**2
        return math.sqrt(total)

def main():
    X=[3,-5,6]
    print('L1:',L1(X))
    print('L2:',L2(X))

if __name__ == '__main__':
    main()