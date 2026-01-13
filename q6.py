#Implement y = 2x1 + 3x2 + 3x3 + 4,
# where x1, x2 and x3 are three independent variables.
# Compute the gradient of y at a few points and print the values.
def y(x1,x2,x3):
    return 2*x1+3*x2+3*x3+4
q=y(2,3,4)
print(q)
def gradient():
    return[2,3,4]
p=gradient()
print(p)
points = [
    (0, 0, 0),
    (1, 2, 3),
    (-1, 4, 0),
    (2, 2, 2)
]

for x1, x2, x3 in points:
    value = y(x1, x2, x3)
    grad = gradient()
    print(f"At point ({x1}, {x2}, {x3}): y = {value}, gradient = {grad}")
