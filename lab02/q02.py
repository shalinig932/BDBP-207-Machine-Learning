def dot_product(a,b):
    result=0
    for i in range(len(a)):
        result+=a[i]*b[i]
    return result

a=[2,1,2]
b=[1,2,2]
print(dot_product(a,b))
#The dot product (also called scalar product) of
# two vectors is a single number obtained by
# multiplying corresponding elements and adding
# them.