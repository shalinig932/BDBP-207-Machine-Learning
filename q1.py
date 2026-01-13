A = [
    [1, 2, 3],
    [4, 5, 6]
]

def transpose(A):
    rows = len(A)
    cols = len(A[0])

    # Create empty transpose matrix
    result = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]

    return result

A_T = transpose(A)
print("A^T:", A_T)

# def transpose(matrix):
#      rows = len(matrix)
#      cols = len(matrix[0])
#      # create new matrix with swapped dims
#      result = [[0 for i in range(rows)] for j in range(cols)]
#
#      for i in range(rows):
#          for j in range(cols):
#              result[j][i] = matrix[i][j]
#
#      return result
#
#
# A_T = transpose(matrix)
# print("A^T:", A_T)
#
# def multiply(x, y):
#      p= len(x)
#      q = len(x[0])
#      r= len(y[0])
#      result = [[0 for _ in range(r)] for _ in range(p)]
#      for i in range(p):
#          for j in range(q):
#           for k in range(r):
#            result[i][j] = x[i][k] * y[k][j]
#      return result
# ATA=multiply(matrix,A_T)
# print("ATA:", ATA)