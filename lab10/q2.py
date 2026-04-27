#Implement information gain measures.
# The function should accept data points for parents, data points for both children and return an information gain value

import numpy as np

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    ent = 0
    for p in probs:
        ent -= p * np.log2(p)
    return ent

def information_gain(parent, left_child, right_child):
    n = len(parent)
    n_left = len(left_child)
    n_right = len(right_child)
    # Entropy calculations
    H_parent = entropy(parent)
    H_left = entropy(left_child)
    H_right = entropy(right_child)
    # Weighted child entropy
    weighted_entropy = (n_left / n) * H_left + (n_right / n) * H_right
    # Information Gain
    IG = H_parent - weighted_entropy
    return IG

parent = np.array([0, 0, 1, 1, 1, 0])
left_child = np.array([0, 0, 1])
right_child = np.array([1, 1, 0])

ig = information_gain(parent, left_child, right_child)
print("Information Gain:", ig)