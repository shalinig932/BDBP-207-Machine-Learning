#Implement entropy measure using Python.
# The function should accept a set of data points and their class labels and return the entropy value.
import math

def entropy(labels):

    total = len(labels)

    # count occurrences of each class
    class_counts = {}

    for label in labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # calculate entropy
    ent = 0

    for count in class_counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent

labels = ["Yes","Yes","No","No","Yes","No"]

print("Entropy:", entropy(labels))