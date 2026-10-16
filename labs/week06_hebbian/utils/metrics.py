import numpy as np

def recall_accuracy(target, recalled):
    return np.mean(target == recalled)