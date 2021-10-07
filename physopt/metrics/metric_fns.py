import numpy as np

def squared_error(predictions, labels):
    assert predictions.ndim == 1
    assert labels.ndim == 1
    return (predictions - labels) ** 2

def accuracy(predictions, labels):
    assert predictions.ndim == 1
    assert labels.ndim == 1
    return np.mean(predictions == labels)

def negative_accuracy(predictions, labels):
    assert predictions.ndim == 1
    assert labels.ndim == 1
    return -accuracy(predictions, labels)
