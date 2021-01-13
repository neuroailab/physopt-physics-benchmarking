import numpy as np



def squared_error(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    return (predictions - labels) ** 2


def accuracy(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    return np.mean(predictions == labels)
