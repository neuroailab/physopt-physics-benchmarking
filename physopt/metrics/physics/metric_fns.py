import numpy as np


def mse(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    return np.mean((predictions - labels) ** 2)


def squared_error(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    return (predictions - labels) ** 2


def particle_position_mse(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    valid = np.where(labels[:, :, :, 14] > 0)
    predictions = predictions[:, :, :, 0:3]
    labels = labels[:, :, :, 0:3]
    return mse(predictions[valid], labels[valid])


def particle_velocity_mse(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    valid = np.where(labels[:, :, :, 14] > 0)
    predictions = predictions[:, :, :, 4:7]
    labels = labels[:, :, :, 4:7]
    return mse(predictions[valid], labels[valid])


def object_position_mse(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    valid = np.where(labels[:, :, :, 0] > 0)
    predictions = predictions[:, :, :, 0:3]
    #TODO DOUBLE CHECK IF 5:8 POSITION!
    labels = labels[:, :, :, 5:8]
    return mse(predictions[valid], labels[valid])


def object_pose_mse(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    valid = np.where(labels[:, :, :, 0] > 0)
    predictions = predictions[:, :, :, 0:3]
    #TODO DOUBLE CHECK IF 1:5 POSE!
    labels = labels[:, :, :, 1:5]
    return mse(predictions[valid], labels[valid])


def object_velocity_mse(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    valid = np.where(labels[:, :, :, 0] > 0)
    predictions = predictions[:, :, :, 0:3]
    #TODO DOUBLE CHECK IF 14:17 VELOCITY!
    labels = labels[:, :, :, 14:17]
    return mse(predictions[valid], labels[valid])


def accuracy(predictions, labels):
    predictions = np.concatenate(predictions, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    return np.mean(predictions == labels)


def negative_accuracy(predictions, labels):
    return -accuracy(predictions, labels)
