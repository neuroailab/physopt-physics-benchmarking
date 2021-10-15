import numpy as np

class MetricModel(object):
    def __init__(self,
            readout_model,
            feature_fn,
            label_fn,
            metric_fn,
            ):
        check_readout_model(readout_model)
        self._readout_model = readout_model
        self._feature_fn = feature_fn
        self._label_fn = label_fn
        self._metric_fn = metric_fn

    def _extract_features_labels(self, data): # data is list of dicts with the states/labels
        features = np.array(list(map(self._feature_fn, data)))
        labels = np.array(list(map(self._label_fn, data)))
        return features, labels

    def _flatten_features_labels(self, features, labels):
        assert labels.size == labels.flatten().size, 'Labels should be scalar'
        labels = labels.flatten()
        features = np.reshape(features, [labels.size, -1])
        return features, labels

    def get_features_labels(self, data):
        features, labels = self._extract_features_labels(data)
        features, labels = self._flatten_features_labels(features, labels)
        return features, labels

    def fit(self, data):
        features, labels = self.get_features_labels(data)
        self._readout_model.fit(features, labels)

    def predict(self, data, proba=False):
        features, _ = self.get_features_labels(data)
        if proba:
            probabilities = self._readout_model.predict_proba(features)
            return probabilities
        else:
            predictions = self._readout_model.predict(features)
            return predictions

    def score(self, data):
        features, labels = self.get_features_labels(data)
        predictions = self._readout_model.predict(features)
        metric = self._metric_fn(predictions, labels)
        return metric

def check_readout_model(readout_model):
    for attr in ['fit', 'predict', 'predict_proba']:
        assert hasattr(readout_model, attr) and callable(getattr(readout_model, attr)), f'Readout model must have {attr} method'

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
