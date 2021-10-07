import numpy as np
from sklearn.model_selection import GridSearchCV
from physopt.metrics.feature_extractor import FeatureExtractor

class MetricModel(object):
    def __init__(self,
            feature_extractor,
            readout_model,
            metric_fn,
            label_fn = lambda x: x,
            grid_search_params = None):

        if grid_search_params:
            readout_model.score = metric_fn
            readout_model = GridSearchCV(readout_model, grid_search_params)

        assert isinstance(feature_extractor, FeatureExtractor)
        self._feature_extractor = feature_extractor
        self._readout_model = readout_model
        self._metric_fn = metric_fn
        self._label_fn = label_fn

    def _extract_features_labels(self, data): # data is list of dicts with the states/labels
        features = np.array(list(map(self._feature_extractor, data)))
        labels = np.array(list(map(self._label_fn, data)))
        print(f'extract features labels {features.shape} {labels.shape}')
        return features, labels

    def _flatten_features_labels(self, features, labels):
        assert labels.size == labels.flatten().size, 'Labels should be scalar'
        labels = labels.flatten()
        features = np.reshape(features, [labels.size, -1])
        print(f'flatten features labels {features.shape} {labels.shape}')
        return features, labels

    def get_features_labels(self, data):
        features, labels = self._extract_features_labels(data)
        features, labels = self._flatten_features_labels(features, labels)
        return features, labels

    def fit(self, data):
        features, labels = self.get_features_labels(data)
        self._readout_model.fit(features, labels)

    def predict(self, data):
        features, _ = self.get_features_labels(data)
        predictions = self._readout_model.predict(features)
        return predictions

    def predict_proba(self, data): # TODO: combine with predict?
        features, _ = self.get_features_labels(data)
        proba = self._readout_model.predict_proba(features)
        return proba

    def score(self, data):
        features, labels = self.get_features_labels(data)
        predictions = self._readout_model.predict(features)
        metric = self._metric_fn(predictions, labels)
        return metric
