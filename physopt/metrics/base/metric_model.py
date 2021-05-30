from physopt.metrics.base.feature_extractor import FeatureExtractor
from physopt.metrics.base.readout_model import ReadoutModel
import numpy as np

from sklearn.model_selection import GridSearchCV

class MetricModel(object):
    """
    Fits and evaluates readout_model on features to predict labels using metric_fn
    """
    def __init__(self,
            feature_extractor,
            readout_model,
            metric_fn):
        assert isinstance(feature_extractor, FeatureExtractor)

        self._feature_extractor = feature_extractor
        self._readout_model = readout_model
        self._metric_fn = metric_fn


    def fit(self,
            data,
            labels):
        features = self._feature_extractor(data)
        self._readout_model.fit(features, labels)


    def predict(self,
            data):
        features = self._feature_extractor(data)
        predictions = self._readout_model.predict(features)
        return predictions


    def predict_proba(self,
            data):
        features = self._feature_extractor(data)
        features = [np.reshape(feat, [-1]) if feat.ndim > 1 else feat for feat in features]
        proba = self._readout_model.predict_proba(features)
        return proba


    def score(self,
            data,
            labels):
        predictions = self.predict(features)
        metric = self._metric_fn(predictions, labels)
        return metric



class BatchMetricModel(MetricModel):
    def __init__(self,
            feature_extractor,
            readout_model,
            metric_fn,
            label_fn = lambda x: x,
            grid_search_params = None):

        if grid_search_params:
            readout_model.score = metric_fn
            readout_model = GridSearchCV(readout_model, grid_search_params)

        super(BatchMetricModel, self).__init__(feature_extractor, readout_model, metric_fn)
        self._label_fn = label_fn


    def extract_features_labels(self,
            data,
            num_steps = 2**10000,
            feature_extractor = None,
            label_fn = None):
        feature_extractor = self._feature_extractor if feature_extractor is None \
                else feature_extractor
        label_fn = self._label_fn if label_fn is None \
                else label_fn

        features = []
        labels = []
        for _ in range(num_steps):
            try:
                batch = next(data)
                features.append(feature_extractor(batch))
                labels.append(label_fn(batch))
            except StopIteration:
                break
        return features, labels


    def flatten_features_labels(self, features, labels):
        labels = np.array(labels)
        features = np.array(features)

        self.labels_shape = labels.shape
        self.features_shape = features.shape

        labels = np.reshape(labels, [-1])
        features = np.reshape(features, [labels.shape[0], -1])

        return features, labels


    def fit(self,
            data,
            num_steps = 2**10000):
        features, labels = self.extract_features_labels(data, num_steps)
        features, labels = self.flatten_features_labels(features, labels)
        self._readout_model.fit(features, labels)


    def predict(self,
            data,
            num_steps = 2**10000):
        features, labels = self.extract_features_labels(data, num_steps)
        features, _ = self.flatten_features_labels(features, labels)
        predictions = self._readout_model.predict(features)
        return predictions


    def predict_proba(self,
            data,
            num_steps = 2**10000,
            feature_extractor = None,
            label_fn = None,
            return_labels = False):
        features, labels = self.extract_features_labels(data, num_steps,
                feature_extractor, label_fn)
        features, _ = self.flatten_features_labels(features, labels)
        features = [np.reshape(feat, [-1]) if feat.ndim > 1 else feat for feat in features]
        proba = self._readout_model.predict_proba(features)
        if return_labels:
            return proba, labels
        else:
            return proba


    def score(self,
            data,
            num_steps = 2**10000):
        features, labels = self.extract_features_labels(data, num_steps)
        features, _ = self.flatten_features_labels(features, labels)
        predictions = self._readout_model.predict(features)
        if not isinstance(predictions, list):
            predictions = predictions.reshape(self.labels_shape)
        metric = self._metric_fn(predictions, labels)
        return metric
