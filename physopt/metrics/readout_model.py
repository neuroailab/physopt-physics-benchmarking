import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

class ReadoutModel(object):
    """
    Fits model on data to predict labels
    """
    def __init__(self, model=None):
        self._model = model

    def fit(self, features, labels, **kwargs):
        assert hasattr(self._model, "fit"), \
                ("model does not have 'fit' method!")
        self._model.fit(features, labels, **kwargs)

    def predict(self, features, **kwargs):
        assert hasattr(self._model, "predict"), \
                ("model does not have 'predict' method!")
        predictions = self._model.predict(features, **kwargs)
        return predictions

    def predict_proba(self, features, **kwargs):
        assert hasattr(self._model, "predict_proba"), \
                ("model does not have 'predict_proba' method!")
        return self._model.predict_proba(features, **kwargs)

class LogisticRegressionReadoutModel(ReadoutModel):
    def __init__(self, scaler=None, *args, **kwargs):
        self._model = LogisticRegression(*args, **kwargs) # TODO: pass in LogisticRegression as arg and use base ReadoutMdoel class
        self._scaler = scaler

    def fit(self, features, labels, **kwargs):
        if self._scaler:
            self._scaler.fit(features)
            features = self._scaler.transform(features)
        self._model.fit(features, labels)

    def predict(self, features, **kwargs):
        if self._scaler:
            features = self._scaler.transform(features)
        predictions = self._model.predict(features)
        return predictions
