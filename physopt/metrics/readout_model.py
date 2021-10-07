class ReadoutModel(object):
    """
    Fits model on data to predict labels
    """
    def __init__(self, model, scaler=None):
        self._model = model
        self._scaler = scaler

    def fit(self, features, labels, **kwargs):
        assert hasattr(self._model, "fit"), \
                ("model does not have 'fit' method!")
        if self._scaler:
            self._scaler.fit(features)
            features = self._scaler.transform(features)
        self._model.fit(features, labels, **kwargs)

    def predict(self, features, **kwargs):
        assert hasattr(self._model, "predict"), \
                ("model does not have 'predict' method!")
        if self._scaler:
            features = self._scaler.transform(features)
        predictions = self._model.predict(features, **kwargs)
        return predictions

    def predict_proba(self, features, **kwargs):
        assert hasattr(self._model, "predict_proba"), \
                ("model does not have 'predict_proba' method!")
        if self._scaler:
            features = self._scaler.transform(features)
        return self._model.predict_proba(features, **kwargs)
