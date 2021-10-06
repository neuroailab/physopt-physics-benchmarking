import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

class ReadoutModel(object):
    """
    Fits model on data to predict labels
    """
    def __init__(self, model=None):
        self._model = model


    def fit(self, data, labels, **kwargs):
        assert hasattr(self._model, "fit"), \
                ("model does not have 'fit' method!")
        self._model.fit(data, labels, **kwargs)


    def predict(self, data, **kwargs):
        assert hasattr(self._model, "predict"), \
                ("model does not have 'predict' method!")
        return self._model.predict(data, **kwargs)


    def predict_proba(self, data, **kwargs):
        assert hasattr(self._model, "predict_proba"), \
                ("model does not have 'predict_proba' method!")
        return self._model.predict_proba(data, **kwargs)


    def get_params(self, *args, **kwargs):
        return self._model.get_params(*args, **kwargs)


    def set_params(self, *args, **kwargs):
        return self._model.set_params(*args, **kwargs)



class IdentityModel(ReadoutModel):
    """
    Predicts labels to be identical to data
    """
    def fit(self, data, labels, **kwargs):
        pass

    def predict(self, data, **kwargs):
        return data

    def predict_proba(self, data, **kwargs):
        return data

class LinearRegressionReadoutModel(ReadoutModel):
    def __init__(self, *args, **kwargs):
        super(ReadoutModel, self).__init__()
        self._model = LinearRegression(*args, **kwargs)
        self.feat_dim = None


    def fit(self, data, labels, **kwargs):
        data = np.stack(data) if isinstance(data, list) else data
        labels = np.stack(labels) if isinstance(labels, list) else labels

        labels = np.reshape(labels, [-1])
        model_data = np.reshape(data, [labels.shape[0], -1])
        self.feat_dim = model_data.shape[-1]

        self._model.fit(model_data, labels)


    def predict(self, data, **kwargs):
        data = np.stack(data) if isinstance(data, list) else data
        model_data = np.reshape(data, [-1, self.feat_dim])

        predictions = self._model.predict(model_data)

        predictions = np.split(predictions, predictions.shape[0], axis = 0)

        return predictions

class LogisticRegressionReadoutModel(ReadoutModel):
    def __init__(self, scaler=None, *args, **kwargs):
        self._model = LogisticRegression(*args, **kwargs)
        self._scaler = scaler


    def fit(self, data, labels, **kwargs):
        data = np.stack(data) if isinstance(data, list) else data
        labels = np.stack(labels) if isinstance(labels, list) else labels

        model_data = np.reshape(data, [data.shape[0], -1])
        labels = np.reshape(labels, [-1])

        if self._scaler:
            self._scaler.fit(model_data)
            model_data = self._scaler.transform(model_data)

        self._model.fit(model_data, labels)


    def predict(self, data, **kwargs):
        data = np.stack(data) if isinstance(data, list) else data

        model_data = np.reshape(data, [data.shape[0], -1])

        if self._scaler:
            model_data = self._scaler.transform(model_data)

        predictions = self._model.predict(model_data)

        predictions = np.split(predictions, predictions.shape[0], axis = 0)

        return predictions
