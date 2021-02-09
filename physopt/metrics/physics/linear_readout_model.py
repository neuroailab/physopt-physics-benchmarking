import numpy as np
from physopt.metrics.base.readout_model import ReadoutModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


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



class MultiOutputLinearRegressionReadoutModel(LinearRegressionReadoutModel):
    def __init__(self, *args, **kwargs):
        super(MultiOutputLinearRegressionReadoutModel, self).__init__(*args, **kwargs)
        self._model = MultiOutputClassifier(self._model)



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


class MultiOutputLogisticRegressionReadoutModel(LogisticRegressionReadoutModel):
    def __init__(self, *args, **kwargs):
        super(MultiOutputLogisticRegressionReadoutModel, self).__init__(*args, **kwargs)
        self._model = MultiOutputClassifier(self._model)
