class FeatureExtractor(object):
    """
    Extracts features from data
    """
    def __init__(self, model):
        self._model = model


    def __call__(self, data):
        features = self._model(data)
        return features
