class FeatureExtractor(object):
    """
    Extracts features from data
    """
    def __init__(self, model):
        self._model = model


    def __call__(self, data):
        features = self._model(data)
        return features



class BatchFeatureExtractor(FeatureExtractor):
    """
    Extracts features from data in a batchwise manner
    """
    def __init__(self, model, num_steps = 2**10000):
        super(BatchFeatureExtractor, self).__init__(model)
        self.num_steps = num_steps


    def __call__(self, data):
        features = []
        for _ in range(self.num_steps):
            try:
                batch = next(data)
                features.append(self._model(batch))
            except StopIteration:
                break
        return features
