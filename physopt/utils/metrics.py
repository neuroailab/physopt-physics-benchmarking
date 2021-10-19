import os
import numpy as np
import csv
import logging
import pickle

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

def build_data(path, max_sequences = 1e9):
    with open(path, 'rb') as f:
        batched_data = pickle.load(f)

    # Unpack batched data into sequencs
    data = [] # each element in list is one example
    for bd in batched_data:
        for bidx in range(len(bd[list(bd.keys())[0]])):
            sequence = {}
            for k, v in bd.items():
                sequence[k] = v[bidx]
            data.append(sequence)
            if len(data) > max_sequences:
                break
        if len(data) > max_sequences:
            break

    logging.info('Data len: {}'.format(len(data)))
    logging.info('Input Shapes:')
    for k, v in data[0].items():
        try: 
            logging.info('{} {}'.format(k, v.shape))
        except Exception:
            pass
    return data

def observed_model_fn(data):
    states = np.concatenate([data['input_states'], data['observed_states']], axis=0)
    states = np.reshape(states, [states.shape[0], -1])
    return states

def simulated_model_fn(data):
    states = np.concatenate([data['input_states'], data['simulated_states']], axis=0)
    states = np.reshape(states, [states.shape[0], -1])
    return states

def input_model_fn(data):
    states = data['input_states']
    states = np.reshape(states, [states.shape[0], -1])
    return states

def get_feature_fn(protocol):
    if protocol == 'observed':
        return observed_model_fn
    elif protocol == 'simulated':
        return simulated_model_fn
    elif protocol == 'input':
        return input_model_fn
    else:
        raise NotImplementedError('Unknown feature function!')

def label_fn(data):
    labels = data['labels'] # use full sequence for labels
    labels = np.any(labels, axis=(0,1)).astype(np.int32).reshape([1])
    return labels

def get_num_samples(data, label_fn):
    pos = []
    neg = []
    for i, d in enumerate(data):
        if label_fn(d):
            pos.append(i)
        else:
            neg.append(i)
    return pos, neg

def oversample(data, pos, neg):
    raise NotImplementedError # Causes leakage during CV
    balanced_data = []
    if len(pos) < len(neg):
        balanced_data = [data[i] for i in neg]
        balanced_data.extend([data[i] for i in np.random.choice(pos, len(neg))])
    elif len(neg) < len(pos):
        balanced_data = [data[i] for i in pos]
        balanced_data.extend([data[i] for i in np.random.choice(neg, len(pos))])
    else:
        balanced_data = data
    return balanced_data

def undersample(data, pos, neg):
    balanced_data = []
    if len(pos) < len(neg):
        balanced_data = [data[i] for i in pos]
        balanced_data.extend([data[i] for i in np.random.choice(neg, len(pos), replace=False)])
    elif len(neg) < len(pos):
        balanced_data = [data[i] for i in neg]
        balanced_data.extend([data[i] for i in np.random.choice(pos, len(neg), replace=False)])
    else:
        balanced_data = data
    return balanced_data

def rebalance(data, label_fn, balance_fn=None):
    pos, neg = get_num_samples(data, label_fn)
    logging.info("Before rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

    if balance_fn is not None:
        data = balance_fn(data, pos, neg)
        pos, neg = get_num_samples(data, label_fn)
        logging.info("After rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))
    else:
        logging.info('Not rebalancing since balance_fn is None')
    return data

def write_metrics(results, metrics_file):
    file_exists = os.path.isfile(metrics_file) # check before opening file
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames = list(results[0].keys()))
        if not file_exists: # only write header once - if file doesn't exist yet
            writer.writeheader()
        writer.writerows(results)

    logging.info('%d results written to %s' % (len(results), metrics_file))

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
