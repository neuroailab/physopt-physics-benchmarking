import os
import numpy as np
import scipy
import pickle
import logging
import joblib
import dill
import csv
import mlflow

from physopt.metrics.feature_extractor import FeatureExtractor
from physopt.metrics.readout_model import IdentityModel
from physopt.metrics.metric_model import BatchMetricModel
from physopt.metrics.linear_readout_model import LogisticRegressionReadoutModel
from physopt.metrics.metric_fns import accuracy 

def build_data(path, max_sequences = 1e9):
    with open(path, 'rb') as f:
        batched_data = pickle.load(f)

    # Unpack batched data into sequencs
    data = []
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


def subselect(data, time_steps):
    assert time_steps.stop <= len(data), (time_steps.stop, len(data))
    assert time_steps.start < len(data), (time_steps.start, len(data))
    return data[time_steps]


def build_model(protocol):
    if protocol == 'observed':
        def observed_model_fn(data):
            states = np.concatenate([data['input_states'], data['observed_states']], axis=0)
            states = np.reshape(states, [states.shape[0], -1])
            return states
        return observed_model_fn
    elif protocol == 'simulated':
        def simulated_model_fn(data):
            states = np.concatenate([data['input_states'], data['simulated_states']], axis=0)
            states = np.reshape(states, [states.shape[0], -1])
            return states
        return simulated_model_fn
    elif protocol == 'input':
        def input_model_fn(data):
            states = data['input_states']
            states = np.reshape(states, [states.shape[0], -1])
            return states
        return input_model_fn
    else:
        raise NotImplementedError('Unknown model function!')

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


def rebalance(data, label_fn, balancing = oversample):
    # Get number of positive and negative samples
    pos, neg = get_num_samples(data, label_fn)
    logging.info("Before rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

    # Rebalance data by oversampling underrepresented calls
    balanced_data = balancing(data, pos, neg)

    pos, neg = get_num_samples(balanced_data, label_fn)
    logging.info("After rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))
    return balanced_data

def run_metrics(
        seed,
        train_feature_file,
        test_feature_file,
        protocol,
        grid_search_params = {'C': np.logspace(-8, 8, 17)},
        ):
    # Build physics model
    feature_model = build_model(protocol)
    feature_extractor = FeatureExtractor(feature_model)

    # Construct data providers
    train_data = build_data(train_feature_file)
    test_data = build_data(test_feature_file)

    # Get stimulus names and labels for test data
    stimulus_names = [d['stimulus_name'] for d in test_data]
    labels = [label_fn(d)[0] for d in test_data]

    # Rebalance data
    np.random.seed(seed)
    logging.info("Rebalancing training data")
    train_data_balanced = rebalance(train_data, label_fn)
    logging.info("Rebalancing testing data")
    test_data_balanced = rebalance(test_data, label_fn)

    # Build logistic regression model
    readout_model = LogisticRegressionReadoutModel(max_iter=100, C=1.0, verbose=1)
    metric_model = BatchMetricModel(feature_extractor, readout_model, accuracy, label_fn, grid_search_params)

    # TODO: clean up this part
    readout_model_file = os.path.join(os.path.dirname(train_feature_file), protocol+'_readout_model.joblib')
    if os.path.exists(readout_model_file):
        logging.info('Loading readout model from: {}'.format(readout_model_file))
        metric_model = joblib.load(readout_model_file)
    else:
        logging.info('Training readout model and saving to: {}'.format(readout_model_file))
        metric_model.fit(iter(train_data_balanced))
        joblib.dump(metric_model, readout_model_file)
    mlflow.log_artifact(readout_model_file, artifact_path='readout_models')

    train_acc = metric_model.score(iter(train_data_balanced))
    test_acc = metric_model.score(iter(test_data_balanced))
    test_proba = metric_model.predict_proba(iter(test_data))

    result = {
        'train_accuracy': train_acc, 
        'test_accuracy': test_acc, 
        'test_proba': test_proba, 
        'stimulus_name': stimulus_names, 
        'labels': labels,
        'protocol': protocol,
        'seed': seed,
        }
    if grid_search_params is not None:
        result['best_params'] = metric_model._readout_model.best_params_

    return result

def write_metrics(results, metrics_file):
    file_exists = os.path.isfile(metrics_file) # check before opening file
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames = list(results[0].keys()))
        if not file_exists: # only write header once - if file doesn't exist yet
            writer.writeheader()
        writer.writerows(results)

    logging.info('%d results written to %s' % (len(results), metrics_file))
