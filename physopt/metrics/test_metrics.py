import os
import numpy as np
import scipy
import pickle
import logging
import joblib
import dill

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


def build_model(protocol, time_steps):
    def visual_scene_model_fn(data):
        predictions = subselect(data['encoded_states'], time_steps)
        predictions = np.reshape(predictions, [predictions.shape[0], -1])
        return predictions

    def rollout_scene_model_fn(data):
        predictions = subselect(data['rollout_states'], time_steps)
        predictions = np.reshape(predictions, [predictions.shape[0], -1])
        return predictions

    if protocol in {'observed', 'input'}:
        return visual_scene_model_fn
    elif protocol in {'simulated'}:
        return rollout_scene_model_fn
    else:
        raise NotImplementedError('Unknown model function!')

def select_label_fn():
    def label_fn(data):
        labels = data['binary_labels'] # use full sequence for labels
        labels = np.any(labels, axis=(0,1)).astype(np.int32).reshape([1])
        return labels
    return label_fn

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
        settings,
        grid_search_params = {'C': np.logspace(-8, 8, 17)},
        ):
    # Build physics model
    feature_model = build_model(settings['protocol'], slice(*settings['inp_time_steps']))
    feature_extractor = FeatureExtractor(feature_model)

    # Construct data providers
    train_data = build_data(train_feature_file)
    test_data = build_data(test_feature_file)

    # Select label function
    label_fn = select_label_fn()

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
    readout_model_file = os.path.join(os.path.dirname(train_feature_file), settings['protocol']+'_readout_model.joblib')
    if os.path.exists(readout_model_file):
        logging.info('Loading readout model from: {}'.format(readout_model_file))
        metric_model = joblib.load(readout_model_file)
    else:
        logging.info('Training readout model and saving to: {}'.format(readout_model_file))
        metric_model.fit(iter(train_data_balanced))
        joblib.dump(metric_model, readout_model_file)

    train_acc = metric_model.score(iter(train_data_balanced))
    test_acc = metric_model.score(iter(test_data_balanced))
    test_proba = metric_model.predict_proba(iter(test_data))

    result = {
        'train_accuracy': train_acc, 
        'test_accuracy': test_acc, 
        'test_proba': test_proba, 
        'stimulus_name': stimulus_names, 
        'labels': labels
        }
    if grid_search_params is not None:
        result['best_params'] = metric_model._readout_model.best_params_

    return result

def write_metrics(
        metrics_file,
        seed,
        train_name,
        train_feature_file,
        test_feature_file,
        model_dir,
        results,
        ):
    data = {
            'seed': seed,
            'train_name': train_name,
            'train_feature_file': train_feature_file,
            'test_feature_file': test_feature_file,
            'model_dir': model_dir,
            'results': results,
            }
    with open(metrics_file, 'wb') as f:
        pickle.dump(data, f)
    print('Metrics results written to %s' % metrics_file)
    return

if __name__ == '__main__':
    seed = 0
    train_data = {'name': 'collision',
            'data': ['/mnt/fs4/hsiaoyut/tdw_physics/data/collision/tfrecords/train']
            }
    feat_data = (
        {'name': 'train_collision',
            'data': ['/mnt/fs4/hsiaoyut/tdw_physics/data/collision/tfrecords/train_readout']
            },
        {'name': 'test_collision',
            'data': ['/mnt/fs4/hsiaoyut/tdw_physics/data/collision/tfrecords/valid_readout']
            }
        )
    output_dir = '/mnt/fs1/mrowca/dummy1/SVG/'
    exp_key = '0_collision_test_collision_metrics'
    objective = Objective(exp_key, seed, train_data, feat_data, output_dir, False)
    objective()
