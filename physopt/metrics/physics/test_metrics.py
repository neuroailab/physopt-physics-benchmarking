import dill
import os
import numpy as np
import scipy
import pickle
import logging
import dill
import joblib
import mlflow

from physopt.metrics.base.feature_extractor import FeatureExtractor
from physopt.metrics.base.readout_model import IdentityModel
from physopt.metrics.base.metric_model import BatchMetricModel

from physopt.metrics.physics.linear_readout_model import LinearRegressionReadoutModel, \
        LogisticRegressionReadoutModel

from physopt.metrics.physics.metric_fns import accuracy, squared_error

SETTINGS = [ # TODO: might not want this to be hardcoded, RPIN only takes 4 frames
        {
            'type': 'observed',
            'inp_time_steps': (0, 25, 1),
            'val_time_steps': (7, 25, 1),
            'model_fn': 'visual_scene_model_fn',
            },
        {
            'type': 'predicted',
            'inp_time_steps': (0, 25, 1),
            'val_time_steps': (7, 25, 1),
            'model_fn': 'rollout_scene_model_fn',
            },
        {
            'type': 'inferred',
            'inp_time_steps': (0, 7, 1),
            'val_time_steps': (7, 25, 1),
            'model_fn': 'visual_scene_model_fn',
            },
        ]

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


def build_model(model_fn, time_steps):
    def visual_scene_model_fn(data):
        predictions = subselect(data['encoded_states'], time_steps)
        predictions = np.reshape(predictions, [predictions.shape[0], -1])
        return predictions

    def rollout_scene_model_fn(data):
        predictions = subselect(data['rollout_states'], time_steps)
        predictions = np.reshape(predictions, [predictions.shape[0], -1])
        return predictions

    if model_fn == 'visual_scene_model_fn':
        return visual_scene_model_fn
    elif model_fn == 'rollout_scene_model_fn':
        return rollout_scene_model_fn
    else:
        raise NotImplementedError('Unknown model function!')

def select_label_fn(time_steps, experiment):
    def label_fn(data, time_steps = time_steps, experiment = experiment):
        labels = subselect(data['binary_labels'], time_steps)
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

def compute_per_example_results(model, file_path, time_steps):
    def reference_label_fn(data):
        labels = subselect(data['reference_ids'], time_steps)
        # Return file index
        assert np.all(labels[0:1,0] == labels[:, 0]), labels
        return labels[0,0].reshape([1])

    data = build_data(file_path)
    proba, file_reference_id = model.predict_proba(data, label_fn = reference_label_fn,
            return_labels = True)
    results = {
            'proba': proba,
            'ref_id': np.array(file_reference_id),
            }
    return results


def get_per_example_labels(label_fn, test_feature_file):
    # Get per example labels
    test_data = build_data(test_feature_file)
    labels = np.array([label_fn(d)[0] for d in test_data])
    return labels

def decode_references(references_path):
    assert os.path.isfile(references_path) and references_path.endswith('.txt'), references_path

    with open(references_path, 'r') as f:
        lines = f.read().splitlines()

    refs = {}
    for l in lines:
        idx, path = l.split('->')
        idx = int(idx.replace('.hdf5', ''))
        if idx not in refs:
            refs[idx] = path
        else:
            raise KeyError('Index already exists in references! %d: %s' % (idx, path))

    return refs


def reference2path(reference_ids, test_feat_name):
    reference_files = {
    'cloth': '/mnt/fs1/tdw_datasets/pilot-clothSagging-redyellow/tfrecords/references.txt',
    'collision': '/mnt/fs1/tdw_datasets/pilot-collision-redyellow/tfrecords/references.txt',
    'containment': '/mnt/fs1/tdw_datasets/pilot-containment-redyellow/tfrecords/references.txt',
    'dominoes': '/mnt/fs1/tdw_datasets/pilot-dominoes-redyellow/tfrecords/references.txt',
    'drop': '/mnt/fs1/tdw_datasets/pilot-drop-redyellow/tfrecords/references.txt',
    'linking': '/mnt/fs1/tdw_datasets/pilot-linking-redyellow/tfrecords/references.txt',
    'rollslide': '/mnt/fs1/tdw_datasets/pilot-rollingSliding-redyellow/tfrecords/references.txt',
    'towers': '/mnt/fs1/tdw_datasets/pilot-towers-redyellow/tfrecords/references.txt',
    }

    references = np.array([])
    for k in reference_files:
        if k in test_feat_name:
            reference_dict = decode_references(reference_files[k])
            references = np.array([reference_dict[idx[0]] for idx in reference_ids])
            break

    return references


def path2name(path):
    assert len(path) > 0, len(path)
    names = []
    for p in path:
        stimulus_set, file_name = p.split(os.sep)[-2:]
        names.append(stimulus_set + '_' + file_name[:4])
    return np.array(names)


def run(
        seed,
        train_feature_file,
        test_feature_file,
        test_feat_name,
        model_dir,
        settings,
        grid_search_params = {'C': np.logspace(-8, 8, 17)},
        calculate_correlation = False,
        ):
    # Build physics model
    feature_model = build_model(settings['model_fn'], slice(*settings['inp_time_steps']))
    feature_extractor = FeatureExtractor(feature_model)

    # Construct data providers
    logging.info('Train feature file: {}'.format(train_feature_file))
    logging.info('Test feature file: {}'.format(test_feature_file))
    train_data = build_data(train_feature_file)
    test_data = build_data(test_feature_file)

    # Select label function
    label_fn = select_label_fn(slice(*settings['val_time_steps']), test_feat_name)

    # Get stimulus names and labels for test data
    stimulus_names = [d['stimulus_name'] for d in test_data]
    labels = [label_fn(d)[0] for d in test_data]

    # Rebalance data
    np.random.seed(0)
    logging.info("Rebalancing training data")
    train_data_balanced = rebalance(train_data, label_fn)
    logging.info("Rebalancing testing data")
    test_data_balanced = rebalance(test_data, label_fn)

    # Build logistic regression model
    readout_model = LogisticRegressionReadoutModel(max_iter = 100, C=1.0, verbose=1)

    metric_model = BatchMetricModel(feature_extractor, readout_model,
            accuracy, label_fn, grid_search_params,
            )

    readout_model_file = os.path.join(os.path.dirname(train_feature_file), settings['type']+'_readout_model.joblib')
    if os.path.exists(readout_model_file):
        logging.info('Loading readout model from: {}'.format(readout_model_file))
        metric_model = joblib.load(readout_model_file)
    else:
        logging.info('Training readout model and saving to: {}'.format(readout_model_file))
        metric_model.fit(iter(train_data_balanced))
        joblib.dump(metric_model, readout_model_file)

    # TODO: make sure this works for tf models too
    train_acc = metric_model.score(iter(train_data_balanced))
    test_acc = metric_model.score(iter(test_data_balanced))
    test_proba = metric_model.predict_proba(iter(test_data))

    logging.info("Categorization train accuracy: %f" % train_acc)
    logging.info("Categorization test accuracy: %f" % test_acc)

    result = {'train_accuracy': train_acc, 'test_accuracy': test_acc, 'test_proba': test_proba, 'stimulus_name': stimulus_names, 'labels': labels}
    if grid_search_params is not None:
        result['best_params'] = metric_model._readout_model.best_params_

    return result

def write_results(
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
