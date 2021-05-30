import os
import numpy as np
import scipy
import pickle

from physopt.metrics.base.feature_extractor import FeatureExtractor
from physopt.metrics.base.readout_model import IdentityModel
from physopt.metrics.base.metric_model import BatchMetricModel

from physopt.metrics.physics.linear_readout_model import LinearRegressionReadoutModel, \
        LogisticRegressionReadoutModel

from physopt.metrics.physics.metric_fns import accuracy, squared_error
from physopt.utils import PhysOptObjective

SETTINGS = [ # TODO: might not want this to be hardcoded
        {
            'inp_time_steps': (0, 24, 1),
            'val_time_steps': (4, 24, 1),
            'model_fn': 'visual_scene_model_fn',
            },
        {
            'inp_time_steps': (0, 24, 1),
            'val_time_steps': (4, 24, 1),
            'model_fn': 'rollout_scene_model_fn',
            },
        {
            'inp_time_steps': (0, 4, 1),
            'val_time_steps': (4, 24, 1),
            'model_fn': 'visual_scene_model_fn',
            },
        {
            'inp_time_steps': (0, 4, 1),
            'val_time_steps': (0, 4, 1),
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

    return iter(data)


def subselect(data, time_steps):
    assert time_steps.stop <= len(data), (time_steps.stop, len(data))
    assert time_steps.start < len(data), (time_steps.start, len(data))
    return data[time_steps]


def build_model(model_fn, time_steps):
    def visual_object_model_fn(data):
        #predictions = subselect(data['encoded_before_relu_states'], time_steps)
        predictions = subselect(data['encoded_states'], time_steps)
        return predictions

    def rollout_object_model_fn(data):
        predictions = subselect(data['rollout_states'], time_steps)
        return predictions

    def visual_scene_model_fn(data):
        #predictions = subselect(data['encoded_before_relu_states'], time_steps)
        predictions = subselect(data['encoded_states'], time_steps)
        predictions = np.reshape(predictions, [predictions.shape[0], -1])
        return predictions

    def rollout_scene_model_fn(data):
        predictions = subselect(data['rollout_states'], time_steps)
        predictions = np.reshape(predictions, [predictions.shape[0], -1])
        return predictions

    if model_fn == 'visual_object_model_fn':
        return visual_object_model_fn
    elif model_fn == 'rollout_object_model_fn':
        return rollout_object_model_fn
    elif model_fn == 'visual_scene_model_fn':
        return visual_scene_model_fn
    elif model_fn == 'rollout_scene_model_fn':
        return rollout_scene_model_fn
    else:
        raise NotImplementedError('Unknown model function!')


def collision_label_fn(data, time_steps):
    labels = subselect(data['binary_labels'], time_steps)
    labels = np.any(labels, axis=(0,1)).astype(np.int32).reshape([1])
    return labels


def stable_label_fn(data, time_steps):
    labels = subselect(data['binary_labels'], time_steps)
    labels = np.all(labels, axis=(0,1)).astype(np.int32).reshape([1])
    return labels


def rolling_label_fn(data, time_steps):
    labels = subselect(data['binary_labels'], time_steps)
    labels = np.any(labels, axis=(0,1)).astype(np.int32).reshape([1])
    return labels


def contain_label_fn(data, time_steps, dist_thres = 0.55, pos_frame_ratio_thres = 0.75):
    # Compute distance between objects
    # TODO Subselect?
    pos = data['binary_labels'][:, :, 5:8]
    dist = np.sqrt(np.sum((pos[:, 0] - pos[:, 1]) ** 2, axis = -1))
    # If objects close to each other in frame one object is containing the other
    objects_are_close = dist < dist_thres
    # If more than pos_sequence_ratio frames objects are close then they sequence is containing
    num_frames = objects_are_close.shape[-1]
    num_close_frames = np.sum(objects_are_close, axis = -1)
    labels = num_close_frames / num_frames > pos_frame_ratio_thres
    return labels.astype(np.int32).reshape([1])


def object_category_label_fn(data, time_steps, object_idx=1):
    labels = subselect(data['binary_labels'], time_steps)
    assert np.all(labels[0:1, :] == labels), ("Object category changed within sequence!", labels)
    labels = labels[0, object_idx]
    return labels.astype(np.int32).reshape([1])


def select_label_fn(time_steps, experiment):
    def label_fn(data, time_steps = time_steps, experiment = experiment):
        if 'test' in experiment:
            return collision_label_fn(data, time_steps)
        elif 'collide' in experiment:
            return collision_label_fn(data, time_steps)
        elif 'tower' in experiment:
            return stable_label_fn(data, time_steps)
        elif 'contain' in experiment:
            return contain_label_fn(data, time_steps)
        elif 'cloth' in experiment:
            return object_category_label_fn(data, time_steps)
        elif 'roll_vs_slide' in experiment: # TODO: should this be roll_slide
            return rolling_label_fn(data, time_steps)
        else:
            raise NotImplementedError(experiment)
    return label_fn


def pos_label_fn(data, time_steps, dim = 2):
    #TODO Write out positions
    #pos = data['bboxes']
    dummy_pos = np.random.rand(*(data['binary_labels'].shape[0], 2))
    pos = dummy_pos
    assert pos.shape[-1] == dim, (pos.shape, dim)
    labels = subselect(pos, time_steps)
    return labels.astype(np.float32).reshape([-1])


def correlation_label_fn(data):
    x = data['human_prob']
    # Format: [0]: Definitely Not, [1]: Probably Not, [2]: Probably, [3]: Definitely
    if len(x) == 4:
        neg = np.nan_to_num(x[0] / (x[0] + x[3]))
        pos = np.nan_to_num(x[3] / (x[0] + x[3]))
        return np.stack([neg, pos])
    # Format: [0]: Category 0, [1]: Category 1, [2]: Category 2, ...
    else:
        return np.nan_to_num(x / np.sum(x, keepdims=True))


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
    data = list(data)

    # Get number of positive and negative samples
    pos, neg = get_num_samples(data, label_fn)
    print("Before rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))
    num_pos = len(pos)
    num_neg = len(neg)

    # Rebalance data by oversampling underrepresented calls
    balanced_data = balancing(data, pos, neg)

    pos, neg = get_num_samples(balanced_data, label_fn)
    print("After rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

    return iter(balanced_data), num_pos, num_neg


def rebalance_human(data, label_fn):
    data = list(data)

    pos, neg = get_num_samples(data, label_fn)
    print("Before rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

    balanced_data = []
    cnt = 0
    for d in data:
        visual_label = label_fn(d, time_steps = slice(0, 4, 1))
        predict_label = label_fn(d, time_steps = slice(4, 10, 1))
        cnt += visual_label + predict_label
        if visual_label or predict_label or cnt > 0:
            if not visual_label or not predict_label:
                cnt -= 1
            balanced_data.append(d)

    pos, neg = get_num_samples(balanced_data, label_fn)
    print("After rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

    return iter(balanced_data)


def remap_and_filter(data, label_fn):
    label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: -1,
            8: 7, 9: 8, 10: -1, 11: 9, 12: 10, 13: -1,}

    data = list(data)

    pos, neg = get_num_samples(data, label_fn)
    print("Before remapping and filtering: all=%d" % (len(pos) + len(neg)))

    def remap_labels(x, x_map):
        return x_map[x]

    remapped_data = []
    for d in data:
        d['binary_labels'] = np.vectorize(remap_labels)(d['binary_labels'], label_map)
        if np.all(d['binary_labels'] != -1):
            remapped_data.append(d)

    pos, neg = get_num_samples(remapped_data, label_fn)
    print("After remapping and filtering: all=%d" % (len(pos) + len(neg)))

    return iter(remapped_data)


def load_best_params(metrics_file, reuse_best_params, num_params):
    if not os.path.isfile(metrics_file):
        best_params = [None] * num_params
        return best_params

    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)

    best_params = []
    for result in metrics['results']:
        if reuse_best_params and 'best_params' in result['result']:
            best_params.append(result['result']['best_params'])
        else:
            best_params.append(None)

    while len(best_params) < num_params:
        best_params.append(None)
    return best_params


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
    'rollingSliding': '/mnt/fs1/tdw_datasets/pilot-rollingSliding-redyellow/tfrecords/references.txt',
    'towers': '/mnt/fs1/tdw_datasets/pilot-towers-redyellow/tfrecords/references.txt',
    }

    references = np.array([])
    for k in reference_files:
        if k in test_feat_name:
            reference_dict = decode_references(reference_files[k])
            references = np.array([reference_dict[idx[0]] for idx in reference_ids])
            break

    return references


def run(
        seed,
        train_feature_file,
        test_feature_file,
        test_feat_name,
        model_dir,
        settings,
        grid_search_params = {'C': np.logspace(-8, 8, 17)},
        calculate_correlation = False,
        best_params = None,
        ):

    # Example 1: Classification via logistic regression

    # Build physics model
    feature_model = build_model(settings['model_fn'], slice(*settings['inp_time_steps']))
    feature_extractor = FeatureExtractor(feature_model)

    # Construct data providers
    train_data = build_data(train_feature_file)
    train_test_data = build_data(train_feature_file)
    test_data = build_data(test_feature_file)

    # Select label function
    label_fn = select_label_fn(slice(*settings['val_time_steps']), test_feat_name)

    # Rebalance data
    if False and 'cloth' in test_feat_name:
        if 'human' in test_feat_name:
            test_data = remap_and_filter(test_data, label_fn)
    else:
        np.random.seed(0)
        train_data, num_train_pos, num_train_neg = rebalance(train_data, label_fn)
        train_test_data, _, _ = rebalance(train_test_data, label_fn)
        if not 'human' in test_feat_name:
            test_data, num_test_pos, num_test_neg = rebalance(test_data, label_fn)
        else:
            num_test_pos = num_test_neg = 0

    # Build logistic regression model
    readout_model = LogisticRegressionReadoutModel(max_iter = 100, C=1.0, verbose=1)

    # Score unfitted predictions
    if best_params:
        # Reuse best params instead of running grid search
        grid_search_params = best_params
        grid_search_params = {k: [v] for k, v in grid_search_params.items()}
    metric_model = BatchMetricModel(feature_extractor, readout_model,
            accuracy, label_fn, grid_search_params,
            )

    metric_model.fit(train_data)
    best_params = metric_model._readout_model.best_params_ \
            if hasattr(metric_model._readout_model, 'best_params_') else {}
    train_acc = metric_model.score(train_test_data)
    test_acc = metric_model.score(test_data)

    print("Categorization accuracy: %f" % test_acc)

    per_example_results = compute_per_example_results(metric_model, test_feature_file,
            slice(*settings['val_time_steps']))
    per_example_results['path'] = reference2path(per_example_results['ref_id'], test_feat_name)

    result = {'per_example': per_example_results,
            'test_accuracy': test_acc, 'train_accuracy': train_acc,
            'num_train_pos': num_train_pos, 'num_train_neg': num_train_neg,
            'num_test_pos': num_test_pos, 'num_test_neg': num_test_neg,
            'best_params': best_params,
            'test_rebalanced': True if num_test_pos + num_test_neg > 0 else False}

    # Calculate human correlation
    if calculate_correlation:
        model_proba = metric_model.predict_proba(build_data(test_feature_file))
        _, human_proba = metric_model.extract_features_labels(build_data(test_feature_file), \
                label_fn = correlation_label_fn)
        human_proba = np.stack(human_proba)
        corr_coeff, p = scipy.stats.pearsonr(model_proba.flatten(), human_proba.flatten())

        print("Correlation coefficient: %f, p-value: %f" % (corr_coeff, p))

        result['corr_coeff'] = corr_coeff
        result['p_value'] = p

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


class Objective(PhysOptObjective):
    def __init__(self,
            exp_key,
            seed,
            train_data,
            feat_data,
            output_dir,
            extract_feat,
            debug,
            max_run_time = 86400 * 100, # 100 days
            reuse_best_params = False,
            ):
        assert len(feat_data) == 2, feat_data
        super().__init__(exp_key, seed, train_data, feat_data, output_dir,
                extract_feat, debug, max_run_time)
        self.reuse_best_params = reuse_best_params


    def __call__(self, *args, **kwargs):
        ret = super().__call__()
        best_params = load_best_params(self.metrics_file, self.reuse_best_params, len(SETTINGS))
        results = []
        for idx, settings in enumerate(SETTINGS):
            result = run(self.seed, self.train_feature_file,
                    self.test_feature_file, self.test_feat_data['name'],
                    self.model_dir, settings, best_params = best_params[idx])
            result = {'result': result}
            result.update(settings)
            results.append(result)
            # Write every iteration to be safe
            write_results(self.metrics_file, self.seed, self.train_data['name'],
                    self.train_feature_file, self.test_feature_file, self.model_dir, results)

        ret['loss'] = 0.0
        ret['results'] = results
        return ret



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
    objective = Objective(exp_key, seed, train_data, feat_data, output_dir, False, False)
    objective()
