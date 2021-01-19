import os
import numpy as np
import scipy
import pickle

from metrics.base.feature_extractor import FeatureExtractor
from metrics.base.readout_model import IdentityModel
from metrics.base.metric_model import BatchMetricModel

from metrics.physics.linear_readout_model import LinearRegressionReadoutModel, \
        LogisticRegressionReadoutModel

from metrics.physics.metric_fns import accuracy, squared_error

from hyperopt import STATUS_OK

SETTINGS = [
        {
            'inp_time_steps': (0, 10, 1),
            'val_time_steps': (4, 10, 1),
            'model_fn': 'visual_scene_model_fn',
            },
        {
            'inp_time_steps': (0, 10, 1),
            'val_time_steps': (4, 10, 1),
            'model_fn': 'rollout_scene_model_fn',
            },
        {
            'inp_time_steps': (0, 4, 1),
            'val_time_steps': (0, 4, 1),
            'model_fn': 'visual_scene_model_fn',
            },
        {
            'inp_time_steps': (0, 4, 1),
            'val_time_steps': (4, 10, 1),
            'model_fn': 'visual_scene_model_fn',
            },
        ]

def build_data(path):
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
        if 'collide' in experiment:
            return collision_label_fn(data, time_steps)
        elif 'tower' in experiment:
            return stable_label_fn(data, time_steps)
        elif 'contain' in experiment:
            return contain_label_fn(data, time_steps)
        elif 'cloth' in experiment:
            return object_category_label_fn(data, time_steps)
        elif 'roll_vs_slide' in experiment:
            return rolling_label_fn(data, time_steps)
        return labels
    return label_fn


def pos_label_fn(data, time_steps, dim = 2):
    #TODO Write out positions
    #pos = data['bboxes']
    dummy_pos = np.random.rand(*(data['binary_labels'].shape[0], 2))
    pos = dummy_pos
    assert pos.shape[-1] == dim, (pos.shape, dim)
    labels = subselect(pos, time_steps)
    return labels.astype(np.float32).reshape([-1])


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

    # Rebalance data by oversampling underrepresented calls
    balanced_data = balancing(data, pos, neg)

    pos, neg = get_num_samples(balanced_data, label_fn)
    print("After rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

    return iter(balanced_data)


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


def get_model_dir(run_name, seed, base_dir = '/mnt/fs4/mrowca/hyperopt/rpin'):
    return os.path.join(base_dir, run_name, str(seed), 'model')


def run(
        seed,
        train_name,
        train_feat_name,
        test_feat_name,
        base_dir,
        settings,
        grid_search_params = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]},
        ):
    model_dir = get_model_dir(train_name, seed, base_dir)
    train_path = os.path.join(model_dir, 'features', train_feat_name, 'feat.pkl')
    test_path = os.path.join(model_dir, 'features', test_feat_name, 'feat.pkl')

    # Example 1: Classification via logistic regression

    # Build physics model
    feature_model = build_model(settings['model_fn'], slice(*settings['inp_time_steps']))
    feature_extractor = FeatureExtractor(feature_model)

    # Construct data providers
    train_data = build_data(train_path)
    test_data = build_data(test_path)

    # Select label function
    label_fn = select_label_fn(slice(*settings['val_time_steps']), test_feat_name)

    # Rebalance data
    if 'cloth' in test_feat_name:
        if 'human' in test_feat_name:
            test_data = remap_and_filter(test_data, label_fn)
    else:
        np.random.seed(0)
        train_data = rebalance(train_data, label_fn)
        if not 'human' in test_feat_name:
            test_data = rebalance(test_data, label_fn)

    # Build logistic regression model
    readout_model = LogisticRegressionReadoutModel(max_iter = 100, C=1.0, verbose=1)

    # Score unfitted predictions
    metric_model = BatchMetricModel(feature_extractor, readout_model,
            accuracy, label_fn, grid_search_params,
            )

    metric_model.fit(train_data)
    result = metric_model.score(test_data)

    print("Categorization accuracy: %f" % result)

    return result


def write_results(
        seed,
        train_name,
        train_feat_name,
        test_feat_name,
        base_dir,
        results,
        ):
    model_dir = get_model_dir(train_name, seed, base_dir)
    write_path = os.path.join(model_dir, 'features', test_feat_name,
            'metrics_results.pkl')
    data = {
            'seed': seed,
            'train_name': train_name,
            'train_feat_name': train_feat_name,
            'test_feat_name': test_feat_name,
            'base_dir': base_dir,
            'results': results,
            }
    with open(write_path, 'wb') as f:
        pickle.dump(data, f)
    return


class Objective():
    def __init__(self,
            seed,
            train_data,
            feat_data,
            output_dir,
            extract_feat):
        assert len(feat_data) == 2, feat_data
        self.seed = seed
        self.train_data = train_data
        self.train_feat_data = feat_data[0]
        self.test_feat_data = feat_data[1]
        self.output_dir = output_dir


    def __call__(self, *args, **kwargs):
        results = []
        for settings in SETTINGS:
            result = run(self.seed, self.train_data['name'], self.train_feat_data['name'],
                    self.test_feat_data['name'], self.output_dir, settings)
            result = {'result': result}
            result.update(settings)
            results.append(result)
            # Write every iteration to be safe
            write_results(self.seed, self.train_data['name'], self.train_feat_data['name'],
                    self.test_feat_data['name'], self.output_dir, results)

        return {
                'loss': 0.0,
                'status': STATUS_OK,
                'seed': self.seed,
                'train_data': self.train_data,
                'train_feat_data': self.train_feat_data,
                'test_feat_data': self.test_feat_data,
                'base_dir': self.output_dir,
                'results': results,
                }

if __name__ == '__main__':
    seed = 0
    train_data = {'name': 'cloth',
            'data': ['/mnt/fs4/mrowca/neurips/images/rigid/cloth_on_object']
            }
    feat_data = (
        {'name': 'cloth',
            'data': ['/mnt/fs4/mrowca/neurips/images/rigid/cloth_on_object']
            },
        {'name': 'human_cloth',
            'data': ['/mnt/fs4/mrowca/neurips/images/rigid/cloth_on_object']
            }
        )
    output_dir = '/mnt/fs4/mrowca/hyperopt/rpin/'
    objective = Objective(seed, train_data, feat_data, output_dir, False)
    objective()
