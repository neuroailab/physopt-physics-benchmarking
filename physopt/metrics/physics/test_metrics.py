import os
import numpy as np
import scipy
import pickle
import logging

from physopt.metrics.base.feature_extractor import FeatureExtractor
from physopt.metrics.base.readout_model import IdentityModel
from physopt.metrics.base.metric_model import BatchMetricModel

from physopt.metrics.physics.linear_readout_model import LinearRegressionReadoutModel, \
        LogisticRegressionReadoutModel

from physopt.metrics.physics.metric_fns import accuracy, squared_error
from physopt.utils import PhysOptObjective

SETTINGS = [ # TODO: might not want this to be hardcoded
        {
            'inp_time_steps': (0, 50, 1),
            'val_time_steps': (15, 50, 1),
            'model_fn': 'visual_scene_model_fn',
            },
        {
            'inp_time_steps': (0, 50, 1),
            'val_time_steps': (15, 50, 1),
            'model_fn': 'rollout_scene_model_fn',
            },
        {
            'inp_time_steps': (0, 15, 1),
            'val_time_steps': (15, 50, 1),
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

    logging.info('Data shape: {}'.format(np.array(data).shape))
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
        elif 'roll_vs_slide' in experiment: # TODO: should this be roll_slide
            return rolling_label_fn(data, time_steps)
        elif 'dominoes' in experiment:
            return collision_label_fn(data, time_steps) # TODO: using collision label fn for now, should be the same
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
    if len(x) == 4: # TODO: might want to change this to account for probably responses
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
    logging.info("Before rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

    # Rebalance data by oversampling underrepresented calls
    balanced_data = balancing(data, pos, neg)

    pos, neg = get_num_samples(balanced_data, label_fn)
    logging.info("After rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

    return iter(balanced_data)


def rebalance_human(data, label_fn):
    data = list(data)

    pos, neg = get_num_samples(data, label_fn)
    logging.info("Before rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

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
    logging.info("After rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))

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


def run(
        seed,
        train_feature_file,
        test_feature_file,
        test_feat_name,
        model_dir,
        settings,
        grid_search_params = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]},
        calculate_correlation = False,
        ):

    # Example 1: Classification via logistic regression

    # Build physics model
    feature_model = build_model(settings['model_fn'], slice(*settings['inp_time_steps']))
    feature_extractor = FeatureExtractor(feature_model)

    # Construct data providers
    logging.info(train_feature_file)
    logging.info(test_feature_file)
    train_data = build_data(train_feature_file)
    test_data = build_data(test_feature_file)

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
        else:
            data = list(test_data)
            pos, neg = get_num_samples(data, label_fn) # Get number of positive and negative samples
            logging.info("No rebalancing: pos=%d, neg=%d" % (len(pos), len(neg)))
            test_data = iter(data)

    # Build logistic regression model
    readout_model = LogisticRegressionReadoutModel(max_iter = 100, C=1.0, verbose=1)

    # Score unfitted predictions
    metric_model = BatchMetricModel(feature_extractor, readout_model,
            accuracy, label_fn, grid_search_params,
            )

    metric_model.fit(train_data)
    train_data = build_data(train_feature_file) # hack to get original data before rebalance iter
    train_acc = metric_model.score(train_data)
    test_acc = metric_model.score(test_data)

    print("Categorization train accuracy: %f" % train_acc)
    print("Categorization test accuracy: %f" % test_acc)

    result = {'train_accuracy': train_acc, 'test_accuracy': test_acc}
    if grid_search_params is not None:
        result['best_params'] = metric_model._readout_model.best_params_

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
            ):
        assert len(feat_data) == 2, feat_data
        super().__init__(exp_key, seed, train_data, feat_data, output_dir, extract_feat, debug)


    def __call__(self, *args, **kwargs):
        ret = super().__call__()
        results = []
        for settings in SETTINGS:
            result = run(self.seed, self.train_feature_file,
                    self.test_feature_file, self.test_feat_data['name'],
                    self.model_dir, settings, 
                    # calculate_correlation=True,
                    # grid_search_params=None if self.debug else {'C': np.logspace(-2, 2, 5)},
                    grid_search_params={'C': np.logspace(-8, 8, 17)},
                    )
            result = {'result': result}
            result.update(settings)
            logging.info(result)
            results.append(result)
            # Write every iteration to be safe
            write_results(self.metrics_file, self.seed, self.train_data['name'],
                    self.train_feature_file, self.test_feature_file, self.model_dir, results)

        ret['loss'] = 0.0
        ret['results'] = results
        return ret



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
    output_dir = '/mnt/fs4/mrowca/hyperopt/RPIN/'
    exp_key = '0_cloth_human_cloth_metrics'
    objective = Objective(exp_key, seed, train_data, feat_data, output_dir, False)
    objective()
