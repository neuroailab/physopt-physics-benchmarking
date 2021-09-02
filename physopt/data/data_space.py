import os
from .config import get_cfg_defaults

# Each subset is defined as a dict('name': NAME, 'data': DATA), where
# NAME = str, DATA = list(subset_paths)

EMPTY_DATA = [{'name': '', 'data': []}]


# All subsets combined
def get_combined_subset(subsets): # return list for consistency?
    return {'name': 'all', 'data': [s for subset in subsets for s in subset['data']]}


# All subsets combined but one
def get_combined_but_one_subsets(subsets):
    combined_but_one = []
    for subset in subsets:
        combined_data = get_combined_subset(subsets)['data']
        [combined_data.remove(s) for s in subset['data']]
        combined_but_one.append({'name': 'no_{0}'.format(subset['name']), 'data': combined_data})
    return combined_but_one


def get_all_subsets(subsets):
    if not isinstance(subsets, list):
        subsets = list(subsets)
    if len(subsets) > 1:
        all_subsets = subsets \
                + get_combined_but_one_subsets(subsets) \
                + [get_combined_subset(subsets)]
    else:
        all_subsets = subsets
    return all_subsets


def construct_extraction_space(seeds, train_data, feat_data):
    return (seeds, train_data, feat_data)


def construct_metrics_space(seeds, train_data, metrics_data):
    return (seeds, train_data, metrics_data)


def construct_data_spaces(seeds, train_data, train_feat_data, test_feat_data, metrics_data):
    spaces = {}
    spaces['train_feat'] = construct_extraction_space(seeds, train_data, train_feat_data)
    spaces['test_feat'] = construct_extraction_space(seeds, train_data, test_feat_data)
    spaces['metrics'] = construct_metrics_space(seeds, train_data, metrics_data)
    return spaces

def _get_subsets(basedir, scenarios, filepattern):
    return [{'name': scenario, 'data': [os.path.join(basedir, scenario, filepattern)]} for scenario in scenarios]

def get_data_space(
        data_space,
        ):
    cfg = get_cfg_defaults()
    dirname = os.path.dirname(__file__)
    cfg.merge_from_file(os.path.join(dirname, data_space+'.yaml'))
    # TODO: add merge debug config?
    cfg.freeze()
    print(cfg)
    
    # Data subsets
    only_dynamics_train_data = _get_subsets(cfg.DYNAMICS_TRAIN_DIR, cfg.SCENARIOS, cfg.FILE_PATTERN)
    dynamics_train_data = []
    if len(cfg.SCENARIOS) > 1:
        if 'only' in cfg.TRAINING_PROTOCOLS:
            dynamics_train_data.extend(only_dynamics_train_data)
        if 'abo' in cfg.TRAINING_PROTOCOLS:
            abo_dynamics_train_data = get_combined_but_one_subset(only_dynamics_train_data)
            dynamics_train_data.extend(abo_dynamics_train_data)
        if 'all' in cfg.TRAINING_PROTOCOLS:
            all_dynamics_train_data = get_combined_subset(only_dynamics_train_data)
            dynamics_train_data.append(all_dynamics_train_data)
    else:
        assert 'abo' not in cfg.TRAINING_PROTOCOLS, "Can't use all-but-one training protocol when there's only one scenario"
        dynamics_train_data = only_dynamics_train_data

    readout_train_data = _get_subsets(cfg.READOUT_TRAIN_DIR, cfg.SCENARIOS, cfg.FILE_PATTERN)
    readout_test_data = _get_subsets(cfg.READOUT_TEST_DIR, cfg.SCENARIOS, cfg.FILE_PATTERN)

    metrics_data = zip(readout_train_data, readout_test_data)
    seeds = list(range(cfg.NUM_SEEDS))

    space = construct_data_spaces(seeds, dynamics_train_data, readout_train_data, readout_test_data, metrics_data)
    return space
