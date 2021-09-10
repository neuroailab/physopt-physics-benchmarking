import os
from .config import get_cfg_defaults
import itertools

# Each subset is defined as a dict('name': NAME, 'data': DATA), where
# NAME = str, DATA = list(subset_paths)

EMPTY_DATA = [{'name': '', 'data': []}]


# All subsets combined
def get_combined_subset(subsets): # return list for consistency?
    return {'name': 'all', 'train': [s for subset in subsets for s in subset['train']], 'test': [s for subset in subsets for s in subset['test']]}

# All subsets combined but one
def get_combined_but_one_subsets(subsets):
    combined_but_one = []
    for subset in subsets:
        abo_subsets = [ss for ss in subsets if ss is not subset]
        abo_data = get_combined_subset(abo_subsets)
        abo_data['name'] = 'no_{}'.format(subset['name']) # change name from 'all' to 'no_XXX'
        combined_but_one.append(abo_data)
    return combined_but_one

def construct_data_spaces(seeds, train_data, train_feat_data, test_feat_data, metrics_data):
    spaces = {}
    spaces['train_feat'] = construct_extraction_space(seeds, train_data, train_feat_data)
    spaces['test_feat'] = construct_extraction_space(seeds, train_data, test_feat_data)
    spaces['metrics'] = construct_metrics_space(seeds, train_data, metrics_data)
    return spaces

def _get_subsets(scenarios, filepattern, traindir, testdir): # list of {'name': scenario, 'train': train_datapaths, 'test': test_datapaths} dicts for each scenario
    return [
        {
            'name': scenario, 
            'train': [os.path.join(traindir, scenario, filepattern)], 
            'test': [os.path.join(testdir, scenario, filepattern)],
            } 
        for scenario in scenarios]

def get_data_space(
        data_space,
        ):
    cfg = get_cfg_defaults()
    dirname = os.path.dirname(__file__)
    cfg.merge_from_file(os.path.join(dirname, data_space+'.yaml'))
    # TODO: add merge debug config?
    cfg.freeze()
    print(cfg)
    
    seeds = list(range(cfg.NUM_SEEDS))

    only_dynamics_data = _get_subsets(cfg.SCENARIOS, cfg.FILE_PATTERN, cfg.DYNAMICS_TRAIN_DIR, cfg.DYNAMICS_TEST_DIR)
    dynamics_data = []
    if len(cfg.SCENARIOS) > 1:
        if 'only' in cfg.TRAINING_PROTOCOLS:
            dynamics_data.extend(only_dynamics_data)
        if 'abo' in cfg.TRAINING_PROTOCOLS:
            abo_dynamics_data = get_combined_but_one_subset(only_dynamics_data)
            dynamics_data.extend(abo_dynamics_data)
        if 'all' in cfg.TRAINING_PROTOCOLS:
            all_dynamics_data = get_combined_subset(only_dynamics_data)
            dynamics_data.append(all_dynamics_data)
    else:
        assert 'abo' not in cfg.TRAINING_PROTOCOLS, "Can't use all-but-one training protocol when there's only one scenario"
        dynamics_data = only_dynamics_data

    readout_data = _get_subsets(cfg.SCENARIOS, cfg.FILE_PATTERN, cfg.READOUT_TRAIN_DIR, cfg.READOUT_TEST_DIR)

    data_spaces = itertools.product(seeds, dynamics_data, [readout_data]) # TODO: remove some readouts depending on args

    return data_spaces
