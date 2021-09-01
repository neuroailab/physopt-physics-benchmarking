import os
import socket
from physopt.data.data_space import get_all_subsets, construct_data_spaces

_NUM_SEEDS = 1

TRAIN_BASE_DIR = '/data1/eliwang/physion_train_data'
TEST_BASE_DIR = '/data1/eliwang/physion_test_data'

SCENARIOS = ['Collide', 'Contain', 'Dominoes', 'Drape', 'Drop', 'Link', 'Roll', 'Support']

# Data subsets
_TRAIN_SCENARIOS = []
_TRAIN_FEAT_SCENARIOS = []
_TEST_FEAT_SCENARIOS = []

for scenario in SCENARIOS:
    _TRAIN_SCENARIOS.append({'name': scenario, 'data': [os.path.join(TRAIN_BASE_DIR, 'dynamics_training', scenario, '*.hdf5')]})
    _TRAIN_FEAT_SCENARIOS.append({'name': scenario, 'data': [os.path.join(TRAIN_BASE_DIR, 'readout_training', scenario, '*.hdf5')]})
    _TEST_FEAT_SCENARIOS.append({'name': scenario, 'data': [os.path.join(TEST_BASE_DIR, 'model_testing', scenario, '*.hdf5')]})

# Spaces
SEEDS = list(range(_NUM_SEEDS))

TRAIN_DATA = get_all_subsets(_TRAIN_SCENARIOS)
TRAIN_FEAT_DATA = _TRAIN_FEAT_SCENARIOS
TEST_FEAT_DATA = _TEST_FEAT_SCENARIOS
METRICS_DATA = zip(_TRAIN_FEAT_SCENARIOS, _TEST_FEAT_SCENARIOS)

# TRAIN_DATA = TRAIN_DATA[:8] # single scenario
# TRAIN_DATA = TRAIN_DATA[8:16] # leave-one-out
# TRAIN_DATA = TRAIN_DATA[16:] #  all
print(TRAIN_DATA)

SPACE = construct_data_spaces(SEEDS, TRAIN_DATA, TRAIN_FEAT_DATA, TEST_FEAT_DATA, METRICS_DATA)
