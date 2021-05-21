import os
import socket
from physopt.data.data_space import get_combined_subset, construct_data_spaces

_NUM_SEEDS = 1

if socket.gethostname() == 'node19-ccncluster':
    DOMINO_BASE_DIR = '/data1/eliwang/dominoes'
elif 'physion' in socket.gethostname():
    DOMINO_BASE_DIR = '/mnt/fs4/hsiaoyut/tdw_physics/data/dominoes'
else:
    DOMINO_BASE_DIR = '/mnt/fs4/eliwang/dominoes'

# Data subsets
_DOMINOES = {'name': 'dominoes',
        'data': [os.path.join(DOMINO_BASE_DIR, '*/train/*.hdf5')]}
_TEST_DOMINOES = {'name': 'test_dominoes',
        'data': [os.path.join(DOMINO_BASE_DIR, '*/train_readout/*.hdf5')]}
_VAL_DOMINOES = {'name': 'val_dominoes',
        'data': [os.path.join(DOMINO_BASE_DIR, '*/valid_readout/*.hdf5')]}

# Spaces
SEEDS = list(range(_NUM_SEEDS))

TRAIN_DATA = [_DOMINOES]

TRAIN_FEAT_DATA = [_TEST_DOMINOES]

TEST_FEAT_DATA = [_VAL_DOMINOES]

METRICS_DATA = [
        (_TEST_DOMINOES, _VAL_DOMINOES),
        ]

SPACE = construct_data_spaces(SEEDS, TRAIN_DATA, TRAIN_FEAT_DATA, TEST_FEAT_DATA, METRICS_DATA)
