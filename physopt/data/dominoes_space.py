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


SUBSETS = [
        'pilot_dominoes_0mid_d3chairs_o1plants_tdwroom',
        'pilot_dominoes_1mid_J025R45_boxroom',
        'pilot_dominoes_1mid_J025R45_o1flex_tdwroom',
        'pilot_dominoes_2mid_J020R15_d3chairs_o1plants_tdwroom',
        'pilot_dominoes_2mid_J025R30_tdwroom',
        'pilot_dominoes_4mid_boxroom',
        'pilot_dominoes_4midRM1_boxroom',
        'pilot_dominoes_4midRM1_tdwroom',
        'pilot_dominoes_4mid_tdwroom',
        'pilot_dominoes_default_boxroom',
        'pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom',
        ]

# Data subsets
_DOMINOES = {'name': 'dominoes',
        'data': [os.path.join(DOMINO_BASE_DIR, s, 'train/0*.hdf5') for s in SUBSETS]}
# _DOMINOES = {'name': 'dominoes_100',
#         'data': [os.path.join(DOMINO_BASE_DIR, s, 'train/00*.hdf5') for s in SUBSETS]}
# _DOMINOES = {'name': 'dominoes_10',
#         'data': [os.path.join(DOMINO_BASE_DIR, s, 'train/000*.hdf5') for s in SUBSETS]}
_TEST_DOMINOES = {'name': 'test_dominoes',
        'data': [os.path.join(DOMINO_BASE_DIR, s, 'test/*.hdf5') for s in SUBSETS]}
_VAL_DOMINOES = {'name': 'val_dominoes',
        'data': [os.path.join(DOMINO_BASE_DIR, s, 'val/*.hdf5') for s in SUBSETS]}

# Spaces
SEEDS = list(range(_NUM_SEEDS))

TRAIN_DATA = [_DOMINOES]

TRAIN_FEAT_DATA = [_TEST_DOMINOES]

TEST_FEAT_DATA = [_VAL_DOMINOES]

METRICS_DATA = [
        (_TEST_DOMINOES, _VAL_DOMINOES),
        ]

SPACE = construct_data_spaces(SEEDS, TRAIN_DATA, TRAIN_FEAT_DATA, TEST_FEAT_DATA, METRICS_DATA)
