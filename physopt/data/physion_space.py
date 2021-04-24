import os
import socket
from physopt.data.data_space import get_all_subsets, construct_data_spaces

_NUM_SEEDS = 1

TRAIN_BASE_DIR = '/mnt/fs4/dbear/tdw_datasets/'

# Data subsets
_DOMINOES = {'name': 'dominoes',
        'data': [os.path.join(TRAIN_BASE_DIR, 'example_dominoes')]}

# Spaces
SEEDS = list(range(_NUM_SEEDS))

TRAIN_DATA = get_all_subsets([_DOMINOES])
#TRAIN_DATA += [_RANDOM]

TRAIN_FEAT_DATA = [_DOMINOES]

TEST_FEAT_DATA = [_DOMINOES]

METRICS_DATA = [
        (_DOMINOES, _DOMINOES),
        ]

SPACE = construct_data_spaces(SEEDS, TRAIN_DATA, TRAIN_FEAT_DATA, TEST_FEAT_DATA, METRICS_DATA)
