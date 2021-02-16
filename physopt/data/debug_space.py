import os
import socket
from physopt.data.data_space import get_all_subsets, construct_data_spaces

_NUM_SEEDS = 1

if socket.gethostname() == 'node19-ccncluster':
    TRAIN_BASE_DIR = '/data1/eliwang/physion/rigid'
    HUMAN_BASE_DIR = '/data1/eliwang/physion//human_stimulis'
else:
    TRAIN_BASE_DIR = '/mnt/fs4/mrowca/neurips/images/rigid'
    HUMAN_BASE_DIR = '/mnt/fs4/fanyun/human_stimulis'

TRAIN = {k: os.path.join(TRAIN_BASE_DIR, k) for k in os.listdir(TRAIN_BASE_DIR)}
HUMAN = {k: os.path.join(HUMAN_BASE_DIR, k) for k in os.listdir(HUMAN_BASE_DIR)}

# Data subsets
_COLLIDE = {'name': 'collide', 'data': [TRAIN['collide2_new']]}
_HUMAN_COLLIDE_1 = {'name': 'human_collide_1', 'data': [HUMAN['collide2_new_1']]}

# Spaces
SEEDS = list(range(_NUM_SEEDS))

TRAIN_DATA = get_all_subsets([_COLLIDE,]) # TODO: use combined dataset?

TRAIN_FEAT_DATA = [_COLLIDE,]

TEST_FEAT_DATA = [_HUMAN_COLLIDE_1,]

METRICS_DATA = [
        (_COLLIDE, _HUMAN_COLLIDE_1),
        ]

SPACE = construct_data_spaces(SEEDS, TRAIN_DATA, TRAIN_FEAT_DATA, TEST_FEAT_DATA, METRICS_DATA)
