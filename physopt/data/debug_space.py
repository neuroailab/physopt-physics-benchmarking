import os
import socket
from physopt.data.data_space import get_combined_subset, construct_data_spaces

_NUM_SEEDS = 1

if socket.gethostname() == 'node19-ccncluster':
    TRAIN_BASE_DIR = '/data1/eliwang/physion/rigid'
    HUMAN_BASE_DIR = '/data1/eliwang/physion/human_stimulis'
else:
    TRAIN_BASE_DIR = '/mnt/fs4/mrowca/neurips/images/rigid'
    HUMAN_BASE_DIR = '/mnt/fs4/fanyun/human_stimulis'

TRAIN = {k: os.path.join(TRAIN_BASE_DIR, k) for k in os.listdir(TRAIN_BASE_DIR)}
HUMAN = {k: os.path.join(HUMAN_BASE_DIR, k) for k in os.listdir(HUMAN_BASE_DIR)}

# Data subsets
_DOMINOES = {'name': 'dominoes', 'data': [TRAIN['testing_dominoes']]}
# _CLOTH = {'name': 'cloth', 'data': [TRAIN['cloth_on_object']]}
# _COLLIDE = {'name': 'collide', 'data': [TRAIN['collide2_new']]}
# _CONTAIN = {'name': 'contain', 'data': [TRAIN['contain']]}
# _TOWER = {'name': 'tower', 'data': [TRAIN['unstable3_tower']]}
# _ROLL_SLIDE = {'name': 'roll_slide', 'data': [TRAIN['roll_cube'], TRAIN['roll_sphere'],
#     TRAIN['slide_cube'], TRAIN['slide_sphere']]}

# _HUMAN_COLLIDE_1 = {'name': 'human_collide_1', 'data': [HUMAN['collide2_new_1']]}
_HUMAN_DOMINOES = {'name': 'human_dominoes', 'data': [HUMAN['example_dominoes']]}

# Spaces
SEEDS = list(range(_NUM_SEEDS))

TRAIN_DATA = [_DOMINOES,]

TRAIN_FEAT_DATA = [_DOMINOES,]

TEST_FEAT_DATA = [_HUMAN_DOMINOES,]

METRICS_DATA = [
        (_DOMINOES, _HUMAN_DOMINOES),
        ]

SPACE = construct_data_spaces(SEEDS, TRAIN_DATA, TRAIN_FEAT_DATA, TEST_FEAT_DATA, METRICS_DATA)
