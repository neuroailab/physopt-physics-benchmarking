import os
from physopt.data.data_space import get_all_subsets, construct_data_spaces



_NUM_SEEDS = 1

TRAIN_BASE_DIR = '/mnt/fs4/mrowca/neurips/images/rigid'
HUMAN_BASE_DIR = '/mnt/fs4/fanyun/human_stimulis'

TRAIN = {k: os.path.join(TRAIN_BASE_DIR, k) for k in os.listdir(TRAIN_BASE_DIR)}
HUMAN = {k: os.path.join(HUMAN_BASE_DIR, k) for k in os.listdir(HUMAN_BASE_DIR)}

# Data subsets
_CLOTH = {'name': 'cloth', 'data': [TRAIN['cloth_on_object']]}

_HUMAN_CLOTH = {'name': 'human_cloth', 'data': [HUMAN['cloth_on_object']]}

# Spaces
SEEDS = list(range(_NUM_SEEDS))

TRAIN_DATA = get_all_subsets([_CLOTH,])

TRAIN_FEAT_DATA = [_CLOTH,]

TEST_FEAT_DATA = [_HUMAN_CLOTH,]

METRICS_DATA = [
        (_CLOTH, _HUMAN_CLOTH),
        ]

SPACE = construct_data_spaces(SEEDS, TRAIN_DATA, TRAIN_FEAT_DATA, TEST_FEAT_DATA, METRICS_DATA)
