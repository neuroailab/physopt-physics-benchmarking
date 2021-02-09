import os
from data_space.data_space import get_all_subsets, construct_data_spaces



_NUM_SEEDS = 3

TRAIN_BASE_DIR = '/mnt/fs4/mrowca/neurips/images/rigid'
HUMAN_BASE_DIR = '/mnt/fs4/fanyun/human_stimulis'

TRAIN = {k: os.path.join(TRAIN_BASE_DIR, k) for k in os.listdir(TRAIN_BASE_DIR)}
HUMAN = {k: os.path.join(HUMAN_BASE_DIR, k) for k in os.listdir(HUMAN_BASE_DIR)}

# Data subsets
_CLOTH = {'name': 'cloth', 'data': [TRAIN['cloth_on_object']]}
_COLLIDE = {'name': 'collide', 'data': [TRAIN['collide2_new']]}
_CONTAIN = {'name': 'contain', 'data': [TRAIN['contain']]}
_TOWER = {'name': 'tower', 'data': [TRAIN['unstable3_tower']]}
_ROLL_SLIDE = {'name': 'roll_slide', 'data': [TRAIN['roll_cube'], TRAIN['roll_sphere'],
    TRAIN['slide_cube'], TRAIN['slide_sphere']]}
#_RANDOM = {'name': 'random', 'data': [TRAIN['random_push'], TRAIN['random_tower']]}

_HUMAN_CLOTH = {'name': 'human_cloth', 'data': [HUMAN['cloth_on_object']]}
_HUMAN_COLLIDE_0 = {'name': 'human_collide_0', 'data': [HUMAN['collide2_new_0']]}
_HUMAN_COLLIDE_1 = {'name': 'human_collide_1', 'data': [HUMAN['collide2_new_1']]}
_HUMAN_CONTAIN = {'name': 'human_contain', 'data': [HUMAN['contain']]}
_HUMAN_TOWER = {'name': 'human_tower', 'data': [HUMAN['unstable3_tower']]}
_HUMAN_ROLL_SLIDE = {'name': 'human_roll_slide', 'data': [HUMAN['slide_roll']]}

# Spaces
SEEDS = list(range(_NUM_SEEDS))

TRAIN_DATA = get_all_subsets([_CLOTH, _COLLIDE, _CONTAIN, _TOWER, _ROLL_SLIDE])
#TRAIN_DATA += [_RANDOM]

TRAIN_FEAT_DATA = [_CLOTH, _COLLIDE, _CONTAIN, _ROLL_SLIDE, _TOWER]

HUMAN_FEAT_DATA = [_HUMAN_CLOTH, _HUMAN_COLLIDE_0, _HUMAN_COLLIDE_1,
    _HUMAN_CONTAIN, _HUMAN_ROLL_SLIDE, _HUMAN_TOWER]

METRICS_DATA = [
        (_CLOTH, _HUMAN_CLOTH),
        (_COLLIDE, _HUMAN_COLLIDE_0),
        (_COLLIDE, _HUMAN_COLLIDE_1),
        (_CONTAIN, _HUMAN_CONTAIN),
        (_ROLL_SLIDE, _HUMAN_ROLL_SLIDE),
        (_TOWER, _HUMAN_TOWER),
        ]

SPACE = construct_data_spaces(SEEDS, TRAIN_DATA, TRAIN_FEAT_DATA, HUMAN_FEAT_DATA, METRICS_DATA)
