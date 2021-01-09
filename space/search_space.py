import os
from hyperopt import hp

_NUM_SEEDS = 3

TRAIN_BASE_DIR = '/mnt/fs4/mrowca/neurips/images/rigid'
HUMAN_BASE_DIR = '/mnt/fs4/fanyun/human_stimulis'

TRAIN = {k: os.path.join(TRAIN_BASE_DIR, k) for k in os.listdir(TRAIN_BASE_DIR)}
HUMAN = {k: os.path.join(HUMAN_BASE_DIR, k) for k in os.listdir(HUMAN_BASE_DIR)}

# Data subsets
_CLOTH = ('cloth', [TRAIN['cloth_on_object']])
_COLLIDE = ('collide', [TRAIN['collide2_new']])
_CONTAIN = ('contain', [TRAIN['contain']])
_TOWER = ('tower', [TRAIN['unstable3_tower']])
_ROLL_SLIDE = ('roll_slide', [TRAIN['roll_cube'], TRAIN['roll_sphere'], 
    TRAIN['slide_cube'], TRAIN['slide_sphere']])

_NO_CLOTH = ('no_cloth', _COLLIDE[1] + _CONTAIN[1] + _TOWER[1] + _ROLL_SLIDE[1])
_NO_COLLIDE = ('no_collide', _CLOTH[1] + _CONTAIN[1] + _TOWER[1] + _ROLL_SLIDE[1])
_NO_CONTAIN = ('no_contain', _CLOTH[1] + _COLLIDE[1] + _TOWER[1] + _ROLL_SLIDE[1])
_NO_TOWER = ('no_tower', _CLOTH[1] + _COLLIDE[1] + _CONTAIN[1] + _ROLL_SLIDE[1])
_NO_ROLL_SLIDE = ('no_roll_slide', _CLOTH[1] + _COLLIDE[1] + _CONTAIN[1] + _TOWER[1])

_ALL = ('all', _CLOTH[1] + _COLLIDE[1] + _CONTAIN[1] + _TOWER[1] + _ROLL_SLIDE[1])

_HUMAN_CLOTH = ('human_cloth', [HUMAN['cloth_on_object']])
_HUMAN_COLLIDE_0 = ('human_collide_0', [HUMAN['collide2_new_0']])
_HUMAN_COLLIDE_1 = ('human_collide_1', [HUMAN['collide2_new_1']])
_HUMAN_CONTAIN = ('human_contain', [HUMAN['contain']])
_HUMAN_TOWER = ('human_tower', [HUMAN['unstable3_tower']])
_HUMAN_ROLL_SLIDE = ('human_roll_slide', [HUMAN['slide_roll']])

# Spaces
SEED_SPACE = hp.choice('seed', list(range(_NUM_SEEDS)))
TRAIN_DATA_SPACE = hp.choice('train_data', [_CLOTH, _COLLIDE, _CONTAIN, _ROLL_SLIDE, _TOWER,
    _ALL, _NO_CLOTH, _NO_COLLIDE, _NO_CONTAIN, _NO_TOWER, _NO_ROLL_SLIDE])
TEST_DATA_SPACE = hp.choice('test_data', [_CLOTH, _COLLIDE, _CONTAIN, _ROLL_SLIDE, _TOWER])
HUMAN_DATA_SPACE = hp.choice('test_data', [_HUMAN_CLOTH, _HUMAN_COLLIDE_0, _HUMAN_COLLIDE_1,
    _HUMAN_CONTAIN, _HUMAN_ROLL_SLIDE, _HUMAN_TOWER])

TRAIN_SPACE = (SEED_SPACE, TRAIN_DATA_SPACE, ('', []))
TRAIN_TEST_SPACE = (SEED_SPACE, TRAIN_DATA_SPACE, TEST_DATA_SPACE)
HUMAN_TEST_SPACE = (SEED_SPACE, TRAIN_DATA_SPACE, HUMAN_DATA_SPACE)

METRICS_DATA_SPACE = hp.choice('metrics_data', [
    (_CLOTH, _HUMAN_CLOTH),
    (_COLLIDE, _HUMAN_COLLIDE_0),
    (_COLLIDE, _HUMAN_COLLIDE_1),
    (_CONTAIN, _HUMAN_CONTAIN),
    (_ROLL_SLIDE, _HUMAN_ROLL_SLIDE),
    (_TOWER, _HUMAN_TOWER),
    ])
METRICS_SPACE = (SEED_SPACE, TRAIN_DATA_SPACE, METRICS_DATA_SPACE)
#TRAIN_SPACE = hp.choice('seed', [(seed, TRAIN_DATA_SPACE, '')
#    for seed in range(_NUM_SEEDS)])
