import os
from hyperopt import hp

_NUM_SEEDS = 3

TRAIN_BASE_DIR = '/mnt/fs4/mrowca/neurips/images/rigid'

TRAIN = {k: os.path.join(TRAIN_BASE_DIR, k) for k in os.listdir(TRAIN_BASE_DIR)}

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

# Spaces
SEED_SPACE = hp.choice('seed', list(range(_NUM_SEEDS)))
TRAIN_DATA_SPACE = hp.choice('train_data', [_CLOTH, _COLLIDE, _CONTAIN, _ROLL_SLIDE, _TOWER,
    _ALL, _NO_CLOTH, _NO_COLLIDE, _NO_CONTAIN, _NO_TOWER, _NO_ROLL_SLIDE])
TEST_DATA_SPACE = hp.choice('test_data', [_CLOTH, _COLLIDE, _CONTAIN, _ROLL_SLIDE, _TOWER])

TRAIN_SPACE = (SEED_SPACE, TRAIN_DATA_SPACE, ('', []))
TRAIN_TEST_SPACE = (SEED_SPACE, TRAIN_DATA_SPACE, TEST_DATA_SPACE)
