import os
import socket
from physopt.data.data_space import get_all_subsets, construct_data_spaces

_NUM_SEEDS = 1

BASE_DIR = '/mnt/fs4/hsiaoyut/tdw_physics/data/'
NEW_BASE_DIR = '/mnt/fs4/tdw_datasets/'

# Data subsets
_TRAIN_CLOTH = {'name': 'cloth',
        'data': [os.path.join(BASE_DIR, 'clothSagging/*/train/*.hdf5')]}
_TRAIN_FEAT_CLOTH = {'name': 'train_cloth',
        'data': [os.path.join(BASE_DIR, 'clothSagging/*/train_readout/*.hdf5')]}
_TEST_FEAT_CLOTH = {'name': 'test_cloth',
        'data': [os.path.join(NEW_BASE_DIR, '*clothSagging*/*redyellow/*.hdf5')]}

_TRAIN_COLLISION = {'name': 'collision',
        'data': [os.path.join(BASE_DIR, 'collision/*/train/*.hdf5')]}
_TRAIN_FEAT_COLLISION = {'name': 'train_collision',
        'data': [os.path.join(BASE_DIR, 'collision/*/train_readout/*.hdf5')]}
_TEST_FEAT_COLLISION = {'name': 'test_collision',
        'data': [os.path.join(NEW_BASE_DIR, '*collision*/*redyellow/*.hdf5')]}

_TRAIN_CONTAINMENT = {'name': 'containment',
        'data': [os.path.join(BASE_DIR, 'containment/*/train/*.hdf5')]}
_TRAIN_FEAT_CONTAINMENT = {'name': 'train_containment',
        'data': [os.path.join(BASE_DIR, 'containment/*/train_readout/*.hdf5')]}
_TEST_FEAT_CONTAINMENT = {'name': 'test_containment',
        'data': [os.path.join(NEW_BASE_DIR, '*containment*/*redyellow/*.hdf5')]}

_TRAIN_DOMINOES = {'name': 'dominoes',
        'data': [os.path.join(BASE_DIR, 'dominoes/*/train/*.hdf5')]}
_TRAIN_FEAT_DOMINOES = {'name': 'train_dominoes',
        'data': [os.path.join(BASE_DIR, 'dominoes/*/train_readout/*.hdf5')]}
_TEST_FEAT_DOMINOES = {'name': 'test_dominoes',
        'data': [os.path.join(NEW_BASE_DIR, '*dominoes*/*redyellow/*.hdf5')]}

_TRAIN_DROP = {'name': 'drop',
        'data': [os.path.join(BASE_DIR, 'drop/*/train/*.hdf5')]}
_TRAIN_FEAT_DROP = {'name': 'train_drop',
        'data': [os.path.join(BASE_DIR, 'drop/*/train_readout/*.hdf5')]}
_TEST_FEAT_DROP = {'name': 'test_drop',
        'data': [os.path.join(NEW_BASE_DIR, '*drop*/*redyellow/*.hdf5')]}

_TRAIN_LINKING = {'name': 'linking',
        'data': [os.path.join(BASE_DIR, 'linking/*/train/*.hdf5')]}
_TRAIN_FEAT_LINKING = {'name': 'train_linking',
        'data': [os.path.join(BASE_DIR, 'linking/*/train_readout/*.hdf5')]}
_TEST_FEAT_LINKING = {'name': 'test_linking',
        'data': [os.path.join(NEW_BASE_DIR, '*linking*/*redyellow/*.hdf5')]}

_TRAIN_ROLLSLIDE = {'name': 'rollslide',
        'data': [os.path.join(BASE_DIR, 'rollingSliding/*/train/*.hdf5')]}
_TRAIN_FEAT_ROLLSLIDE = {'name': 'train_rollslide',
        'data': [os.path.join(BASE_DIR, 'rollingSliding/*/train_readout/*.hdf5')]}
_TEST_FEAT_ROLLSLIDE = {'name': 'test_rollslide',
        'data': [os.path.join(NEW_BASE_DIR, '*rollingSliding*/*redyellow/*.hdf5')]}

_TRAIN_TOWERS = {'name': 'towers',
        'data': [os.path.join(BASE_DIR, 'towers/*/train/*.hdf5')]}
_TRAIN_FEAT_TOWERS = {'name': 'train_towers',
        'data': [os.path.join(BASE_DIR, 'towers/*/train_readout/*.hdf5')]}
_TEST_FEAT_TOWERS = {'name': 'test_towers',
        'data': [os.path.join(NEW_BASE_DIR, '*towers*/*redyellow/*.hdf5')]}

# Spaces
SEEDS = list(range(_NUM_SEEDS))

TRAIN_DATA = get_all_subsets([
        _TRAIN_CLOTH,
        _TRAIN_COLLISION,
        _TRAIN_CONTAINMENT,
        _TRAIN_DOMINOES,
        _TRAIN_DROP,
        _TRAIN_LINKING,
        _TRAIN_ROLLSLIDE,
        _TRAIN_TOWERS,
        ])

TRAIN_FEAT_DATA = [
        _TRAIN_FEAT_CLOTH,
        _TRAIN_FEAT_COLLISION,
        _TRAIN_FEAT_CONTAINMENT,
        _TRAIN_FEAT_DOMINOES,
        _TRAIN_FEAT_DROP,
        _TRAIN_FEAT_LINKING,
        _TRAIN_FEAT_ROLLSLIDE,
        _TRAIN_FEAT_TOWERS,
        ]

TEST_FEAT_DATA = [
        _TEST_FEAT_CLOTH,
        _TEST_FEAT_COLLISION,
        _TEST_FEAT_CONTAINMENT,
        _TEST_FEAT_DOMINOES,
        _TEST_FEAT_DROP,
        _TEST_FEAT_LINKING,
        _TEST_FEAT_ROLLSLIDE,
        _TEST_FEAT_TOWERS,
        ]

METRICS_DATA = [
        (_TRAIN_FEAT_CLOTH, _TEST_FEAT_CLOTH),
        (_TRAIN_FEAT_COLLISION, _TEST_FEAT_COLLISION),
        (_TRAIN_FEAT_CONTAINMENT, _TEST_FEAT_CONTAINMENT),
        (_TRAIN_FEAT_DOMINOES, _TEST_FEAT_DOMINOES),
        (_TRAIN_FEAT_DROP, _TEST_FEAT_DROP),
        (_TRAIN_FEAT_LINKING, _TEST_FEAT_LINKING),
        (_TRAIN_FEAT_ROLLSLIDE, _TEST_FEAT_ROLLSLIDE),
        (_TRAIN_FEAT_TOWERS, _TEST_FEAT_TOWERS),
        ]

# TRAIN_DATA = TRAIN_DATA[:8] # single scenario
# TRAIN_DATA = TRAIN_DATA[8:16] # leave-one-out
# TRAIN_DATA = TRAIN_DATA[16:] #  all
print(TRAIN_DATA)

SPACE = construct_data_spaces(SEEDS, TRAIN_DATA, TRAIN_FEAT_DATA, TEST_FEAT_DATA, METRICS_DATA)
