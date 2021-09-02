from yacs.config import CfgNode as CN

_C = CN()

_C.DYNAMICS_TRAIN_DIR = None
_C.READOUT_TRAIN_DIR = None
_C.READOUT_TEST_DIR = None
_C.FILE_PATTERN = '*.hdf5'

_C.SCENARIOS = ['Collide', 'Contain', 'Dominoes', 'Drape', 'Drop', 'Link', 'Roll', 'Support']
_C.NUM_SEEDS = 1
_C.TRAINING_PROTOCOLS = ['all', 'abo', 'only']

def get_cfg_defaults():
  return _C.clone()

