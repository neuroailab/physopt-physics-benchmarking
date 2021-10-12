from yacs.config import CfgNode as CN

_C = CN()

# for optimization pipeline
_C.OUTPUT_DIR = '/home/{}/physopt/'
_C.NUM_THREADS = 1

_C.DATA_SPACE = CN()
_C.DATA_SPACE.MODULE = None # required
_C.DATA_SPACE.FUNC = 'get_data_spaces'
_C.DATA_SPACE.SEEDS = (0, ) # if int, uses range(seeds) otherwise pass list of seeds
_C.DATA_SPACE.KWARGS = CN(new_allowed=True)

_C.MONGO = CN()
_C.MONGO.HOST = 'localhost'
_C.MONGO.PORT = 25555
_C.MONGO.DBNAME = 'local'

_C.OBJECTIVE = CN()
_C.OBJECTIVE.MODULE = None # required
_C.OBJECTIVE.NAME = 'Objective'

# for physopt objective
_C.CONFIG = CN()
_C.CONFIG.DEBUG = False
_C.CONFIG.DELETE_LOCAL = True # delete local files (not artifact store)
_C.CONFIG.EXPERIMENT_NAME = 'Default'
_C.CONFIG.ADD_TIMESTAMP = False
_C.CONFIG.TRAIN_STEPS = 1000
_C.CONFIG.BATCH_SIZE = 32
_C.CONFIG.LOG_FREQ = 10
_C.CONFIG.VAL_FREQ = 25
_C.CONFIG.CKPT_FREQ = 100
_C.CONFIG.READOUT_LOAD_STEP = None # which pretraining ckpt to extract readout features from

_C.CONFIG.TRAIN = CN(new_allowed=True)
_C.CONFIG.MODEL = CN(new_allowed=True)
_C.CONFIG.DATA = CN(new_allowed=True)
_C.CONFIG.READOUT = CN(new_allowed=True)
_C.CONFIG.READOUT.LOGSPACE: (-4, 4, 9)
_C.CONFIG.READOUT.CV = 5 # number of folds
_C.CONFIG.READOUT.MAX_ITER = 1000
_C.CONFIG.READOUT.NORM_INPUT = True # whether to standardize input features 
_C.CONFIG.READOUT.DO_RESTORE = True # whether to restore readout model, hacky

_C.CONFIG.POSTGRES = CN()
_C.CONFIG.POSTGRES.HOST = 'localhost'
_C.CONFIG.POSTGRES.PORT = 5444
_C.CONFIG.POSTGRES.DBNAME = 'local'


def get_cfg_defaults():
    return _C.clone()

def get_cfg_debug():
    C = CN()
    C.CONFIG = CN()
    C.CONFIG.DEBUG = True
    C.CONFIG.TRAIN_STEPS = 10
    C.CONFIG.LOG_FREQ = 1
    C.CONFIG.VAL_FREQ = 10
    C.CONFIG.CKPT_FREQ = 5
    C.CONFIG.READOUT = CN()
    C.CONFIG.READOUT.NUM_C = 1 # just use one regularization value
    return C.clone()
