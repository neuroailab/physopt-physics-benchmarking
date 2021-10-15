from yacs.config import CfgNode as CN

_C = CN()

# for optimization pipeline
_C.NUM_THREADS = 1

_C.DATA_SPACE = CN()
_C.DATA_SPACE.MODULE = None # required
_C.DATA_SPACE.FUNC = 'get_data_spaces'
_C.DATA_SPACE.SEEDS = (0, ) # list of seeds
_C.DATA_SPACE.KWARGS = CN(new_allowed=True)

_C.MONGO = CN()
_C.MONGO.HOST = 'localhost'
_C.MONGO.PORT = 25555
_C.MONGO.DBNAME = 'local'

_C.PRETRAINING = CN()
_C.PRETRAINING.OBJECTIVE_MODULE = None # required
_C.PRETRAINING.OBJECTIVE_NAME = 'PretrainingObjective'
_C.PRETRAINING.MODEL_NAME = None
_C.PRETRAINING.TRAIN_STEPS = 1000
_C.PRETRAINING.BATCH_SIZE = 32
_C.PRETRAINING.LOG_FREQ = 10
_C.PRETRAINING.VAL_FREQ = 25
_C.PRETRAINING.CKPT_FREQ = 100
_C.PRETRAINING.TRAIN = CN(new_allowed=True)
_C.PRETRAINING.MODEL = CN(new_allowed=True)
_C.PRETRAINING.DATA = CN(new_allowed=True)


_C.EXTRACTION= CN()
_C.EXTRACTION.OBJECTIVE_MODULE = None # required
_C.EXTRACTION.OBJECTIVE_NAME = 'ExtractionObjective'
_C.EXTRACTION.LOAD_STEP = None # which pretraining ckpt to extract readout features from

_C.READOUT= CN()
_C.READOUT.OBJECTIVE_MODULE = 'physion.objective'
_C.READOUT.OBJECTIVE_NAME = 'PhysionReadoutObjective'
_C.READOUT.MODEL = CN(new_allowed=True)
_C.READOUT.MODEL.LOGSPACE = (-4, 4, 9)
_C.READOUT.MODEL.CV = 5 # number of folds
_C.READOUT.MODEL.MAX_ITER = 1000
_C.READOUT.MODEL.NORM_INPUT = True # whether to standardize input features 

# for physopt objective
_C.CONFIG = CN()
_C.CONFIG.DEBUG = False
_C.CONFIG.OUTPUT_DIR = '/home/{}/physopt/'
_C.CONFIG.DELETE_LOCAL = True # delete local files (not artifact store)
_C.CONFIG.EXPERIMENT_NAME = 'Default'
_C.CONFIG.ADD_TIMESTAMP = False # append timestamp to experiment name

_C.CONFIG.POSTGRES = CN()
_C.CONFIG.POSTGRES.HOST = 'localhost'
_C.CONFIG.POSTGRES.PORT = 5444
_C.CONFIG.POSTGRES.DBNAME = 'local'


def get_cfg_defaults():
    return _C.clone()

def get_cfg_debug():
    C = CN()
    C.DATA_SPACE = CN()
    C.DATA_SPACE.SEEDS = (0,) # just use one seed
    C.CONFIG = CN()
    C.CONFIG.DEBUG = True
    C.PRETRAINING = CN()
    C.PRETRAINING.TRAIN_STEPS = 3
    C.PRETRAINING.LOG_FREQ = 1
    C.PRETRAINING.VAL_FREQ = 3
    C.PRETRAINING.CKPT_FREQ = 2
    C.READOUT = CN()
    C.READOUT.MODEL = CN()
    C.READOUT.MODEL.LOGSPACE = (0, 0, 1) # just use one regularization value
    C.READOUT.MODEL.CV = 2 
    return C.clone()
