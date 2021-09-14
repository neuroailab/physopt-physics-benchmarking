from yacs.config import CfgNode as CN

# Default config for models using physopt
_C = CN()

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 20
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR = 1e-3
_C.TRAIN.VAL_FREQ = 10 
_C.TRAIN.SAVE_FREQ = 10

_C.MODEL = CN(new_allowed=True)

_C.DATA = CN(new_allowed=True)

def get_cfg():
    C =  _C.clone()
    return C
