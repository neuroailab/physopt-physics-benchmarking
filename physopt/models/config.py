from yacs.config import CfgNode as CN

# Default config for models using physopt
_C = CN()

# required params
_C.EPOCHS = 20
_C.BATCH_SIZE = 32
_C.VAL_FREQ = 10 
_C.SAVE_FREQ = 10

# optional params
_C.TRAIN = CN(new_allowed=True)
_C.MODEL = CN(new_allowed=True)
_C.DATA = CN(new_allowed=True)

def get_cfg():
    C =  _C.clone()
    return C
