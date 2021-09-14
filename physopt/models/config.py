from yacs.config import CfgNode as CN

# Default config for models using physopt
_C = CN()

# required params
_C.EPOCHS = 20
_C.BATCH_SIZE = 32
_C.LOG_FREQ = 10 # how often to log training loss
_C.VAL_FREQ = 1000 # how often to perform validation
_C.CKPT_FREQ = 1000 # how often to save model

# optional params
_C.TRAIN = CN(new_allowed=True)
_C.MODEL = CN(new_allowed=True)
_C.DATA = CN(new_allowed=True)

def get_cfg():
    C =  _C.clone()
    return C
