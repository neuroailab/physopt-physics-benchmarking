from yacs.config import CfgNode as CN

_C = CN()

# Data Provider Params
_C.DATA = CN(new_allowed=True) # placeholder, should be overwritten by dataset specific config

def get_data_cfg(subsets, debug=False):
    C = _C.clone()
    C.DATA = get_tdw_cfg(subsets)
    if debug:
        C.DATA.TRAIN_SIZE = 100
        C.DATA.TEST_SIZE = 10
        C.DATA.SHIFTS = (30, 1024, 16)
    return C

def get_tdw_cfg(subsets):
    D = None
    for subset in subsets:
         D = merge_tdw_cfg(D, get_subset_cfg(subset))
    return D

def merge_tdw_cfg(C1, C2):
    if C1 is None:
        return C2.clone()
    C = CN()
    C.LABEL_KEY = _merge_label_key(C1.LABEL_KEY, C2.LABEL_KEY)
    C.SHIFTS = _merge_shifts(C1.SHIFTS, C2.SHIFTS)
    C.TRAIN_SIZE = _merge_size(C1.TRAIN_SIZE, C2.TRAIN_SIZE)
    C.TEST_SIZE = _merge_size(C1.TEST_SIZE, C2.TEST_SIZE)
    return C

def _merge_label_key(lk1, lk2):
    if lk1 == lk2: 
        return lk1
    else: # return generic label_key if they don't match (multiple datasets)
        return 'object_data'

def _merge_shifts(s1, s2):
    funcs = (min, max, min) # (start, end, stride)
    return tuple(f(_s1, _s2) for f,_s1, _s2 in zip(funcs, s1, s2))

def _merge_size(s1, s2):
    return s1 + s2

def get_subset_cfg(name):
    if 'collide' in name:
        return get_collide_cfg()
    elif 'tower' in name:
        return get_tower_cfg()
    elif 'contain' in name:
        return get_contain_cfg()
    elif 'cloth' in name:
        return get_cloth_cfg()
    elif 'roll' in name:
        return get_rollslide_cfg()
    elif 'slide' in name:
        return get_rollslide_cfg()
    else:
        raise ValueError('Unknown config for name: {}'.format(name))

def get_collide_cfg():
    DATA = CN()
    DATA.LABEL_KEY = 'is_colliding_dynamic'
    DATA.SHIFTS = (30, 1024, 1) # TODO: add test shifts?
    DATA.TRAIN_SIZE = 32470
    DATA.TEST_SIZE = 62
    return DATA

def get_tower_cfg():
    DATA = CN()
    DATA.LABEL_KEY = 'is_stable'
    DATA.SHIFTS = (0, 1024, 1) # TODO: add test shifts?
    DATA.TRAIN_SIZE = 63360
    DATA.TEST_SIZE = 384
    return DATA

def get_contain_cfg():
    DATA = CN()
    DATA.LABEL_KEY = 'object_data'
    DATA.SHIFTS = (30, 1024, 1) # TODO: add test shifts?
    DATA.TRAIN_SIZE = 36355
    DATA.TEST_SIZE = 28
    return DATA

def get_cloth_cfg():
    DATA = CN()
    DATA.LABEL_KEY = 'object_category'
    DATA.SHIFTS = (0, 1024, 1) # TODO: add test shifts?
    DATA.TRAIN_SIZE = 63360
    DATA.TEST_SIZE = 192
    return DATA

def get_rollslide_cfg():
    DATA = CN()
    DATA.LABEL_KEY = 'is_rolling'
    DATA.SHIFTS = (32, 1024, 1) # TODO: add test shifts?
    DATA.TRAIN_SIZE = 74572 // 4
    DATA.TEST_SIZE = 320 // 4
    return DATA

if __name__ == '__main__':
    # test merge tdw cfg
    D = merge_tdw_cfg(None, get_collide_cfg())
    print('Added collide\n', D)
    D = merge_tdw_cfg(D, get_cloth_cfg())
    print('Added cloth\n', D)
    D = merge_tdw_cfg(D, get_rollslide_cfg())
    print('Added rollside\n', D)
