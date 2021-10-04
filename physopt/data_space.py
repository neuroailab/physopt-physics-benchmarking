from  importlib import import_module
from physopt.utils import PRETRAINING_PHASE_NAME, READOUT_PHASE_NAME

def verify_data_spaces(data_spaces):
    def check_inner(space):
        assert space.keys() == {'name', 'train', 'test'}
        assert isinstance(space['name'], str)
        assert isinstance(space['train'], list)
        assert isinstance(space['test'], list)
        
    assert isinstance(data_spaces, list)
    for data_space in data_spaces:
        assert data_space.keys() == {'seed', PRETRAINING_PHASE_NAME, READOUT_PHASE_NAME}
        assert isinstance(data_space['seed'], int)
        assert isinstance(data_space[PRETRAINING_PHASE_NAME], dict)
        check_inner(data_space[PRETRAINING_PHASE_NAME])
        assert isinstance(data_space[READOUT_PHASE_NAME], list)
        for space in data_space[READOUT_PHASE_NAME]:
            check_inner(space)

def add_seed_to_data_spaces(seeds, data_spaces):
    full_data_spaces = [] # full data space with seed
    for seed in seeds:
        for space in data_spaces:
            space = space.copy()
            space['seed'] = seed
            full_data_spaces.append(space)
    return full_data_spaces

def build_data_spaces(module_name, func_name='get_data_spaces', seeds=1, kwargs=None):
    module = import_module(module_name, package=None)
    func = getattr(module, func_name)
    if kwargs is not None:
        data_spaces = func(**kwargs) 
    else:
        data_spaces = func()
    if not isinstance(seeds, (tuple, list)):
        assert isinstance(seeds, int)
        seeds = tuple(range(seeds)) # interprets int value for "seeds" as "num_seeds"
    full_data_spaces = add_seed_to_data_spaces(seeds, data_spaces)
    verify_data_spaces(full_data_spaces)
    return full_data_spaces
