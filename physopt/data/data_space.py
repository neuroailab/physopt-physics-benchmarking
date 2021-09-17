import importlib

def verify_data_spaces(data_spaces):
    def check_inner(space):
        assert space.keys() == {'name', 'train', 'test'}
        assert isinstance(space['name'], str)
        assert isinstance(space['train'], list)
        assert isinstance(space['test'], list)
        
    assert isinstance(data_spaces, list)
    for data_space in data_spaces:
        assert data_space.keys() == {'seed', 'pretraining', 'readout'}
        assert isinstance(data_space['seed'], int)
        assert isinstance(data_space['pretraining'], dict)
        check_inner(data_space['pretraining'])
        assert isinstance(data_space['readout'], list)
        for space in data_space['readout']:
            check_inner(space)

def build_data_spaces(module_name, func_name='get_data_spaces', cfg_file=None):
    module = importlib.import_module(module_name, package=None)
    func = getattr(module, func_name)
    if cfg_file is not None:
        data_spaces = func(cfg_file) 
    else:
        data_spaces = func()
    verify_data_spaces(data_spaces)
    return data_spaces
