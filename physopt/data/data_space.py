import importlib

def verify_data_spaces(data_spaces):
    def check_inner(space):
        assert space.keys() == {'name', 'train', 'test'}
        assert isinstance(space['name'], str)
        assert isinstance(space['train'], list)
        assert isinstance(space['test'], list)
        
    assert isinstance(data_spaces, list)
    for data_space in data_spaces:
        assert data_space.keys() == {'seed', 'dynamics', 'readout'}
        assert isinstance(data_space['seed'], int)
        assert isinstance(data_space['dynamics'], dict)
        check_inner(data_space['dynamics'])
        assert isinstance(data_space['readout'], list)
        for space in data_space['readout']:
            check_inner(space)

def build_data_spaces(module_name, cfg_file):
    module = importlib.import_module(module_name, package=None)
    data_spaces = module.get_data_spaces(cfg_file)
    verify_data_spaces(data_spaces)
    return data_spaces
