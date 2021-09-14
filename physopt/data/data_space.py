import os
from .config import get_cfg_defaults

def build_paths(name, scenarios, filepattern, traindir, testdir):
    return {
        'name': name, 
        'train': [os.path.join(traindir, scenario, filepattern) for scenario in scenarios], 
        'test': [os.path.join(testdir, scenario, filepattern) for scenario in scenarios],
        } 

def get_data_space(cfg_file):
    cfg = get_cfg_defaults()
    if not os.path.isabs(cfg_file): # if not absolute path, looks in this file's dir (i.e. `.../physopt/data/`)
        dirname =  os.path.dirname(__file__)
        cfg_file = os.path.join(dirname, cfg_file)
    cfg.merge_from_file(cfg_file)
    # TODO: add merge debug config?
    cfg.freeze()

    # TODO: constructing the data_spaces could be a bit cleaner
    data_spaces = [] # only dynamics and readout spaces
    for scenario in cfg.SCENARIOS:
        if 'only' in cfg.TRAINING_PROTOCOLS:
            space = {
                'dynamics': build_paths(scenario, [scenario], cfg.FILE_PATTERN, cfg.DYNAMICS_TRAIN_DIR, cfg.DYNAMICS_TEST_DIR),
                'readout': [build_paths(scenario, [scenario], cfg.FILE_PATTERN, cfg.READOUT_TRAIN_DIR, cfg.READOUT_TEST_DIR)],
                }
            data_spaces.append(space)

        if 'abo' in cfg.TRAINING_PROTOCOLS:
            assert len(cfg.SCENARIOS) > 1, 'Must have more than one scenario to do all-but-one protocol.'
            abo_scenarios = [s for s in cfg.SCENARIOS if s is not scenario]
            space = {
                'dynamics': build_paths('no_'+scenario, abo_scenarios, cfg.FILE_PATTERN, cfg.DYNAMICS_TRAIN_DIR, cfg.DYNAMICS_TEST_DIR), # train on all but the scenario
                'readout': [build_paths(scenario, [scenario], cfg.FILE_PATTERN, cfg.READOUT_TRAIN_DIR, cfg.READOUT_TEST_DIR)], # readout on only the single scenario that was left out
                }
            data_spaces.append(space)
        
    if 'all' in cfg.TRAINING_PROTOCOLS:
        assert len(cfg.SCENARIOS) > 1, 'Must have more than one scenario to do all protocol.'
        space = {
            'dynamics': build_paths('all', cfg.SCENARIOS, cfg.FILE_PATTERN, cfg.DYNAMICS_TRAIN_DIR, cfg.DYNAMICS_TEST_DIR), # train on all scenarios
            'readout': [build_paths(scenario, [scenario], cfg.FILE_PATTERN, cfg.READOUT_TRAIN_DIR, cfg.READOUT_TEST_DIR) for scenario in cfg.SCENARIOS], # readout on each scenario individually
            }
        data_spaces.append(space)

    seeds = list(range(cfg.NUM_SEEDS))
    full_data_spaces = [] # full data space with seed
    for seed in seeds:
        for space in data_spaces:
            space = space.copy()
            space['seed'] = seed
            full_data_spaces.append(space)

    return full_data_spaces

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

def build_data_spaces(func, cfg_file):
    if not isinstance(func, list):
        func = [func]
    
    data_spaces = []
    for f in func:
        data_spaces.extend(f(cfg_file))
    verify_data_spaces(data_spaces)
    return data_spaces
