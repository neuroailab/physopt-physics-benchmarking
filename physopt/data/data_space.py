import os
from .config import get_cfg_defaults

def build_paths(name, scenarios, filepattern, traindir, testdir):
    return {
        'name': name, 
        'train': [os.path.join(traindir, scenario, filepattern) for scenario in scenarios], 
        'test': [os.path.join(testdir, scenario, filepattern) for scenario in scenarios],
        } 

def get_data_space(
        data_space,
        ):
    cfg = get_cfg_defaults()
    dirname = os.path.dirname(__file__)
    cfg.merge_from_file(os.path.join(dirname, data_space+'.yaml'))
    # TODO: add merge debug config?
    cfg.freeze()
    print(cfg)
    
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
