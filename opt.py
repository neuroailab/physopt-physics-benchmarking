import os
import glob
import getpass
import traceback
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
from importlib import import_module

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials

from physopt.utils import PRETRAINING_PHASE_NAME, READOUT_PHASE_NAME
from physopt.data_space import build_data_spaces
from physopt.search.grid_search import suggest
from physopt.config import get_cfg_defaults, get_cfg_debug

NO_PARAM_SPACE = hp.choice('dummy', [0])
CONFIG_ENV_VAR = 'PHYSOPT_CONFIG_DIR'
OUTPUT_ENV_VAR = 'PHYSOPT_OUTPUT_DIR'

def arg_parse():
    parser = argparse.ArgumentParser(description='Large-scale physics prediction')
    parser.add_argument('-C', '--config', required=True, type=str, help='path to physopt configuration file')
    parser.add_argument('-D', '--debug', action='store_true', help='debug mode')
    parser.add_argument('-O', '--output', type=str, help='Output directory for physopt artifacts')
    return parser.parse_args()

def get_mongo_path(host, port, database):
    return 'mongo://{0}:{1}/{2}/jobs'.format(host, port, database)

class OptimizationPipeline():
    def __init__(self, cfg):
        self.cfg = cfg
        self.pool = Pool(cfg.NUM_THREADS) if cfg.NUM_THREADS > 1 else None

    def __del__(self):
        self.close()

    def run(self):
        cfg = self.cfg
        data_spaces = build_data_spaces(cfg.DATA_SPACE.MODULE, cfg.DATA_SPACE.FUNC, cfg.DATA_SPACE.SEEDS, cfg.DATA_SPACE.KWARGS)
        Objective = getattr(import_module(cfg.OBJECTIVE.MODULE), cfg.OBJECTIVE.NAME)

        def run_once(data_space): # data_space: list of space tuples, first corresponds to dynamics pretraining and the rest are readout
            seed, pretraining_space, readout_spaces = (data_space['seed'], data_space[PRETRAINING_PHASE_NAME], data_space[READOUT_PHASE_NAME])

            def run_inner(readout_space=None):
                objective = Objective(
                    seed, 
                    pretraining_space, 
                    readout_space, 
                    cfg.OUTPUT_DIR,
                    cfg.CONFIG,
                    )

                if cfg.MONGO.DBNAME == 'local' or cfg.CONFIG.DEBUG: # don't use MongoTrials when debugging
                    trials = Trials()
                else:
                    mongo_path = get_mongo_path(cfg.MONGO.HOST, cfg.MONGO.PORT, cfg.MONGO.DBNAME)
                    trials = MongoTrials(mongo_path, exp_key=objective.run_name)

                try:
                    best = fmin(
                        objective,
                        space=NO_PARAM_SPACE, trials=trials,
                        algo=suggest, max_evals=1,
                        )
                except ValueError as e:
                    print("Job died: {0}".format(objective.run_name))
                    traceback.print_exc()
                return 

            # Pretraining Phase
            run_inner(None)

            # Readout Phase # TODO: Implement parallel readout evaluation
            for readout_space in readout_spaces:
                run_inner(readout_space)
            return

        # Parallel processing
        if self.pool:
            self.pool.map(run_once, data_spaces)
        # Sequential processing
        else:
            for data_space in data_spaces:
                run_once(data_space)

        self.close()


    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()

class MissingEnvironmentVariable(Exception):
    pass

def resolve_config_path(config_file):
    if not os.path.isabs(config_file): # skip search if abs path provided
        try:
            config_dir = os.environ[CONFIG_ENV_VAR]
        except KeyError:
            raise MissingEnvironmentVariable(f'Must set environment variable "{CONFIG_ENV_VAR}" if using relative path for config file')
        assert os.path.isdir(config_dir), f'Directory not found: {config_dir}'
        print(f'Searching for config in {config_dir}')
        pathname = os.path.join(config_dir, '**', config_file)
        files = glob.glob(pathname, recursive=True)
        assert len(files) > 0, f'No config file found matching {pathname}.'
        assert len(files) == 1, f'Found multiple ({len(files)}) files that match {pathname}'
        config_file = files[0]
    assert os.path.isfile(config_file), f'File not found: {config_file}'
    print(f'Found config file: {config_file}')
    return config_file

def resolve_output_dir(output_dir, args): # updates output dir with the following priority: cmdline, environ, config (if not debug)
    if args.output is not None:
        output_dir = args.output
    elif OUTPUT_ENV_VAR in os.environ:
        output_dir = os.environ[OUTPUT_ENV_VAR]
    else:
        output_dir = output_dir.format(getpass.getuser()) # fill in current username into path
    print(f'Output dir: {output_dir}')
    return output_dir

def check_cfg(cfg):
    assert cfg.DATA_SPACE.MODULE is not None, 'DATA_SPACE.MODULE must be set in the config'
    assert cfg.OBJECTIVE.MODULE is not None, 'OBJECTIVE.MODULE must be set in the config' 
    return True

def get_cfg_from_args(args):
    cfg = get_cfg_defaults()
    config_file  = resolve_config_path(args.config)
    cfg.merge_from_file(config_file)
    cfg.OUTPUT_DIR = resolve_output_dir(cfg.OUTPUT_DIR, args)
    if args.debug: # merge debug at end so takes priority
        cfg.merge_from_other_cfg(get_cfg_debug())
    cfg.freeze()
    check_cfg(cfg)
    return cfg

if __name__ == '__main__':
    args = arg_parse()
    cfg = get_cfg_from_args(args)
    pipeline = OptimizationPipeline(cfg)
    pipeline.run()
