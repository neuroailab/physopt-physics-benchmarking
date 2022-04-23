import os
import glob
import socket
import getpass
import yaml
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
from importlib import import_module
from operator import attrgetter

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials

from physopt.objective.utils import PRETRAINING_PHASE_NAME, EXTRACTION_PHASE_NAME, READOUT_PHASE_NAME
from physopt.data_space import build_data_spaces
from physopt.utils.grid_search import suggest
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
    return f'mongo://{host}:{port}/{database}/jobs'

def get_objective(phase, seed, pretraining_space, readout_space, cfg):
    args = [seed, pretraining_space, readout_space, cfg.CONFIG, cfg.PRETRAINING]
    if phase == PRETRAINING_PHASE_NAME:
            Objective = getattr(import_module(cfg.PRETRAINING.OBJECTIVE_MODULE), cfg.PRETRAINING.OBJECTIVE_NAME)
    elif phase == EXTRACTION_PHASE_NAME:
            Objective = getattr(import_module(cfg.EXTRACTION.OBJECTIVE_MODULE), cfg.EXTRACTION.OBJECTIVE_NAME)
            args.append(cfg.EXTRACTION)
    elif phase == READOUT_PHASE_NAME:
            Objective = getattr(import_module(cfg.READOUT.OBJECTIVE_MODULE), cfg.READOUT.OBJECTIVE_NAME)
            args.extend([cfg.EXTRACTION, cfg.READOUT])
    else:
        raise NotImplementedError(f'Unknown phase {phase}')
    objective = Objective(*args)
    return objective

class OptimizationPipeline():
    def __init__(self, cfg):
        self.cfg = cfg
        self.pool = Pool(cfg.NUM_THREADS) if cfg.NUM_THREADS > 1 else None

    def __del__(self):
        self.close()

    def run(self):
        cfg = self.cfg
        data_spaces = build_data_spaces(cfg.DATA_SPACE.MODULE, cfg.DATA_SPACE.FUNC, cfg.DATA_SPACE.SEEDS, cfg.DATA_SPACE.KWARGS)

        def run_once(data_space): # data_space: list of space tuples, first corresponds to dynamics pretraining and the rest are readout
            seed, pretraining_space, readout_spaces = (data_space['seed'], data_space[PRETRAINING_PHASE_NAME], data_space[READOUT_PHASE_NAME])

            def run_inner(phase, readout_space=None):
                objective = get_objective(phase, seed, pretraining_space, readout_space, cfg)

                if cfg.MONGO.DBNAME == 'local' or cfg.CONFIG.DEBUG: # don't use MongoTrials when debugging
                    trials = Trials()
                else:
                    mongo_path = get_mongo_path(cfg.MONGO.HOST, cfg.MONGO.PORT, cfg.MONGO.DBNAME)
                    trials = MongoTrials(mongo_path, exp_key=objective.run_id) # use run_id to ensure it's unique

                best = None
                try:
                    best = fmin(
                        objective,
                        space=NO_PARAM_SPACE, trials=trials,
                        algo=suggest, max_evals=1,
                        )
                except ValueError as e:
                    print("Job died")
                    raise
                return best

            # Pretraining Phase
            if not cfg.SKIP_PRETRAINING:
                run_inner(PRETRAINING_PHASE_NAME, None)

            # Extraction+Readout Phase # TODO: Implement parallel readout evaluation
            for readout_space in readout_spaces:
                run_inner(EXTRACTION_PHASE_NAME, readout_space)
                run_inner(READOUT_PHASE_NAME, readout_space)
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

def resolve_config_file(config_file):
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

def resolve_output_dir(cfg_output, args_output): # updates output dir with the following priority: cmdline, environ, config (if not debug)
    if args_output is not None:
        output_dir = args_output
    elif OUTPUT_ENV_VAR in os.environ:
        output_dir = os.environ[OUTPUT_ENV_VAR]
    else:
        output_dir = cfg_output.format(getpass.getuser()) # fill in current username into path
    print(f'Output dir: {output_dir}')
    return output_dir

def check_cfg(cfg): # TODO: just check that none are none?
    attrs = [
        'DATA_SPACE.MODULE',
        'PRETRAINING.OBJECTIVE_MODULE',
        'PRETRAINING.MODEL_NAME',
        'EXTRACTION.OBJECTIVE_MODULE',
        'READOUT.OBJECTIVE_MODULE',
        ]
    for attr in attrs:
        retriever = attrgetter(attr)
        assert retriever(cfg) is not None, f'{attr} must be set in the config'
    if cfg.EXTRACTION.LOAD_STEP is None:
        cfg.defrost()
        cfg.EXTRACTION.LOAD_STEP = cfg.PRETRAINING.TRAIN_STEPS
        cfg.freeze()
    return True

def get_cfg_from_args(args):
    cfg = get_cfg_defaults()
    config_file  = resolve_config_file(args.config)
    cfg.merge_from_file(config_file)
    cfg.CONFIG.OUTPUT_DIR = resolve_output_dir(cfg.CONFIG.OUTPUT_DIR, args.output)
    if args.debug: # merge debug at end so takes priority
        cfg.merge_from_other_cfg(get_cfg_debug())
    cfg.freeze()
    check_cfg(cfg)
    return cfg

def setup_environment_vars(env_file='environment.yaml'):
    if os.path.isfile(env_file):
        environment = yaml.safe_load(open(env_file, 'rb'))
        hostname = socket.gethostname()
        if hostname in environment:
            assert isinstance(environment[hostname], dict)
            for k,v in environment[hostname].items():
                print(f'Setting environment variable {k} to {v}')
                os.environ[k] = str(v)

def run():
    args = arg_parse()
    cfg = get_cfg_from_args(args)
    pipeline = OptimizationPipeline(cfg)
    pipeline.run()
