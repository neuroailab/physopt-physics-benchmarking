import os
import getpass
import itertools
import traceback
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
import time
from importlib import import_module
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials

from physopt.data import build_data_spaces
from physopt.search.grid_search import suggest

NO_PARAM_SPACE = hp.choice('dummy', [0])

def arg_parse():
    parser = argparse.ArgumentParser(description='Large-scale physics prediction')

    parser.add_argument('--output', default='/home/{}/physopt/', help='output directory', type=str)
    parser.add_argument('--data_module', required=True, type=str)
    parser.add_argument('--data_func', default='get_data_spaces', type=str)
    parser.add_argument('--data_cfg', type=str)
    parser.add_argument('--objective_module', required=True, type=str)
    parser.add_argument('--objective_name', default='Objective', type=str)
    parser.add_argument('--mongo_host', default='localhost', help='mongo host', type=str)
    parser.add_argument('--mongo_port', default='25555', help='mongo port', type=str)
    parser.add_argument('--mongo_dbname', default='local', help='mongodb database name, if not "local"', type=str)
    parser.add_argument('--postgres_host', default='localhost', help='postgres host', type=str)
    parser.add_argument('--postgres_port', default='5444', help='postgres port', type=str)
    parser.add_argument('--postgres_dbname', default='local', help='postgres database and s3 bucket name, if not "local"', type=str)
    parser.add_argument('--num_threads', default=1, help='number of parallel threads', type=int)
    parser.add_argument('-D', '--debug', action='store_true', help='debug mode')
    parser.add_argument('-E', '--experiment_name', default='Default', type=str, help='Name for mlflow experiment')
    parser.add_argument('-T', '--add_timestamp', action='store_true', help='whether to add timestamp to experiment name')
    return parser.parse_args()

def get_output_directory(output_dir):
    output_dir = output_dir.format(getpass.getuser()) # fill in current username into path
    return output_dir


def get_mongo_path(host, port, database):
    return 'mongo://{0}:{1}/{2}/jobs'.format(host, port, database)


def get_exp_key(model, seed, pretraining_name, readout_name, suffix=''):
    return '{0}_{1}_{2}_{3}_{4}'.format(model, seed, pretraining_name, readout_name, suffix)

def get_exp_name(name, add_ts=False, debug=False):
        if debug:
            experiment_name = 'DEBUG'
        elif add_ts:
            experiment_name = name + '_' + time.strftime("%Y%m%d-%H%M%S")
        else:
            experiment_name = name
        return experiment_name

class OptimizationPipeline():
    def __init__(self, args):
        self.pool = Pool(args.num_threads) if args.num_threads > 1 else None
        self.mongo_dbname = args.mongo_dbname
        self.postgres_dbname = args.postgres_dbname
        self.mongo_host =  args.mongo_host
        self.mongo_port = args.mongo_port
        self.postgres_host =  args.postgres_host
        self.postgres_port = args.postgres_port
        self.data_spaces  = build_data_spaces(args.data_module, args.data_func, args.data_cfg)
        self.output_dir = get_output_directory(args.output)
        self.debug = args.debug
        self.Objective = getattr(import_module(args.objective_module), args.objective_name)
        self.experiment_name = get_exp_name(args.experiment_name, args.add_timestamp, args.debug)

    def __del__(self):
        self.close()

    def run(self):
        def run_once(data_space): # data_space: list of space tuples, first corresponds to dynamics pretraining and the rest are readout
            seed, pretraining_space, readout_spaces = (data_space['seed'], data_space['pretraining'], data_space['readout'])
            def run_inner(readout_space):
                phase  = 'pretraining' if readout_space is None else 'readout'
                readout_name = 'none' if readout_space is None else readout_space['name']

                objective = self.Objective(
                    seed, pretraining_space, readout_space, self.output_dir, phase, self.debug, 
                    self.postgres_host, self.postgres_port, self.postgres_dbname, self.experiment_name,
                    )

                exp_key = get_exp_key(objective.model_name, seed, pretraining_space['name'], readout_name, phase)
                print("Experiment: {0}".format(exp_key))
                if self.mongo_dbname == 'local' or self.debug: # don't use MongoTrials when debugging
                    trials = Trials()
                else:
                    mongo_path = get_mongo_path(self.mongo_host, self.mongo_port, self.mongo_dbname)
                    trials = MongoTrials(mongo_path, exp_key)

                try:
                    fmin(
                        objective,
                        space=NO_PARAM_SPACE, trials=trials,
                        algo=suggest, max_evals=1e5,
                        )
                except ValueError as e:
                    print("Job died: {0}".format(exp_key))
                    traceback.print_exc()
                return 

            # Pretraining Phase
            run_inner(None)

            # Readout Phase
            for readout_space in readout_spaces:
                run_inner(readout_space)
            # TODO: Evaluate readout in parallel, doesn't work yet
            # pool = Pool(len(readout_spaces)) # setup pool to do readouts across scenarios in parallel
            # pool.map(run_inner, itertools.product([seed], [pretraining_space], readout_spaces))
            # pool.close()
            # pool.join()

            return

        # Parallel processing
        if self.pool:
            self.pool.map(run_once, self.data_spaces)
        # Sequential processing
        else:
            for data_space in self.data_spaces:
                run_once(data_space)

        self.close()


    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()



if __name__ == '__main__':
    args = arg_parse()
    pipeline = OptimizationPipeline(args)
    pipeline.run()
