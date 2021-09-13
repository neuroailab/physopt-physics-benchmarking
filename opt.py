import os
import getpass
import itertools
import traceback
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials

from physopt.utils import MultiAttempt, MAX_RUN_TIME
from physopt.models import get_Objective
from physopt.data import get_data_space
from physopt.search.grid_search import suggest

NO_PARAM_SPACE = hp.choice('dummy', [0])

def arg_parse():
    parser = argparse.ArgumentParser(description='Large-scale physics prediction')

    parser.add_argument('-D', '--data', required=True,
            help='Check "physopt/data/__init__.py" for options', type=str)
    parser.add_argument('-M', '--model', required=True,
            help='Check "physopt/models/__init__.py" for options', type=str)
    parser.add_argument('-O', '--output', default='/home/{}/physopt/',
            help='output directory', type=str)
    parser.add_argument('--host', default='localhost', help='mongo host', type=str)
    parser.add_argument('--port', default='25555', help='mongo port', type=str)
    parser.add_argument('--database', default='physopt', help='mongo database name', type=str)
    parser.add_argument('--num_threads', default=1, help='number of parallel threads', type=int)
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--mongo', action='store_true', help='whether to use mongo trials')
    parser.add_argument('--max_run_time', default=MAX_RUN_TIME,
            help='Maximum model training time in seconds', type=int)

    return parser.parse_args()

def get_output_directory(output_dir, model):
    output_dir = output_dir.format(getpass.getuser()) # fill in current username into path
    return os.path.join(output_dir, model)


def get_mongo_path(host, port, database):
    return 'mongo://{0}:{1}/{2}/jobs'.format(host, port, database)


def get_exp_key(model, seed, dynamics_name, readout_name, suffix=''):
    return '{0}_{1}_{2}_{3}_{4}'.format(model, seed, dynamics_name, readout_name, suffix)

class OptimizationPipeline():
    def __init__(self, args = None):
        args = arg_parse() if not args else args

        self.pool = Pool(args.num_threads) if args.num_threads > 1 else None
        self.data_spaces  = get_data_space(args.data)
        self.model = args.model
        self.output_dir = get_output_directory(args.output, args.model)
        self.mongo_path = get_mongo_path(args.host, args.port, args.database)
        self.debug = args.debug
        self.mongo = args.mongo
        self.max_run_time = args.max_run_time

    def __del__(self):
        self.close()

    def run(self):
        optimization_space = NO_PARAM_SPACE
        algo = suggest
        max_evals = 1e5
        def run_once(data_space): # data_space: list of space tuples, first corresponds to dynamics training and the rest are readout
            seed, dynamics_data, readout_data = data_space
            def run_inner(data_space):
                seed, dynamics_data, readout_data = data_space
                mode  = 'dynamics' if readout_data is None else 'readout'
                readout_name = 'none' if readout_data is None else readout_data['name']
                exp_key = get_exp_key(self.model, seed, dynamics_data['name'], readout_name, mode)
                print("Experiment: {0}".format(exp_key))
                if self.mongo:
                    trials = MongoTrials(self.mongo_path, exp_key)
                else:
                    trials = Trials()
                Objective = get_Objective(self.model) # TODO: consolidate?
                objective = Objective(self.model, seed, dynamics_data, readout_data, self.output_dir,
                        mode, self.debug, self.max_run_time)

                try:
                    fmin(
                        MultiAttempt(objective) if not self.debug else objective,
                        space=optimization_space, trials=trials,
                        algo=algo, max_evals=max_evals,
                        )
                except ValueError as e:
                    print("Job died: {0}/{1}".format(self.mongo_path, exp_key))
                    traceback.print_exc()
                return 

            # Train dynamics model
            run_inner((seed, dynamics_data, None))

            print(len(readout_data))
            print(readout_data)
            for _rd in readout_data:
                run_inner((seed, dynamics_data, _rd))
            # TODO: Evaluate readout in parallel, doesn't work yet
            # pool = Pool(len(readout_data)) # setup pool to do readouts across scenarios in parallel
            # pool.map(run_inner, itertools.product([seed], [dynamics_data], readout_data))
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
