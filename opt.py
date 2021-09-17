import os
import getpass
import itertools
import traceback
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials

from physopt.models import get_Objective
from physopt.data import build_data_spaces
from physopt.search.grid_search import suggest
from physion.data.data_space import get_data_space

NO_PARAM_SPACE = hp.choice('dummy', [0])

def arg_parse():
    parser = argparse.ArgumentParser(description='Large-scale physics prediction')

    parser.add_argument('-D', '--data', required=True,
            help='Check "physopt/data/__init__.py" for options', type=str)
    parser.add_argument('-M', '--model', required=True,
            help='Check "physopt/models/__init__.py" for options', type=str)
    parser.add_argument('-O', '--output', default='/home/{}/physopt/',
            help='output directory', type=str)
    parser.add_argument('--host', default='localhost', help='mongo/postgres host', type=str)
    parser.add_argument('--mongo_port', default='25555', help='mongo port', type=str)
    parser.add_argument('--postgres_port', default='5432', help='postgres port', type=str)
    parser.add_argument('--dbname', default='local', help='mongodb, postgres database and s3 bucket name, if not "local"', type=str)
    parser.add_argument('--num_threads', default=1, help='number of parallel threads', type=int)
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return parser.parse_args()

def get_output_directory(output_dir):
    output_dir = output_dir.format(getpass.getuser()) # fill in current username into path
    return output_dir


def get_mongo_path(host, port, database):
    return 'mongo://{0}:{1}/{2}/jobs'.format(host, port, database)


def get_exp_key(model, seed, dynamics_name, readout_name, suffix=''):
    return '{0}_{1}_{2}_{3}_{4}'.format(model, seed, dynamics_name, readout_name, suffix)

class OptimizationPipeline():
    def __init__(self, args):
        self.pool = Pool(args.num_threads) if args.num_threads > 1 else None
        self.dbname = args.dbname
        self.host =  args.host
        self.mongo_port = args.mongo_port
        self.postgres_port = args.postgres_port
        self.data_spaces  = build_data_spaces(get_data_space, args.data)
        self.model = args.model
        self.output_dir = get_output_directory(args.output)
        self.debug = args.debug

    def __del__(self):
        self.close()

    def run(self):
        def run_once(data_space): # data_space: list of space tuples, first corresponds to dynamics training and the rest are readout
            seed, dynamics_space, readout_spaces = (data_space['seed'], data_space['dynamics'], data_space['readout'])
            def run_inner(readout_space):
                phase  = 'dynamics' if readout_space is None else 'readout'
                readout_name = 'none' if readout_space is None else readout_space['name']
                exp_key = get_exp_key(self.model, seed, dynamics_space['name'], readout_name, phase)
                print("Experiment: {0}".format(exp_key))
                if self.dbname == 'local' or self.debug: # don't use MongoTrials when debugging
                    trials = Trials()
                else:
                    mongo_path = get_mongo_path(self.host, self.mongo_port, self.dbname)
                    trials = MongoTrials(mongo_path, exp_key)
                Objective = get_Objective(self.model)
                objective = Objective(
                    self.model, seed, dynamics_space, readout_space, 
                    self.output_dir, phase, self.debug, self.host, self.postgres_port, self.dbname,
                    )

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

            # Train dynamics model
            run_inner(None)

            for readout_space in readout_spaces:
                run_inner(readout_space)
            # TODO: Evaluate readout in parallel, doesn't work yet
            # pool = Pool(len(readout_spaces)) # setup pool to do readouts across scenarios in parallel
            # pool.map(run_inner, itertools.product([seed], [dynamics_space], readout_spaces))
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
