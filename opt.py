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
    parser.add_argument('-O', '--output', default='/mnt/fs4/{}/physopt/',
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


def get_exp_key(model, seed, train_data, feat_data, suffix=''):
    feat_data = feat_data if isinstance(feat_data, dict) else feat_data[-1]
    return '{0}_{1}_{2}_{3}_{4}'.format(model, seed, train_data['name'],
            feat_data['name'], suffix)


def run(
        model,
        data_space,
        output_dir,
        mongo_path,
        exp_key_suffix='',
        optimization_space=NO_PARAM_SPACE,
        algo=suggest,
        max_evals=1e5,
        multiprocessing_pool = None,
        debug=False,
        mongo=False,
        max_run_time=MAX_RUN_TIME,
        mode='train_dynamics',
        ):

    def run_once(data):
        seed, train_data, feat_data = data

        # Do not evaluate across datasets unless trained on all
        if train_data['name'] != 'all':
            if isinstance(feat_data, dict):
                if feat_data['name'] != 'train' and train_data['name'].replace('no_', '') \
                        not in feat_data['name']:
                    return
            else:
                if train_data['name'].replace('no_', '') not in feat_data[0]['name'] or \
                        train_data['name'].replace('no_', '') not in feat_data[1]['name']:
                    return

        exp_key = get_exp_key(model, seed, train_data, feat_data, exp_key_suffix)
        print("Experiment: {0}".format(exp_key))

        if mongo:
            trials = MongoTrials(mongo_path, exp_key)
        else:
            trials = Trials()

        if mode == 'compute_metric': # TODO
            Objective = get_Objective('metrics')
        else:
            Objective = get_Objective(model)
        objective = Objective(exp_key, seed, train_data, feat_data, output_dir,
                mode, debug, max_run_time) # TODO: more flexible args

        try:
            fmin(
                MultiAttempt(objective) if not debug else objective,
                space=optimization_space, trials=trials,
                algo=algo, max_evals=max_evals,
                )
        except ValueError as e:
            print("Job died: {0}/{1}".format(mongo_path, exp_key))
            traceback.print_exc()

        return

    # Model training requires training only once
    if mode == 'train_dynamics':
        data_space = (data_space[0], data_space[1], [{'name': 'train', 'data': []}])

    # Parallel processing
    if multiprocessing_pool:
        multiprocessing_pool.map(run_once, itertools.product(*data_space))
    # Sequential processing
    else:
        for data in itertools.product(*data_space):
            run_once(data)
    return


def train_model(*args, **kwargs):
    return run(*args, **kwargs, mode='train_dynamics')


def extract_features(*args, **kwargs):
    return run(*args, **kwargs, mode='extract_feat')


def compute_metrics(*args, **kwargs):
    return run(*args, **kwargs, mode='compute_metric')



class OptimizationPipeline():
    def __init__(self, args = None):
        args = arg_parse() if not args else args

        self.pool = Pool(args.num_threads) if args.num_threads > 1 else None
        self.data = get_data_space(args.data)
        self.model = args.model
        self.output_dir = get_output_directory(args.output, args.model)
        self.mongo_path = get_mongo_path(args.host, args.port, args.database)
        self.debug = args.debug
        self.mongo = args.mongo
        self.max_run_time = args.max_run_time

    def __del__(self):
        self.close()


    def train_model(self, exp_key_suffix = 'train'):
        print('Training models on data subsets...')
        train_model(self.model, self.data['train_feat'], self.output_dir,
                self.mongo_path, exp_key_suffix = exp_key_suffix,
                multiprocessing_pool = self.pool, debug=self.debug, mongo=self.mongo,
                max_run_time = self.max_run_time,
                )
        print('...all models trained!')


    def extract_train_features(self, exp_key_suffix = 'train_feat'):
        print('Extracting train features...')
        extract_features(self.model, self.data['train_feat'], self.output_dir,
                self.mongo_path, exp_key_suffix = exp_key_suffix,
                multiprocessing_pool = self.pool, debug=self.debug, mongo=self.mongo,
                )
        print('...all train features extracted!')


    def extract_test_features(self, exp_key_suffix = 'human_feat'):
        print('Extracting test features...')
        extract_features(self.model, self.data['test_feat'], self.output_dir,
                self.mongo_path, exp_key_suffix = exp_key_suffix,
                multiprocessing_pool = self.pool, debug=self.debug, mongo=self.mongo,
                )
        print('...all test features extracted!')


    def compute_metrics(self, exp_key_suffix = 'metrics'):
        print('Computing metrics...')
        compute_metrics(self.model, self.data['metrics'], self.output_dir,
                self.mongo_path, exp_key_suffix = exp_key_suffix,
                multiprocessing_pool = self.pool, debug=self.debug, mongo=self.mongo,
                )
        print('...all metrics computed!')


    def run(self):
        self.train_model()
        self.extract_train_features()
        self.extract_test_features()
        self.compute_metrics()
        self.close()


    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()



if __name__ == '__main__':
    args = arg_parse()
    pipeline = OptimizationPipeline(args)
    pipeline.run()
