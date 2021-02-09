import os
import itertools
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials
from physopt.utils import MultiAttempt
from physopt.models import get_Objective
from physopt.data import get_data_space
from physopt.search.grid_search import suggest


NO_PARAM_SPACE = hp.choice('dummy', [0])


def arg_parse():
    parser = argparse.ArgumentParser(description='Large-scale physics prediction')

    parser.add_argument('--data', required=True,
            help='data: TDW | DEBUG', type=str)
    parser.add_argument('--model', required=True,
            help='model: RPIN | SVG', type=str)
    parser.add_argument('--output', default='/mnt/fs4/mrowca/hyperopt/',
            help='output directory', type=str)
    parser.add_argument('--host', default='localhost', help='mongo host', type=str)
    parser.add_argument('--port', default='25555', help='mongo port', type=str)
    parser.add_argument('--database', default='physopt', help='mongo database name', type=str)
    parser.add_argument('--num_threads', default=16, help='number of parallel threads', type=int)

    return parser.parse_args()


def get_output_directory(output_dir, model):
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
        extract_feat = False,
        compute_metrics = False,
        ):

    def run_once(data):
        seed, train_data, feat_data = data
        exp_key = get_exp_key(model, seed, train_data, feat_data, exp_key_suffix)
        print("Experiment: {0}".format(exp_key))

        #trials = Trials()
        trials = MongoTrials(mongo_path, exp_key)

        if compute_metrics:
            Objective = get_Objective('metrics')
        else:
            Objective = get_Objective(model)
        objective = Objective(exp_key, seed, train_data, feat_data, output_dir, extract_feat)

        try:
            fmin(#objective,
                    MultiAttempt(objective),
                    space=optimization_space, trials=trials,
                    algo=algo, max_evals=max_evals,
                    )
        except ValueError as e:
            print("Job died: {0}/{1}".format(mongo_path, exp_key))
            print("Error: {0}".format(e))

        return

    # Parallel processing
    if multiprocessing_pool:
        multiprocessing_pool.map(run_once, itertools.product(*data_space))
    # Sequential processing
    else:
        for data in itertools.product(*data_space):
            run_once(data)
    return


def train_model(*args, **kwargs):
    return run(*args, **kwargs)


def extract_features(*args, **kwargs):
    return run(*args, **kwargs, extract_feat = True)


def compute_metrics(*args, **kwargs):
    return run(*args, **kwargs, compute_metrics = True)



class OptimizationPipeline():
    def __init__(self, args = None):
        args = arg_parse() if not args else args

        self.data = get_data_space(args.data)
        self.model = args.model
        self.output_dir = get_output_directory(args.output, args.model)
        self.mongo_path = get_mongo_path(args.host, args.port, args.database)
        self.pool = Pool(args.num_threads) if args.num_threads > 1 else None


    def __del__(self):
        self.close()


    def train_model(self, exp_key_suffix = 'train'):
        print('Training models on data subsets...')
        train_model(self.model, self.data['train_feat'], self.output_dir,
                self.mongo_path, exp_key_suffix = exp_key_suffix,
                multiprocessing_pool = self.pool,
                )
        print('...all models trained!')


    def extract_train_features(self, exp_key_suffix = 'train_feat'):
        print('Extracting train features...')
        extract_features(self.model, self.data['train_feat'], self.output_dir,
                self.mongo_path, exp_key_suffix = exp_key_suffix,
                multiprocessing_pool = self.pool,
                )
        print('...all train features extracted!')


    def extract_test_features(self, exp_key_suffix = 'human_feat'):
        print('Extracting test features...')
        extract_features(self.model, self.data['test_feat'], self.output_dir,
                self.mongo_path, exp_key_suffix = exp_key_suffix,
                multiprocessing_pool = self.pool,
                )
        print('...all test features extracted!')


    def compute_metrics(self, exp_key_suffix = 'metrics'):
        print('Computing metrics...')
        compute_metrics(self.model, self.data['metrics'], self.output_dir,
                self.mongo_path, exp_key_suffix = exp_key_suffix,
                multiprocessing_pool = self.pool,
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
