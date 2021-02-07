import os
import itertools
from pathos.multiprocessing import ProcessingPool as Pool
import argparse
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials
from utils import MultiAttempt

from space.tdw_space import TRAIN_FEAT_SPACE, HUMAN_FEAT_SPACE, METRICS_SPACE
from search.grid_search import suggest
import metrics.physics.test_metrics as test_metrics

import pdb

NO_PARAM_SPACE = hp.choice('dummy', [0])


def arg_parse():
    parser = argparse.ArgumentParser(description='Large-scale physics prediction')

    parser.add_argument('-m', '--model', required=True,
            help='model: RPIN | SVG', type=str)
    parser.add_argument('--output', default='/mnt/fs4/eliwang/hyperopt/',
            help='output directory', type=str)
    parser.add_argument('--host', default='localhost', help='mongo host', type=str)
    parser.add_argument('--port', default='25555', help='mongo port', type=str)
    parser.add_argument('--database', default='physopt', help='mongo database name', type=str)
    parser.add_argument('--num_threads', default=1, help='number of parallel threads', type=int)

    return parser.parse_args()


def get_Objective(model):
    # import from _models to avoid collision with models in hubconf.py
    if model == 'metrics':
        return test_metrics.Objective
    elif model == 'RPIN':
        from _models.RPIN import Objective as RPINObjective
        return RPINObjective
    elif model == 'SVG':
        from _models.SVG import VGGObjective as SVGObjective
        return SVGObjective
    elif model == 'CSWM':
        from _models.CSWM import Objective as CSWMObjective
        return CSWMObjective
    elif model == 'SVG_FROZEN':
        from _models.SVG_FROZEN import Objective as SVGFObjective
        return SVGFObjective
    elif model == 'OP3':
        from _models.OP3 import Objective as OP3Objective
        return OP3Objective
    else:
        raise ValueError('Unknown model: {0}'.format(model))


def get_output_directory(output_dir, model):
    return os.path.join(output_dir, model)


def get_mongo_path(host, port, database):
    return 'mongo://{0}:{1}/{2}/jobs'.format(host, port, database)


def get_exp_key(seed, train_data, feat_data, suffix=''):
    feat_data = feat_data if isinstance(feat_data, dict) else feat_data[-1]
    return '{0}_{1}_{2}_{3}'.format(seed, train_data['name'], feat_data['name'], suffix)


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
        if isinstance(train_data, tuple):
            print(train_data)
        exp_key = get_exp_key(seed, train_data, feat_data, exp_key_suffix)
        print("Experiment: {0}".format(exp_key))

        trials = Trials()
        # trials = MongoTrials(mongo_path, exp_key)

        if compute_metrics:
            Objective = get_Objective('metrics')
        else:
            Objective = get_Objective(model)
        objective = Objective(exp_key, seed, train_data, feat_data, output_dir, extract_feat)

        try:
            fmin(#objective,
                    # MultiAttempt(objective),
                    objective,
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


def main():
    args = arg_parse()

    output_dir = get_output_directory(args.output, args.model)
    mongo_path = get_mongo_path(args.host, args.port, args.database)
    pool = Pool(args.num_threads) if args.num_threads > 1 else None

    print('Training models on data subsets...')
    train_model(args.model, TRAIN_FEAT_SPACE, output_dir,
            mongo_path, exp_key_suffix = 'train',
            multiprocessing_pool = pool,
            )
    print('...all models trained!')

    print('Extracting train features...')
    extract_features(args.model, TRAIN_FEAT_SPACE, output_dir,
            mongo_path, exp_key_suffix = 'train_feat',
            multiprocessing_pool = pool,
            )
    print('...all train features extracted!')

    print('Extracting test features...')
    extract_features(args.model, HUMAN_FEAT_SPACE, output_dir,
            mongo_path, exp_key_suffix = 'human_feat',
            multiprocessing_pool = pool,
            )
    print('...all test features extracted!')

    print('Computing metrics...')
    compute_metrics(args.model, METRICS_SPACE, output_dir,
            mongo_path, exp_key_suffix = 'metrics',
            multiprocessing_pool = pool,
            )
    print('...all metrics computed!')

    if pool:
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
