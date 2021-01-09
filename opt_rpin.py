from hyperopt import fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials
from space.search_space import TRAIN_SPACE, TRAIN_TEST_SPACE, HUMAN_TEST_SPACE, METRICS_SPACE
from search.grid_search import suggest
from utils import MultiAttempt
import models.RPIN as model
import physics_metrics.test_metrics as test_metrics



BASE_DIR = '/mnt/fs4/mrowca/hyperopt/rpin'


def model_objective(args):
    return model.objective(args, BASE_DIR)


def metrics_objective(args):
    return test_metrics.objective(args, BASE_DIR)



if __name__ == '__main__':
    #trials = Trials()
    trials = MongoTrials('mongo://localhost:25555/rpin/jobs', exp_key='train_exp1')
    best = fmin(
            #objective,
            MultiAttempt(model_objective),
            TRAIN_SPACE,
            trials=trials,
            algo=suggest,
            max_evals=1e5,
            )
    print('All models trained')

    #trials = Trials()
    trials = MongoTrials('mongo://localhost:25555/rpin/jobs', exp_key='train_feat_exp1')
    best = fmin(
            MultiAttempt(model_objective),
            TRAIN_TEST_SPACE,
            trials=trials,
            algo=suggest,
            max_evals=1e5,
            )
    print('All train features extracted')

    #trials = Trials()
    trials = MongoTrials('mongo://localhost:25555/rpin/jobs', exp_key='test_feat_exp1')
    best = fmin(
            MultiAttempt(model_objective),
            HUMAN_TEST_SPACE,
            trials=trials,
            algo=suggest,
            max_evals=1e5,
            )
    print('All test features extracted')

    #trials = Trials()
    trials = MongoTrials('mongo://localhost:25555/rpin/jobs', exp_key='metrics_exp1')
    best = fmin(
            MultiAttempt(metrics_objective, 1),
            METRICS_SPACE,
            trials=trials,
            algo=suggest,
            max_evals=1e5,
            )
    print('All metrics computed')
