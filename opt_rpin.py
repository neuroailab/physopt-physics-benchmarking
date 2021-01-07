from hyperopt import fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials
from space.search_space import TRAIN_SPACE, TRAIN_TEST_SPACE
from search.grid_search import suggest
from utils import MultiAttempt
import models.RPIN as model



BASE_DIR = '/mnt/fs4/mrowca/hyperopt/rpin'


def model_objective(args):
    return model.objective(args, BASE_DIR)



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
