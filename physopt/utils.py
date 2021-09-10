import os
import logging
import errno
import traceback
import tempfile
import time
from datetime import datetime
from hyperopt import STATUS_OK, STATUS_FAIL
from physopt.metrics.physics.test_metrics import * # TODO

MAX_RUN_TIME = 86400 * 2 # 2 days

class MultiAttempt():
    def __init__(self, func, max_attempts=10):
        self.func = func
        self.max_attempts = max_attempts
        self.exp_key = self.get_exp_key(func)
        self.log_file = self.create_log_file_path(func)


    def __call__(self, *args, **kwargs):
        for num_attempts in range(self.max_attempts):
            results = {
                    'loss': 2*10000,
                    'status': STATUS_FAIL,
                    'time': time.time(),
                    'num_attempts': num_attempts + 1,
                    }

            try:
                results.update(self.func(*args, **kwargs))
                break

            except Exception as e:
                print("Optimization attempt %d for func failed: %s" % \
                        (num_attempts + 1, self.func))
                self.log_error(e)

        return results


    def get_exp_key(self, func):
        return func.exp_key if hasattr(func, 'exp_key') \
                else 'unknown_experiment'


    def create_log_file_path(self, func):
        output_dir = func.model_dir if hasattr(func, 'model_dir') \
                else tempfile.NamedTemporaryFile().name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '%s_log.txt' % self.exp_key
        return os.path.join(output_dir, log_name)


    def log_error(self, error):
        with open(self.log_file, 'a+') as f:
            f.write('=============================================================\n')
            f.write('Time: %s | Experiment: %s\n' % (datetime.now(), self.exp_key))
            f.write('\n%s\n\n' % str(error))
            f.write(traceback.format_exc())
            f.write('\n\n')
        print("Stack trace written to %s" % self.log_file)



class PhysOptObjective():
    def __init__(self,
            exp_key,
            seed,
            dynamics_data,
            readout_data,
            output_dir,
            mode,
            debug,
            max_run_time=MAX_RUN_TIME,
            ):
        self.exp_key = exp_key
        self.seed = seed
        self.dynamics_data = dynamics_data
        self.dynamics_name = dynamics_data['name']
        self.readout_data = readout_data
        self.readout_name = None if readout_data is None else readout_data['name']
        self.output_dir = output_dir
        self.mode = mode
        self.model_dir = get_model_dir(self.output_dir, self.dynamics_name, self.seed)
        self.model_file = os.path.join(self.model_dir, 'model.pt')
        self.train_feature_file = get_feature_file(self.model_dir, self.readout_name, 'train')
        self.test_feature_file = get_feature_file(self.model_dir, self.readout_name, 'test')
        self.metrics_file = get_metrics_file(self.model_dir, self.readout_name)
        self.debug = debug
        self.max_run_time = max_run_time

        # setup logging TODO
        logging.root.handlers = [] # necessary to get handler to work
        logging.basicConfig(
            handlers=[
                logging.FileHandler(os.path.join(self.model_dir, 'output.log')),
                logging.StreamHandler(),
                ],
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.DEBUG if self.debug else logging.INFO,
            )

    def __call__(self, *args, **kwargs):
        if self.mode == 'dynamics':  # run model training
            self.dynamics()
        elif self.mode == 'readout':# extract features, then train and test readout
            self.readout() 
        else:
            raise NotImplementedError

        ret = {
                'loss': 0.0,
                'status': STATUS_OK,
                'exp_key': self.exp_key,
                'seed': self.seed,
                'output_dir': self.output_dir,
                'mode': self.mode,
                'model_dir': self.model_dir,
                }

        for k in ['feat_data', 'train_feat_data', 'test_feat_data',
                'feature_file', 'train_feature_file', 'test_feature_file', 'metrics_file']: # TODO
            if hasattr(self, k):
                ret[k] = getattr(self, k)

        return ret

    def compute_metrics(self):
        logging.info('\n\n{}\nStart Compute Metrics:'.format('*'*80))
        results = []
        for settings in SETTINGS:
            result = run(self.seed, self.train_feature_file,
                    self.test_feature_file, self.readout_name,
                    self.model_dir, settings, 
                    grid_search_params=None if self.debug else {'C': np.logspace(-8, 8, 17)},
                    )
            result = {'result': result}
            result.update(settings) 
            results.append(result)
            mlflow.log_metrics({
                'train_acc_'+settings['type']: result['result']['train_accuracy'], 
                'test_acc_'+settings['type']: result['result']['test_accuracy']
                }) # TODO: cleanup and log other info too
            # Write every iteration to be safe
            write_results(self.metrics_file, self.seed, self.dynamics_name,
                    self.train_feature_file, self.test_feature_file, self.model_dir, results) # TODO: log artifact

def get_model_dir(output_dir, train_name, seed):
    assert train_name is not None
    model_dir = os.path.join(output_dir, train_name, str(seed), 'model/')
    _create_dir(model_dir)
    return model_dir

def get_feature_file(model_dir, test_name, mode):
    if test_name is not None:
        feature_file = os.path.join(model_dir, 'features', test_name, mode+'_feat.pkl')
        _create_dir(feature_file)
        return feature_file


def get_metrics_file(model_dir, test_name):
    if test_name is not None:
        metrics_file = os.path.join(model_dir, 'features', test_name, 'metrics_results.pkl')
        _create_dir(metrics_file)
        return metrics_file

def _create_dir(path): # creates dir from path or filename, if doesn't exist
    dirname, basename = os.path.split(path)
    assert '.' in basename if basename else True, 'Are you sure filename is "{}", or should it be a dir'.format(basename) # checks that basename has file-extension
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

if __name__ == '__main__':
    # ---PhysOptObjective tests---

    # ------dir and files---------
    output_dir = '/mnt/fs4/eliwang/'
    model_dir = get_model_dir(output_dir, 'train_name', 0)
    assert os.path.exists(model_dir)
    print('Model dir: {}'.format(model_dir))

    feature_file = get_feature_file(model_dir, 'test_name')
    assert os.path.exists(os.path.dirname(feature_file))
    print('Feature file: {}'.format(feature_file))

    metrics_file = get_metrics_file(model_dir, 'test_name')
    assert os.path.exists(os.path.dirname(metrics_file))
    print('Metrics file: {}'.format(metrics_file))
