import os
import errno
import traceback
import tempfile
import time
from datetime import datetime
from hyperopt import STATUS_OK, STATUS_FAIL



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
            train_data,
            feat_data,
            output_dir,
            extract_feat,
            debug,
            ):
        self.exp_key = exp_key
        self.seed = seed
        self.train_data = train_data
        self.output_dir = output_dir
        self.extract_feat = extract_feat
        self.model_dir = self.get_model_dir(self.output_dir,
                self.train_data['name'], self.seed)
        self.debug = debug

        if isinstance(feat_data, dict):
            # feature data space
            self.feat_data = feat_data
            self.feature_file = self.get_feature_file(self.model_dir,
                    self.feat_data['name'])
        else:
            # metrics data space
            self.train_feat_data = feat_data[0]
            self.test_feat_data = feat_data[1]
            self.train_feature_file = self.get_feature_file(self.model_dir,
                    self.train_feat_data['name'])
            self.test_feature_file = self.get_feature_file(self.model_dir,
                    self.test_feat_data['name'])
            self.metrics_file = self.get_metrics_file(self.model_dir,
                    self.test_feat_data['name'])

    @staticmethod
    def get_model_dir(output_dir, train_name, seed):
        model_dir = os.path.join(output_dir, train_name, str(seed), 'model/')
        _create_dir(model_dir)
        return model_dir

    @staticmethod
    def get_feature_file(model_dir, test_name):
        feature_file = os.path.join(model_dir, 'features', test_name, 'feat.pkl')
        _create_dir(feature_file)
        return feature_file


    @staticmethod
    def get_metrics_file(model_dir, test_name):
        metrics_file = os.path.join(model_dir, 'features', test_name, 'metrics_results.pkl')
        _create_dir(metrics_file)
        return metrics_file

    def __call__(self, *args, **kwargs):
        ret = {
                'loss': 0.0,
                'status': STATUS_OK,
                'exp_key': self.exp_key,
                'seed': self.seed,
                'train_data': self.train_data,
                'output_dir': self.output_dir,
                'extract_feat': self.extract_feat,
                'model_dir': self.model_dir,
                }

        for k in ['feat_data', 'train_feat_data', 'test_feat_data',
                'feature_file', 'train_feature_file', 'test_feature_file', 'metrics_file']:
            if hasattr(self, k):
                ret[k] = getattr(self, k)

        return ret

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
    model_dir = PhysOptObjective.get_model_dir(output_dir, 'train_name', 0)
    assert os.path.exists(model_dir)
    print('Model dir: {}'.format(model_dir))

    feature_file = PhysOptObjective.get_feature_file(model_dir, 'test_name')
    assert os.path.exists(os.path.dirname(feature_file))
    print('Feature file: {}'.format(feature_file))

    metrics_file = PhysOptObjective.get_metrics_file(model_dir, 'test_name')
    assert os.path.exists(os.path.dirname(metrics_file))
    print('Metrics file: {}'.format(metrics_file))
