import os
import traceback
import tempfile
import time
from datetime import datetime
from hyperopt import STATUS_FAIL

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
