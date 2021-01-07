import time
from hyperopt import STATUS_FAIL

class MultiAttempt():
    def __init__(self, func, max_attempts=10):
        self.func = func
        self.max_attempts = max_attempts

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
                print("Error:", e)
                print("Optimization attempt %d for func failed: %s" % \
                        (num_attempts + 1, self.func.__name__))

        return results
