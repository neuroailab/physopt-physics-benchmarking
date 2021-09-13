import os
import logging
import errno
import traceback
import tempfile
import time
import torch
from datetime import datetime
import numpy as np
import mlflow
import pickle
from hyperopt import STATUS_OK, STATUS_FAIL
from physopt.metrics.test_metrics import run_metrics, write_metrics

MAX_RUN_TIME = 86400 * 2 # 2 days in seconds

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
            model,
            seed,
            dynamics_space,
            readout_space,
            output_dir,
            mode,
            debug,
            max_run_time=MAX_RUN_TIME,
            ):
        self.model = model
        self.seed = seed
        self.dynamics_space = dynamics_space
        self.dynamics_name = dynamics_space['name']
        self.readout_space = readout_space
        self.readout_name = None if readout_space is None else readout_space['name']
        self.output_dir = output_dir
        self.mode = mode
        self.model_dir = get_model_dir(self.output_dir, self.dynamics_name, self.seed)
        self.model_file = os.path.join(self.model_dir, 'model.pt')
        self.train_feature_file = get_feature_file(self.model_dir, self.readout_name, 'train') # TODO: consolidate into feature_dir?
        self.test_feature_file = get_feature_file(self.model_dir, self.readout_name, 'test')
        self.metrics_file = get_metrics_file(self.model_dir, self.readout_name)
        self.debug = debug
        self.max_run_time = max_run_time

        self.experiment_name = self.get_experiment_name()
        self.run_name = self.get_run_name()
        self.cfg = self.get_config()
        self.model = self.get_model()
        self.model = self.load_model()


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

    def get_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    def get_experiment_name(self):
        return self.model

    def get_run_name(self):
        to_join = [self.mode, str(self.seed), self.dynamics_name]
        if self.readout_name is not None:
            to_join.append(self.readout_name)
        return '{' + '}_{'.join(to_join) + '}'

    def train_step(self, data):
        raise NotImplementedError

    def val_step(self, data):
        raise NotImplementedError

    def extract_feat_step(self, data):
        raise NotImplementedError

    def get_dataloader(self, datapaths, train, shuffle): # returns object that can be iterated over for batches of data
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params({
            'seed': self.seed,
            })
        mlflow.log_params({f'dynamics_{k}':v for k,v in self.dynamics_space.items()})
        if self.readout_space is not None:
            mlflow.log_params({f'readout_{k}':v for k,v in self.readout_space.items()})

        if self.mode == 'dynamics':  # run model training
            self.dynamics()
        elif self.mode == 'readout':# extract features, then train and test readout
            self.readout() 
        else:
            raise NotImplementedError

        mlflow.end_run()

        ret = {
                'loss': 0.0,
                'status': STATUS_OK,
                'model': self.model,
                'seed': self.seed,
                'output_dir': self.output_dir,
                'mode': self.mode,
                'model_dir': self.model_dir,
                }

        return ret

    def dynamics(self):
        trainloader = self.get_dataloader(self.dynamics_space['train'], train=True, shuffle=True)
        best_loss = 1e9
        for epoch in range(self.cfg.EPOCHS): 
            logging.info('Starting epoch {}/{}'.format(epoch+1, self.cfg.EPOCHS))
            running_loss = 0.
            for i, data in enumerate(trainloader):
                loss = self.train_step(data)
                running_loss += loss
                avg_loss = running_loss/(i+1)
                print(avg_loss)
            mlflow.log_metric(key='train_loss', value=avg_loss, step=epoch)

            # perform validation step after every epoch
            valloader = self.get_dataloader(self.dynamics_space['test'], train=True, shuffle=False) # Train since this is part of dynamics training
            val_losses = []
            for i, data in enumerate(valloader):
                val_losses.append(self.val_step(data))
            mlflow.log_metric(key='val_loss', value=np.mean(val_losses), step=epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            logging.info('Saving model with loss {} at epoch {}'.format(best_loss, epoch))
            self.save_model()

    def readout(self):
        assert os.path.isfile(self.model_file), 'No model ckpt found, cannot extract features'

        trainloader = self.get_dataloader(self.readout_space['train'], train=False, shuffle=False)
        self.extract_feats(trainloader, self.train_feature_file)
        testloader = self.get_dataloader(self.readout_space['test'], train=False, shuffle=False)
        self.extract_feats(testloader, self.test_feature_file)

        self.compute_metrics()

    def extract_feats(self, dataloader, feature_file):
        extracted_feats = []
        for i, data in enumerate(dataloader):
            output = self.extract_feat_step(data)
            extracted_feats.append(output)
        pickle.dump(extracted_feats, open(feature_file, 'wb')) 
        print('Saved features to {}'.format(feature_file))

    def compute_metrics(self):
        logging.info('\n\n{}\nStart Compute Metrics:'.format('*'*80))
        protocols = ['observed', 'simulated', 'input']
        results = []
        for protocol in protocols:
            result = run_metrics(
                self.seed,
                self.train_feature_file,
                self.test_feature_file, 
                protocol, 
                grid_search_params=None if self.debug else {'C': np.logspace(-8, 8, 17)},
                )
            result = {'result': result}
            results.append(result)
            mlflow.log_metrics({
                'train_acc_'+protocol: result['result']['train_accuracy'], 
                'test_acc_'+protocol: result['result']['test_accuracy']
                }) # TODO: cleanup and log other info too
            mlflow.log_artifact(self.train_feature_file, artifact_path='features')
            mlflow.log_artifact(self.test_feature_file, artifact_path='features')

            # Write every iteration to be safe
            write_metrics(self.metrics_file, self.seed, self.dynamics_name,
                    self.train_feature_file, self.test_feature_file, self.model_dir, results) # TODO: log artifact

class PytorchPhysOptObjective(PhysOptObjective):
    def __init__(self, *args, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # must set device first since used in get_model, called in super
        super().__init__(*args, **kwargs)
        self.init_seed()

    def load_model(self):
        if os.path.isfile(self.model_file): # load existing model ckpt TODO: add option to disable reloading
            self.model.load_state_dict(torch.load(self.model_file))
            logging.info('Loaded existing ckpt')
        else:
            torch.save(self.model.state_dict(), self.model_file) # save initial model
            logging.info('No model found, saved initial model')
        return self.model

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)
        logging.info('Saved model checkpoint to: {}'.format(self.model_file))

    def init_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

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
