import os
import abc
import logging
import errno
import traceback
import time
import numpy as np
import mlflow
import pickle
import psycopg2
import boto3

from hyperopt import STATUS_OK, STATUS_FAIL
from physopt.metrics.test_metrics import run_metrics, write_metrics
from physopt.models.config import get_cfg

class PhysOptObjective(metaclass=abc.ABCMeta):
    def __init__(
            self,
            seed,
            pretraining_space,
            readout_space,
            output_dir,
            phase,
            debug,
            host,
            port,
            dbname,
            ):
        self.seed = seed
        self.pretraining_space = pretraining_space
        self.pretraining_name = pretraining_space['name']
        self.readout_space = readout_space
        self.readout_name = None if readout_space is None else readout_space['name']
        self.output_dir = output_dir
        self.phase = phase
        self.debug = debug
        self.model_dir = get_model_dir(self.output_dir, self.model_name, self.pretraining_name, self.seed, self.debug)
        self.model_file = os.path.join(self.model_dir, 'model.pt')
        self.train_feature_file = get_feature_file(self.model_dir, self.readout_name, 'train') # TODO: consolidate into feature_dir?
        self.test_feature_file = get_feature_file(self.model_dir, self.readout_name, 'test')
        self.metrics_file = get_metrics_file(self.model_dir, self.readout_name)

        self.host = host
        self.dbname = dbname
        self.port = port
        self.experiment_name = self.get_experiment_name()
        self.run_name = self.get_run_name()
        self.cfg = self.get_config()
        self.model = self.get_model()
        self.model = self.load_model()
        self.setup_logger()

    def setup_mlflow(self):
        if self.dbname  == 'local':
            mlflow.set_tracking_uri(os.path.join(self.output_dir, 'mlruns'))
            artifact_location = None
        else:
            # create postgres db, and use for backend store
            connection = None
            try:
                connection = psycopg2.connect("user='physopt' password='physopt' host='{}' port='{}' dbname='postgres'".format(self.host, self.port)) # use postgres db just for connection
                print('Database connected.')

            except:
                print('Database not connected.')

            if connection is not None:
                connection.autocommit = True

                cur = connection.cursor()

                cur.execute("SELECT datname FROM pg_database;")

                list_database = cur.fetchall()

                if (self.dbname,) in list_database:
                    print("'{}' Database already exist".format(self.dbname))
                else:
                    print("'{}' Database not exist.".format(self.dbname))
                    sql_create_database = 'create database "{}";'.format(self.dbname)
                    cur.execute(sql_create_database)
                connection.close()
            mlflow.set_tracking_uri('postgresql://physopt:physopt@{}:{}/{}'.format(self.host, self.port, self.dbname)) # need to make sure backend store is setup before we look up experiment name 
            # create s3 bucket, and use for artifact store
            s3 = boto3.resource('s3')
            s3.create_bucket(Bucket=self.dbname)
            artifact_location =  's3://{}'.format(self.dbname) # TODO: add run name to make it more human-readable?

        if mlflow.get_experiment_by_name(self.experiment_name) is None: # create experiment if doesn't exist
            mlflow.create_experiment(self.experiment_name, artifact_location=artifact_location)
        else: # uses old experiment settings (e.g. artifact store location)
            logging.info('Experiment with name "{}" already exists'.format(self.experiment_name))

    def setup_logger(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(self.model_dir, 'output_{}.log'.format(timestr))
        logging.root.handlers = [] # necessary to get handler to work
        logging.basicConfig(
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(),
                ],
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.DEBUG if self.debug else logging.INFO,
            )

    @property
    @abc.abstractmethod
    def model_name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, step):
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def val_step(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_feat_step(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataloader(self, datapaths, phase, train, shuffle): # returns object that can be iterated over for batches of data
        raise NotImplementedError

    def get_config(self):
        cfg = get_cfg()
        cfg.freeze()
        return cfg

    def get_experiment_name(self):
        return self.model_name

    def get_run_name(self):
        to_join = [self.phase, str(self.seed), self.pretraining_name]
        if self.readout_name is not None:
            to_join.append(self.readout_name)
        return '{' + '}_{'.join(to_join) + '}'

    def __call__(self, *args, **kwargs):
        self.setup_mlflow()
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        logging.info(mlflow.get_tracking_uri())
        logging.info(mlflow.get_artifact_uri())
        logging.info(mlflow.active_run())
        mlflow.set_tag('phase', self.phase)
        mlflow.log_params({
            'seed': self.seed,
            })
        mlflow.log_params({f'pretraining_{k}':v for k,v in self.pretraining_space.items()})
        if self.readout_space is not None:
            mlflow.log_params({f'readout_{k}':v for k,v in self.readout_space.items()})

        if self.phase == 'pretraining':  # run model training
            self.pretraining()
        elif self.phase == 'readout':# extract features, then train and test readout
            self.readout() 
        else:
            raise NotImplementedError

        mlflow.log_artifact(self.log_file)
        mlflow.end_run()

        ret = {
                'loss': 0.0,
                'status': STATUS_OK,
                'model': self.model_name,
                'seed': self.seed,
                'output_dir': self.output_dir,
                'phase': self.phase,
                'model_dir': self.model_dir,
                }

        return ret

    def pretraining(self):
        trainloader = self.get_dataloader(self.pretraining_space['train'], phase='pretraining', train=True, shuffle=True)
        best_loss = 1e9
        step = 0
        for epoch in range(self.cfg.EPOCHS): 
            logging.info('Starting epoch {}/{}'.format(epoch+1, self.cfg.EPOCHS))
            for _, data in enumerate(trainloader):
                loss = self.train_step(data)
                logging.info('Step: {0:>10} Loss: {1:>10.4f}'.format(step, loss))

                if (step % self.cfg.LOG_FREQ) == 0:
                    mlflow.log_metric(key='train_loss', value=loss, step=step)
                if (step % self.cfg.VAL_FREQ) == 0:
                    val_results = self.validation()
                    mlflow.log_metrics(val_results, step=step)
                if (step % self.cfg.CKPT_FREQ) == 0:
                    logging.info('Saving model at step {}'.format(step))
                    self.save_model(step)
                step += 1

        # do final validation at end
        val_results = self.validation()
        mlflow.log_metrics(val_results, step=step)

    def validation(self):
        valloader = self.get_dataloader(self.pretraining_space['test'], phase='pretraining', train=False, shuffle=False)
        val_results = []
        for i, data in enumerate(valloader):
            val_res = self.val_step(data)
            assert isinstance(val_res, dict)
            val_results.append(val_res)
            logging.info('Val Step: {0:>10}'.format(i))
        # convert list of dicts into single dict by aggregating over values for a given key
        val_results = {k: np.mean([res[k] for res in val_results]) for k in val_results[0]} # assumes all keys are the same across list
        return val_results

    def readout(self):
        assert os.path.isfile(self.model_file), 'No model ckpt found, cannot extract features'

        trainloader = self.get_dataloader(self.readout_space['train'], phase='readout', train=False, shuffle=False)
        self.extract_feats(trainloader, self.train_feature_file)
        testloader = self.get_dataloader(self.readout_space['test'], phase='readout', train=False, shuffle=False)
        self.extract_feats(testloader, self.test_feature_file)

        self.compute_metrics()

    def extract_feats(self, dataloader, feature_file):
        extracted_feats = []
        for i, data in enumerate(dataloader):
            output = self.extract_feat_step(data)
            extracted_feats.append(output)
        pickle.dump(extracted_feats, open(feature_file, 'wb')) 
        logging.info('Saved features to {}'.format(feature_file))

    def compute_metrics(self):
        logging.info('\n\n{}\nStart Compute Metrics:'.format('*'*80))
        if os.path.exists(self.metrics_file): # rename old results, just in case
            dst = os.path.join(os.path.dirname(self.metrics_file), '.metric_results.csv')
            os.rename(self.metrics_file, dst)
        protocols = ['observed', 'simulated', 'input']
        for protocol in protocols:
            results = run_metrics(
                self.seed,
                self.train_feature_file,
                self.test_feature_file, 
                protocol, 
                grid_search_params=None if self.debug else {'C': np.logspace(-8, 8, 17)},
                )
            results.update({
                'model_name': self.model_name,
                'pretraining_name': self.pretraining_name,
                'readout_name': self.readout_name,
                })
            mlflow.log_metrics({
                'train_acc_'+protocol: results['train_accuracy'], 
                'test_acc_'+protocol: results['test_accuracy'],
                })
            mlflow.log_artifact(self.train_feature_file, artifact_path='features')
            mlflow.log_artifact(self.test_feature_file, artifact_path='features')

            # Write every iteration to be safe
            processed_results = self.process_results(results)
            write_metrics(processed_results, self.metrics_file)
            mlflow.log_artifact(self.metrics_file)

    @staticmethod
    def process_results(results):
        output = []
        for i, (stim_name, test_proba, label) in enumerate(zip(results['stimulus_name'], results['test_proba'], results['labels'])):
            data = {
                'Model Name': results['model_name'],
                'Pretraining Name': results['pretraining_name'],
                'Readout Name': results['readout_name'],
                'Train Accuracy': results['train_accuracy'],
                'Test Accuracy': results['test_accuracy'],
                'Readout Protocol': results['protocol'],
                'Predicted Prob_false': test_proba[0],
                'Predicted Prob_true': test_proba[1],
                'Predicted Outcome': np.argmax(test_proba),
                'Actual Outcome': label,
                'Stimulus Name': stim_name,
                'Seed': results['seed'],
                }
            output.append(data)
        return output

def get_model_dir(output_dir, model_name, train_name, seed, debug=False):
    assert train_name is not None
    if debug:
        model_dir = os.path.join(output_dir,'debug', model_name, train_name, str(seed), 'model/')
    else:
        model_dir = os.path.join(output_dir, model_name, train_name, str(seed), 'model/')
    assert model_dir[-1] == '/', '"{}" missing trailing "/"'.format(model_dir) # need trailing '/' to make dir explicit
    _create_dir(model_dir)
    return model_dir

def get_feature_file(model_dir, test_name, mode):
    if test_name is not None:
        feature_file = os.path.join(model_dir, 'features', test_name, mode+'_feat.pkl')
        _create_dir(feature_file)
        return feature_file

def get_metrics_file(model_dir, test_name):
    if test_name is not None:
        metrics_file = os.path.join(model_dir, 'features', test_name, 'metrics_results.csv')
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
