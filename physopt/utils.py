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
            experiment_name,
            ):
        self.seed = seed
        self.pretraining_space = pretraining_space
        self.pretraining_name = pretraining_space['name']
        self.readout_space = readout_space
        self.readout_name = None if readout_space is None else readout_space['name']
        self.output_dir = output_dir
        self.phase = phase
        self.debug = debug
        self.experiment_name = experiment_name
        self.host = host
        self.dbname = dbname
        self.port = port

        self.model_dir = get_model_dir(self.output_dir, self.experiment_name, self.model_name, self.pretraining_name, self.seed)
        self.readout_dir = get_readout_dir(self.model_dir, self.readout_name)
        self.log_file = os.path.join(self.model_dir, 'logs', 'output_{}.log'.format(time.strftime("%Y%m%d-%H%M%S")))
        setup_logger(self.log_file, self.debug)
        self.cfg = self.get_config()
        self.model = self.get_model()

        self.tracking_uri, self.artifact_location = self.get_mlflow_backend()
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment = self.create_experiment(self.experiment_name)
        self.run_name = get_run_name(self.model_name, self.pretraining_name, self.seed, self.phase, self.readout_name)
        pretraining_run_name = get_run_name(self.model_name, self.pretraining_name, self.seed, 'pretraining')
        if self.phase == 'pretraining':
            assert pretraining_run_name == self.run_name, 'Names should match'
            runs = self.search_runs(pretraining_run_name)
            assert len(runs) <= 1, f'Should be at most 1 run with name "{pretraining_run_name}", but found {len(runs)}'
            self.restore_run_id = None
            self.restore_step = None
            self.initial_step = 1
            if len(runs) == 0: # no run with matching name found
                logging.info(f'Creating run with name:"{pretraining_run_name}"')
                run = self.client.create_run(self.experiment.experiment_id, tags={'mlflow.runName': pretraining_run_name}) # TODO: mflow.start_run to have system tags set
                self.run_id = run.info.run_id
            else: # found existing run with matching name
                assert len(runs) == 1
                logging.info(f'Found run with name:"{pretraining_run_name}"')
                self.run_id = runs[0].info.run_id
                if 'step' in runs[0].data.metrics:
                    self.restore_run_id = self.run_id # restoring from same run
                    self.restore_step = int(runs[0].data.metrics['step'])
                    self.initial_step = self.restore_step + 1 # start with next step
                else:
                    logging.info('Run found, but no ckpts')
            logging.info(f'Set initial step to {self.initial_step}')
        elif self.phase == 'readout': 
            runs = self.search_runs(pretraining_run_name)
            assert len(runs) == 1, f'Should be exactly 1 run with name "{pretraining_run_name}", but found {len(runs)}'
            self.restore_step = int(runs[0].data.metrics['step'])
            assert self.restore_step == self.cfg.TRAIN_STEPS, f'Training not finished - found checkpoint at {step} steps, but expected {self.cfg.TRAIN_STEPS} steps'
            self.restore_run_id = runs[0].info.run_id

            logging.info(f'Creating run with name:"{self.run_name}"')
            run = self.client.create_run(self.experiment.experiment_id, tags={'mlflow.runName': self.run_name})
            self.run_id = run.info.run_id
            # TODO: implement continuing readout if extracted feats found
        else:
            raise NotImplementedError

    def get_ckpt_from_artifact_store(self, run_id, step): # returns path to downloaded ckpt
        artifact_path = f'model_ckpts/model_{step:06d}.pt'
        self.client.download_artifacts(run_id, artifact_path, self.model_dir)
        logging.info(f'Downloaded {artifact_path} to {self.model_dir}')
        model_file = os.path.join(self.model_dir, artifact_path)
        return model_file

    def search_runs(self, run_name):
        filter_string = 'tags.mlflow.runName="{}"'.format(run_name)
        runs = self.client.search_runs([self.experiment.experiment_id], filter_string=filter_string)
        return runs

    def get_mlflow_backend(self): # TODO: split this?
        if self.dbname  == 'local':
            tracking_uri = os.path.join(self.output_dir, 'mlruns')
            artifact_location = None
        else:
            # create postgres db, and use for backend store
            create_postgres_db(self.host, self.port, self.dbname)
            tracking_uri = 'postgresql://physopt:physopt@{}:{}/{}'.format(self.host, self.port, self.dbname) 

            # create s3 bucket, and use for artifact store
            s3 = boto3.resource('s3')
            s3.create_bucket(Bucket=self.dbname)
            artifact_location =  's3://{}'.format(self.dbname) # TODO: add run name to make it more human-readable?
        return tracking_uri, artifact_location

    def create_experiment(self, experiment_name):
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None: # create experiment if doesn't exist
            experiment_id = self.client.create_experiment(experiment_name, artifact_location=self.artifact_location)
            experiment = self.client.get_experiment(experiment_id)
        else: # uses old experiment settings (e.g. artifact store location)
            logging.info('Experiment with name "{}" already exists'.format(experiment_name))
            # TODO: check that experiment settings match?
        return experiment

    def __call__(self, *args, **kwargs):
        setup_logger(self.log_file, self.debug)
        mlflow.set_tracking_uri(self.tracking_uri) # needs to be done (again) in __call__ since might be run by worker on different machine
        mlflow.start_run(run_id=self.run_id)
        logging.info(mlflow.get_tracking_uri())
        logging.info(mlflow.get_artifact_uri())
        logging.info(mlflow.active_run())
        mlflow.set_tags({
            'phase': self.phase,
            'model': self.model_name,
            })
        mlflow.log_params({
            'seed': self.seed,
            'train_steps': self.cfg.TRAIN_STEPS,
            'batch_size': self.cfg.BATCH_SIZE,
            })
        mlflow.log_params({f'pretraining_{k}':v for k,v in self.pretraining_space.items()})
        if self.readout_space is not None:
            mlflow.log_params({f'readout_{k}':v for k,v in self.readout_space.items()})

        # download ckpt from artifact store and load model, if not doing pretraining from scratch
        if (self.restore_step is not None) and (self.restore_run_id is not None):
            model_file = self.get_ckpt_from_artifact_store(self.restore_run_id, self.restore_step)
            self.model = self.load_model(model_file)
            mlflow.set_tags({
                'restore_step': self.restore_step,
                'restore_run_id': self.restore_run_id,
                'restore_model_file': model_file,
                })
        else:
            assert (self.phase == 'pretraining') and (self.initial_step == 1), 'Should be doing pretraining from scratch if not loading model'

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

    def save_model_with_logging(self, step):
        logging.info('Saving model at step {}'.format(step))
        model_file = os.path.join(self.model_dir, f'model_{step:06d}.pt') # create model file with step 
        self.save_model(model_file)
        mlflow.log_artifact(model_file, artifact_path='model_ckpts')
        mlflow.log_metric('step', step, step=step) # used to know most recent step with model ckpt

    def validation_with_logging(self, step):
        val_results = self.validation()
        mlflow.log_metrics(val_results, step=step)

    def pretraining(self):
        trainloader = self.get_dataloader(self.pretraining_space['train'], phase='pretraining', train=True, shuffle=True)
        try:
            mlflow.log_param('trainloader_size', len(trainloader))
        except:
            logging.info("Couldn't get trainloader size")

        if self.initial_step <= self.cfg.TRAIN_STEPS: # only do it if pretraining isn't complete
            logging.info('Doing initial validation') 
            self.validation_with_logging(self.initial_step-1) # -1 for "zeroth" step
            self.save_model_with_logging(self.initial_step-1) # -1 for "zeroth" step

        for step in range(self.initial_step, self.cfg.TRAIN_STEPS+1):
            for _, data in enumerate(trainloader):
                loss = self.train_step(data)
                logging.info('Step: {0:>10} Loss: {1:>10.4f}'.format(step, loss))

                if (step % self.cfg.LOG_FREQ) == 0:
                    mlflow.log_metric(key='train_loss', value=loss, step=step)
                if (step % self.cfg.VAL_FREQ) == 0:
                    self.validation_with_logging(step)
                if (step % self.cfg.CKPT_FREQ) == 0:
                    self.save_model_with_logging(step)

        # do final validation at end, save model, and log final ckpt -- if it wasn't done at last step
        if not (self.cfg.TRAIN_STEPS % self.cfg.VAL_FREQ) == 0:
            self.validation_with_logging(step)
        if not (self.cfg.TRAIN_STEPS % self.cfg.CKPT_FREQ) == 0:
            self.save_model_with_logging(step)

    def validation(self):
        valloader = self.get_dataloader(self.pretraining_space['test'], phase='pretraining', train=False, shuffle=False)
        val_results = []
        for i, data in enumerate(valloader):
            val_res = self.val_step(data)
            assert isinstance(val_res, dict)
            val_results.append(val_res)
            logging.info('Val Step: {0:>5}'.format(i+1))
        # convert list of dicts into single dict by aggregating with mean over values for a given key
        val_results = {k: np.mean([res[k] for res in val_results]) for k in val_results[0]} # assumes all keys are the same across list
        return val_results

    def readout(self):
        for mode in ['train', 'test']:
            self.extract_feats(mode)
        self.compute_metrics()

    def extract_feats(self,  mode):
        dataloader = self.get_dataloader(self.readout_space[mode], phase='readout', train=False, shuffle=False)
        extracted_feats = []
        for i, data in enumerate(dataloader):
            output = self.extract_feat_step(data)
            extracted_feats.append(output)
        feature_file = os.path.join(self.readout_dir, mode+'_feat.pkl')
        pickle.dump(extracted_feats, open(feature_file, 'wb')) 
        logging.info('Saved features to {}'.format(feature_file))
        mlflow.log_artifact(feature_file, artifact_path='features')

    def compute_metrics(self):
        logging.info('\n\n{}\nStart Compute Metrics:'.format('*'*80))
        metrics_file = os.path.join(self.readout_dir, 'metrics_results.csv')
        if os.path.exists(metrics_file): # rename old results, just in case
            dst = os.path.join(self.readout_dir, '.metrics_results.csv')
            os.rename(metrics_file, dst)
        protocols = ['observed', 'simulated', 'input']
        for protocol in protocols:
            results = run_metrics(
                self.seed,
                self.readout_dir,
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

            # Write every iteration to be safe
            processed_results = self.process_results(results)
            write_metrics(processed_results, metrics_file)
            mlflow.log_artifact(metrics_file)

    @property
    @abc.abstractmethod
    def model_name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, run_id):
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
        cfg = get_cfg(self.debug)
        cfg.freeze()
        return cfg

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

def setup_logger(log_file, debug=False):
    _create_dir(log_file)
    logging.root.handlers = [] # necessary to get handler to work
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
            ],
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        )

def get_model_dir(output_dir, experiment_name, model_name, pretraining_name, seed):
    assert pretraining_name is not None
    model_dir = os.path.join(output_dir, experiment_name, model_name, pretraining_name, str(seed), '')
    assert model_dir[-1] == '/', '"{}" missing trailing "/"'.format(model_dir) # need trailing '/' to make dir explicit
    _create_dir(model_dir)
    return model_dir

def get_readout_dir(model_dir, readout_name):
    if readout_name is not None:
        readout_dir = os.path.join(model_dir, readout_name, '')
        _create_dir(readout_dir)
        return readout_dir

def _create_dir(path): # creates dir from path or filename, if doesn't exist
    dirname, basename = os.path.split(path)
    assert '.' in basename if basename else True, 'Are you sure filename is "{}", or should it be a dir'.format(basename) # checks that basename has file-extension
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def create_postgres_db(host, port, dbname):
    connection = None
    try:
        connection = psycopg2.connect(f"user='physopt' password='physopt' host='{host}' port='{port}' dbname='postgres'") # use postgres db just for connection
        logging.info('Database connected.')
    except Exception as e:
        logging.info('Database not connected.')
        raise e

    if connection is not None:
        connection.autocommit = True
        cur = connection.cursor()
        cur.execute("SELECT datname FROM pg_database;")
        list_database = cur.fetchall()

        if (dbname,) in list_database:
            logging.info(f"'{dbname}' Database already exist")
        else:
            logging.info(f"'{dbname}' Database not exist.")
            sql_create_database = f'create database "{dbname}";'
            cur.execute(sql_create_database)
        connection.close()

def get_run_name(model_name, pretraining_name, seed, phase, readout_name=None):
    to_join = [model_name, pretraining_name, str(seed), phase]
    if readout_name is not None:
        to_join.append(readout_name)
    return '{' + '}_{'.join(to_join) + '}'

