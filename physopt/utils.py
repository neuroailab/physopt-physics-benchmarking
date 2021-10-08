import os
import shutil
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
import botocore

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from hyperopt import STATUS_OK, STATUS_FAIL
from physopt.metrics.readout_model import ReadoutModel
from physopt.metrics.test_metrics import run_metrics, write_metrics

PRETRAINING_PHASE_NAME = 'pretraining'
READOUT_PHASE_NAME = 'readout'

class PhysOptObjective(metaclass=abc.ABCMeta):
    def __init__(
            self,
            seed,
            pretraining_space,
            readout_space,
            output_dir,
            cfg,
            ):
        self.seed = seed
        self.pretraining_space = pretraining_space
        self.pretraining_name = pretraining_space['name']
        self.readout_space = readout_space
        self.readout_name = None if readout_space is None else readout_space['name']
        self.cfg = cfg
        self.phase = PRETRAINING_PHASE_NAME if readout_space is None else READOUT_PHASE_NAME

        experiment_name = get_exp_name(cfg.EXPERIMENT_NAME, cfg.ADD_TIMESTAMP, cfg.DEBUG)
        self.model_dir = get_model_dir(output_dir, experiment_name, self.model_name, self.pretraining_name, self.seed)
        self.readout_dir = get_readout_dir(self.model_dir, self.readout_name) # TODO: combine readout and model dir
        self.log_file = os.path.join(self.model_dir, 'logs', 'output_{}.log'.format(time.strftime("%Y%m%d-%H%M%S")))
        setup_logger(self.log_file, self.cfg.DEBUG)
        self.model = self.get_model()

        self.tracking_uri, artifact_location = get_mlflow_backend(output_dir, cfg.POSTGRES.HOST, cfg.POSTGRES.PORT, cfg.POSTGRES.DBNAME)
        experiment = create_experiment(self.tracking_uri, experiment_name, artifact_location)
        self.run_name = get_run_name(self.model_name, self.pretraining_name, self.seed, self.readout_name)
        pretraining_run_name = get_run_name(self.model_name, self.pretraining_name, self.seed)
        pretraining_run = get_run(self.tracking_uri, experiment.experiment_id, pretraining_run_name)
        if self.phase == PRETRAINING_PHASE_NAME: # pretraining
            assert pretraining_run_name == self.run_name, 'Names should match: {} and {}'.format(pretraining_run_name, self.run_name)
            self.restore_run_id = None
            self.restore_step = None
            self.initial_step = 1
            self.run_id = pretraining_run.info.run_id
            if 'step' in pretraining_run.data.metrics:
                self.restore_run_id = self.run_id # restoring from same run
                self.restore_step = int(pretraining_run.data.metrics['step'])
                self.initial_step = self.restore_step + 1 # start with next step
            logging.info(f'Set initial step to {self.initial_step}')
        else: # readout
            assert pretraining_run is not None, f'Should be exactly 1 run with name "{pretraining_run_name}", but found None'
            assert 'step' in pretraining_run.data.metrics, f'No checkpoint found for "{pretraining_run_name}"'
            if cfg.LOAD_STEP is not None: # restore from specified checkpoint
                client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
                metric_history = client.get_metric_history(pretraining_run.info.run_id, 'step')
                assert cfg.LOAD_STEP in [m.value for m in metric_history], f'Checkpoint for step {cfg.LOAD_STEP} not found'
                self.restore_step = cfg.LOAD_STEP
            else: # restore from last checkpoint
                self.restore_step = int(pretraining_run.data.metrics['step'])
                assert self.restore_step == cfg.TRAIN_STEPS, f'Training not finished - found checkpoint at {step} steps, but expected {cfg.TRAIN_STEPS} steps'
            self.restore_run_id = pretraining_run.info.run_id

            readout_run = get_run(self.tracking_uri, experiment.experiment_id, self.run_name)
            self.run_id = readout_run.info.run_id

    def __call__(self, *args, **kwargs):
        setup_logger(self.log_file, self.cfg.DEBUG)
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
            'batch_size': self.cfg.BATCH_SIZE,
            })
        # TODO: log params in self.cfg?
        mlflow.log_params({f'pretraining_{k}':v for k,v in self.pretraining_space.items()})
        if self.readout_space is not None:
            mlflow.log_params({f'readout_{k}':v for k,v in self.readout_space.items()})

        # download ckpt from artifact store and load model, if not doing pretraining from scratch
        if (self.restore_step is not None) and (self.restore_run_id is not None):
            model_file = get_ckpt_from_artifact_store(self.restore_step, self.tracking_uri, self.restore_run_id, self.model_dir)
            self.model = self.load_model(model_file)
            restore_settings = {
                'restore_step': self.restore_step,
                'restore_run_id': self.restore_run_id,
                'restore_model_file': model_file,
                }
            if self.phase == PRETRAINING_PHASE_NAME: # set as tags for pretraining since can resume run multiple times
                mlflow.set_tags(restore_settings)
            else: # log restore settings as params for readout since features and metric results depend on model ckpt
                mlflow.log_params(restore_settings)
        else:
            assert (self.phase == PRETRAINING_PHASE_NAME) and (self.initial_step == 1), 'Should be doing pretraining from scratch if not loading model'

        if self.phase == PRETRAINING_PHASE_NAME:  # run model training
            self.pretraining()
        elif self.phase == READOUT_PHASE_NAME:# extract features, then train and test readout
            self.readout() 
        else:
            raise NotImplementedError

        mlflow.log_artifact(self.log_file)
        mlflow.end_run()

        if self.cfg.DELETE_LOCAL: # delete locally saved files, since they're already logged as mlflow artifacts -- saves disk storage space
            try:
                logging.info(f'Removing all local files in {self.model_dir}')
                shutil.rmtree(self.model_dir)
            except OSError as e:
                print(f'Error: {e.filename} - {e.strerror}.')

        ret = {
                'loss': 0.0,
                'status': STATUS_OK,
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
        trainloader = self.get_pretraining_dataloader(self.pretraining_space['train'], train=True)
        try:
            mlflow.log_param('trainloader_size', len(trainloader))
        except:
            logging.info("Couldn't get trainloader size")

        if (not self.cfg.DEBUG) and (self.initial_step <= self.cfg.TRAIN_STEPS): # only do it if pretraining isn't complete and not debug
            logging.info('Doing initial validation') 
            self.validation_with_logging(self.initial_step-1) # -1 for "zeroth" step
            self.save_model_with_logging(self.initial_step-1) # -1 for "zeroth" step

        step = self.initial_step
        while step <= self.cfg.TRAIN_STEPS:
            for _, data in enumerate(trainloader):
                loss = self.train_step(data)
                logging.info('Step: {0:>10} Loss: {1:>10.4f}'.format(step, loss))

                if (step % self.cfg.LOG_FREQ) == 0:
                    mlflow.log_metric(key='train_loss', value=loss, step=step)
                if (step % self.cfg.VAL_FREQ) == 0:
                    self.validation_with_logging(step)
                if (step % self.cfg.CKPT_FREQ) == 0:
                    self.save_model_with_logging(step)
                step += 1
                if step > self.cfg.TRAIN_STEPS:
                    break

        # do final validation at end, save model, and log final ckpt -- if it wasn't done at last step
        if not (self.cfg.TRAIN_STEPS % self.cfg.VAL_FREQ) == 0:
            self.validation_with_logging(step)
        if not (self.cfg.TRAIN_STEPS % self.cfg.CKPT_FREQ) == 0:
            self.save_model_with_logging(step)

    def validation(self):
        valloader = self.get_pretraining_dataloader(self.pretraining_space['test'], train=False)
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
        kwargs = {}
        for mode in ['train', 'test']:
            feature_file = self.extract_feats(mode) 
            kwargs[mode+'_feature_file'] = feature_file
        self.compute_metrics(**kwargs)

    def extract_feats(self,  mode):
        feature_file = get_feats_from_artifact_store(mode, self.tracking_uri, self.run_id, self.readout_dir)
        if feature_file is None: # features weren't found in artifact store
            dataloader = self.get_readout_dataloader(self.readout_space[mode])
            extracted_feats = []
            for i, data in enumerate(dataloader):
                output = self.extract_feat_step(data)
                extracted_feats.append(output)
            feature_file = os.path.join(self.readout_dir, mode+'_feat.pkl')
            pickle.dump(extracted_feats, open(feature_file, 'wb')) 
            logging.info('Saved features to {}'.format(feature_file))
            mlflow.log_artifact(feature_file, artifact_path='features')
        return feature_file

    def compute_metrics(self, train_feature_file, test_feature_file):
        logging.info('\n\n{}\nStart Compute Metrics:'.format('*'*80))
        metrics_file = os.path.join(self.readout_dir, 'metrics_results.csv')
        if os.path.exists(metrics_file): # rename old results, just in case
            dst = os.path.join(self.readout_dir, '.metrics_results.csv')
            os.rename(metrics_file, dst)
        protocols = ['observed', 'simulated', 'input']
        for protocol in protocols:
            readout_model_or_file = get_readout_model_from_artifact_store(protocol, self.tracking_uri, self.run_id, self.readout_dir)
            if readout_model_or_file is None:
                readout_model_or_file = self.get_readout_model()
            results = run_metrics(
                self.seed,
                readout_model_or_file,
                self.readout_dir,
                train_feature_file,
                test_feature_file,
                protocol, 
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
            if 'best_params' in results:
                assert isinstance(results['best_params'], dict)
                prefix = f'best_params_{protocol}_'
                best_params = {prefix+str(k): v for k, v in results['best_params'].items()}
                mlflow.log_metrics(best_params)

            # Write every iteration to be safe
            processed_results = self.process_results(results)
            write_metrics(processed_results, metrics_file)
            mlflow.log_artifact(metrics_file)

    def get_readout_model(self):
        grid_search_params = {'C': np.logspace(-(self.cfg.READOUT.NUM_C//2), self.cfg.READOUT.NUM_C//2, self.cfg.READOUT.NUM_C)}
        model = GridSearchCV(LogisticRegression(max_iter=100), grid_search_params)
        readout_model = ReadoutModel(model)
        return readout_model

    @property
    @classmethod
    @abc.abstractmethod
    def model_name(cls):
        raise NotImplementedError

    @abc.abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, model_file):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, model_file):
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
    def get_pretraining_dataloader(self, datapaths, train): # returns object that can be iterated over for batches of data
        raise NotImplementedError

    @abc.abstractmethod
    def get_readout_dataloader(self, datapaths): # returns object that can be iterated over for batches of data
        raise NotImplementedError

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

def get_run_name(model_name, pretraining_name, seed, readout_name=None, separator='-'):
    to_join = [model_name, pretraining_name, str(seed)]
    if readout_name is not None:
        to_join.extend([READOUT_PHASE_NAME, readout_name])
    else:
        to_join.append(PRETRAINING_PHASE_NAME)
    return separator.join(to_join)

def get_exp_name(name, add_ts=False, debug=False):
        if debug:
            experiment_name = 'DEBUG'
        elif add_ts:
            experiment_name = name + '_' + time.strftime("%Y%m%d-%H%M%S")
        else:
            experiment_name = name
        return experiment_name

def get_mlflow_backend(output_dir, host, port, dbname): # TODO: split this?
    if dbname  == 'local':
        tracking_uri = os.path.join(output_dir, 'mlruns')
        artifact_location = None
    else:
        # create postgres db, and use for backend store
        create_postgres_db(host, port, dbname)
        tracking_uri = 'postgresql://physopt:physopt@{}:{}/{}'.format(host, port, dbname) 

        # create s3 bucket, and use for artifact store
        s3 = boto3.resource('s3')
        s3.create_bucket(Bucket=dbname)
        artifact_location =  's3://{}'.format(dbname) # TODO: add run name to make it more human-readable?
    return tracking_uri, artifact_location

def download_from_artifact_store(artifact_path, tracking_uri, run_id, output_dir): # Tries to download artifact, returns None if not found
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        client.download_artifacts(run_id, artifact_path, output_dir)
        logging.info(f'Downloaded {artifact_path} to {output_dir}')
        output_file = os.path.join(output_dir, artifact_path)
    except (FileNotFoundError, botocore.exceptions.ClientError):
        logging.info(f"Couldn't find artifact at {artifact_path} in artifact store")
        logging.debug(traceback.format_exc())
        output_file = None
    return output_file

def get_ckpt_from_artifact_store(step, tracking_uri, run_id, model_dir): # returns path to downloaded ckpt, if found
    artifact_path = f'model_ckpts/model_{step:06d}.pt'
    model_file = download_from_artifact_store(artifact_path, tracking_uri, run_id, model_dir)
    return model_file

def get_feats_from_artifact_store(mode, tracking_uri, run_id, readout_dir): # returns path to downloaded feats, if found
    artifact_path = f'features/{mode}_feat.pkl'
    feat_file = download_from_artifact_store(artifact_path, tracking_uri, run_id, readout_dir)
    return feat_file

def get_readout_model_from_artifact_store(protocol, tracking_uri, run_id, readout_dir): # returns path to downloaded readout model, if found
    artifact_path = f'readout_models/{protocol}_readout_model.joblib'
    readout_model_file = download_from_artifact_store(artifact_path, tracking_uri, run_id, readout_dir)
    return readout_model_file

def get_run(tracking_uri, experiment_id, run_name):
    runs = search_runs(tracking_uri, experiment_id, run_name)
    assert len(runs) <= 1, f'Should be at most one (1) run with name "{run_name}", but found {len(runs)}'
    if len(runs) == 0:
        logging.info(f'Creating run with name:"{run_name}"')
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        run = client.create_run(experiment_id, tags={'mlflow.runName': run_name})
    else: # found existing run with matching name
        logging.info(f'Found run with name:"{run_name}"')
        run = runs[0]
    return run

def search_runs(tracking_uri, experiment_id, run_name):
    filter_string = 'tags.mlflow.runName="{}"'.format(run_name)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs([experiment_id], filter_string=filter_string)
    return runs

def create_experiment(tracking_uri, experiment_name, artifact_location):
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None: # create experiment if doesn't exist
        logging.info('Creating new experiment with name "{}"'.format(experiment_name))
        experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_location)
        experiment = client.get_experiment(experiment_id)
    else: # uses old experiment settings (e.g. artifact store location)
        logging.info('Experiment with name "{}" already exists'.format(experiment_name))
        # TODO: check that experiment settings match?
    return experiment

