import os
import shutil
import abc
import logging
import time
import numpy as np
import mlflow
import pickle
import psycopg2
import boto3
import botocore

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from hyperopt import STATUS_OK, STATUS_FAIL
from physopt.metrics.test_metrics import run_metrics, write_metrics
from physopt import utils
from physopt.utils import PRETRAINING_PHASE_NAME, READOUT_PHASE_NAME

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

        experiment_name = utils.get_exp_name(cfg.EXPERIMENT_NAME, cfg.ADD_TIMESTAMP, cfg.DEBUG)
        self.model_dir = utils.get_model_dir(output_dir, experiment_name, self.model_name, self.pretraining_name, self.seed)
        self.readout_dir = utils.get_readout_dir(self.model_dir, self.readout_name) # TODO: combine readout and model dir
        self.log_file = os.path.join(self.model_dir, 'logs', 'output_{}.log'.format(time.strftime("%Y%m%d-%H%M%S")))
        utils.setup_logger(self.log_file, self.cfg.DEBUG)
        self.model = self.get_model()

        self.tracking_uri, artifact_location = utils.get_mlflow_backend(output_dir, cfg.POSTGRES.HOST, cfg.POSTGRES.PORT, cfg.POSTGRES.DBNAME)
        experiment = utils.create_experiment(self.tracking_uri, experiment_name, artifact_location)
        self.run_name = utils.get_run_name(self.model_name, self.pretraining_name, self.seed, self.readout_name)
        pretraining_run_name = utils.get_run_name(self.model_name, self.pretraining_name, self.seed)
        pretraining_run = utils.get_run(self.tracking_uri, experiment.experiment_id, pretraining_run_name)
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
            if cfg.READOUT_LOAD_STEP is not None: # restore from specified checkpoint
                client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
                metric_history = client.get_metric_history(pretraining_run.info.run_id, 'step')
                assert cfg.READOUT_LOAD_STEP in [m.value for m in metric_history], f'Checkpoint for step {cfg.READOUT_LOAD_STEP} not found'
                self.restore_step = cfg.READOUT_LOAD_STEP
            else: # restore from last checkpoint
                self.restore_step = int(pretraining_run.data.metrics['step'])
                assert self.restore_step == cfg.TRAIN_STEPS, f'Training not finished - found checkpoint at {self.restore_step} steps, but expected {cfg.TRAIN_STEPS} steps'
            self.restore_run_id = pretraining_run.info.run_id

            readout_run = utils.get_run(self.tracking_uri, experiment.experiment_id, self.run_name)
            self.run_id = readout_run.info.run_id

    def __call__(self, args): # TODO: hyperparam settings from hyperopt are passed in as a dict/tuple
        utils.setup_logger(self.log_file, self.cfg.DEBUG)
        mlflow.set_tracking_uri(self.tracking_uri) # needs to be done (again) in __call__ since might be run by worker on different machine
        mlflow.start_run(run_id=self.run_id)
        logging.info(f'Starting run id: {mlflow.active_run().info.run_id}')
        logging.info(f'Tracking URI: {mlflow.get_tracking_uri()}')
        logging.info(f'Artifact URI: {mlflow.get_artifact_uri()}')
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
            model_file = utils.get_ckpt_from_artifact_store(self.restore_step, self.tracking_uri, self.restore_run_id, self.model_dir)
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

        mlflow.log_artifact(self.log_file) # TODO: log to artifact store more frequently?
        mlflow.end_run()

        if self.cfg.DELETE_LOCAL: # delete locally saved files, since they're already logged as mlflow artifacts -- saves disk storage space
            try:
                logging.info(f'Removing all local files in {self.model_dir}')
                shutil.rmtree(self.model_dir)
            except OSError as e:
                print(f'Error: {e.filename} - {e.strerror}.')

        ret = {
                'loss': 0.0, # TODO: set as val metric from pretraining
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
        step -= 1 # reset final step increment, so aligns with actual number of steps run

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
        feature_file = utils.get_feats_from_artifact_store(mode, self.tracking_uri, self.run_id, self.readout_dir)
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
        logging.info(self.cfg.READOUT)
        metrics_file = os.path.join(self.readout_dir, 'metrics_results.csv')
        if os.path.exists(metrics_file): # rename old results, just in case
            dst = os.path.join(self.readout_dir, '.metrics_results.csv')
            os.rename(metrics_file, dst)
        protocols = ['observed', 'simulated', 'input']
        for protocol in protocols:
            readout_model_or_file = utils.get_readout_model_from_artifact_store(protocol, self.tracking_uri, self.run_id, self.readout_dir)
            if not self.cfg.READOUT.DO_RESTORE or (readout_model_or_file is None):
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
        steps = [('clf', LogisticRegression(max_iter=self.cfg.READOUT.MAX_ITER))]
        if self.cfg.READOUT.NORM_INPUT:
            steps.insert(0, ('scale', StandardScaler()))
        logging.info(f'Readout model steps: {steps}')
        pipe = Pipeline(steps)

        assert len(self.cfg.READOUT.LOGSPACE) == 3, 'logspace must contain start, stop, and num'
        grid_search_params = {
            'clf__C': np.logspace(*self.cfg.READOUT.LOGSPACE),
            }
        skf = StratifiedKFold(n_splits=self.cfg.READOUT.CV, shuffle=True, random_state=self.seed)
        logging.info(f'CV folds: {skf}')
        readout_model = GridSearchCV(pipe, param_grid=grid_search_params, cv=skf, verbose=3)
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
