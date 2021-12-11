import os
import abc
import logging
import numpy as np
import mlflow
import pickle
import joblib
import dill
import pandas as pd

from hyperopt import STATUS_OK, STATUS_FAIL
from physopt.objective.core import PhysOptObjective, PhysOptModel
from physopt.utils import metrics as metric_utils
from physopt.objective import utils
from physopt.objective.utils import PRETRAINING_PHASE_NAME, EXTRACTION_PHASE_NAME, READOUT_PHASE_NAME

class PretrainingObjectiveBase(PhysOptModel, PhysOptObjective):
    @property
    def phase(self):
        return PRETRAINING_PHASE_NAME

    @property
    def run_id(self):
        pretraining_run = self.get_run(PRETRAINING_PHASE_NAME)
        return pretraining_run.info.run_id

    def setup(self):
        super().setup() # starts mlflow run and does some logging
        pretraining_run = self.get_run(PRETRAINING_PHASE_NAME)
        if 'step' in pretraining_run.data.metrics:
            restore_run_id = pretraining_run.info.run_id # restoring from same run
            restore_step = int(pretraining_run.data.metrics['step'])
            # download ckpt from artifact store and load model, if not doing pretraining from scratch
            model_file = utils.get_ckpt_from_artifact_store(self.tracking_uri, restore_run_id, self.output_dir, artifact_path=f'step_{restore_step}/model_ckpts')
            self.model = self.load_model(model_file)
            mlflow.set_tags({ # set as tags for pretraining since can resume run multiple times
                'restore_step': restore_step,
                'restore_run_id': restore_run_id,
                })
            self.initial_step = restore_step + 1 # start with next step
        else:
            self.initial_step = 1
        logging.info(f'Set initial step to {self.initial_step}')

    def call(self, args):
        if (self.initial_step > self.pretraining_cfg.TRAIN_STEPS): # training completed
            if self.initial_step == 1: # TRAIN_STEPS = 0
                self.save_model_with_logging(step=0) # save model since checkpoint might not already exist
            else: # loaded ckpt from final step
                logging.info(f'Loaded fully trained model ({self.pretraining_cfg.TRAIN_STEPS} steps) --  skipping pretraining')
        else:
            trainloader = self.get_pretraining_dataloader(self.pretraining_space['train'], train=True)
            try:
                mlflow.log_param('trainloader_size', len(trainloader))
            except:
                logging.info("Couldn't get trainloader size")

            if self.cfg.DEBUG == False: # skip initial val when debugging
                logging.info(f'Doing initial validation for step {self.initial_step-1}') 
                self.validation_with_logging(self.initial_step-1) # -1 for "zeroth" step
            if self.initial_step == 1: # save initial model if from scratch, other ckpt should already exist
                self.save_model_with_logging(step=0)

            step = self.initial_step
            while step <= self.pretraining_cfg.TRAIN_STEPS:
                for _, data in enumerate(trainloader):
                    loss = self.train_step(data)
                    logging.info('Step: {0:>10} Loss: {1:>10.4f}'.format(step, loss))

                    if (step % self.pretraining_cfg.LOG_FREQ) == 0:
                        mlflow.log_metric(key='train_loss', value=loss, step=step)
                    if (step % self.pretraining_cfg.VAL_FREQ) == 0:
                        self.validation_with_logging(step)
                    if (step % self.pretraining_cfg.CKPT_FREQ) == 0:
                        self.save_model_with_logging(step)
                    # TODO: add function to run at end of each epoch
                    step += 1
                    if step > self.pretraining_cfg.TRAIN_STEPS:
                        break

            # do final validation at end, save model, and log final ckpt -- if it wasn't done at last step
            if (self.pretraining_cfg.TRAIN_STEPS % self.pretraining_cfg.VAL_FREQ != 0):
                self.validation_with_logging(self.pretraining_cfg.TRAIN_STEPS)
            if (self.pretraining_cfg.TRAIN_STEPS % self.pretraining_cfg.CKPT_FREQ != 0):
                self.save_model_with_logging(self.pretraining_cfg.TRAIN_STEPS)
        # TODO: return result dict for hyperopt

    def save_model_with_logging(self, step):
        logging.info('Saving model at step {}'.format(step))
        model_file = os.path.join(self.output_dir, 'model.pt')
        self.save_model(model_file)
        mlflow.log_artifact(model_file, artifact_path=f'step_{step}/model_ckpts')
        mlflow.log_metric('step', step, step=step) # used to know most recent step with model ckpt

    def validation_with_logging(self, step):
        val_results = self.validation()
        mlflow.log_metrics(val_results, step=step)

    def validation(self):
        valloader = self.get_pretraining_dataloader(self.pretraining_space['test'], train=False)
        val_results = []
        for i, data in enumerate(valloader):
            if self.cfg.DEBUG and i >= 2: # stop val early if debug
                break
            val_res = self.val_step(data)
            assert isinstance(val_res, dict)
            val_results.append(val_res)
            logging.info('Val Step: {0:>5}'.format(i+1))
        val_results = self.validation_agg_func(val_results)
        return val_results

    @staticmethod
    def validation_agg_func(val_results):
        # convert list of dicts into single dict by aggregating with mean over values for a given key
        val_results = {k+'_mean': np.mean([res[k] for res in val_results]) for k in val_results[0]} # assumes all keys are the same across list
        return val_results

    @abc.abstractmethod
    def train_step(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def val_step(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def get_pretraining_dataloader(self, datapaths, train): # returns object that can be iterated over for batches of data
        raise NotImplementedError

class ExtractionObjectiveBase(PhysOptModel, PhysOptObjective):
    @property
    def phase(self):
        return EXTRACTION_PHASE_NAME

    @property
    def run_id(self):
        extraction_run = self.get_run(EXTRACTION_PHASE_NAME)
        return extraction_run.info.run_id
    
    def setup(self):
        super().setup()
        pretraining_run = self.get_run(PRETRAINING_PHASE_NAME, create_new=False)
        assert 'step' in pretraining_run.data.metrics, f'No checkpoint found for pretraining run'

        if self.extraction_cfg.LOAD_STEP is not None: # restore from specified checkpoint
            self.restore_step = self.extraction_cfg.LOAD_STEP
        else:
            self.restore_step = self.pretraining_cfg.TRAIN_STEPS
        client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        metric_history = client.get_metric_history(pretraining_run.info.run_id, 'step')
        assert self.restore_step in [m.value for m in metric_history], f'Checkpoint for step {self.restore_step} not found'
        restore_run_id = pretraining_run.info.run_id
        # download ckpt from artifact store and load model
        model_file = utils.get_ckpt_from_artifact_store(self.tracking_uri, restore_run_id, self.output_dir, artifact_path=f'step_{self.restore_step}/model_ckpts')
        self.model = self.load_model(model_file)
        mlflow.log_params({ # log restore settings as params for extraction since features depend on model ckpt
            'restore_step': self.restore_step,
            'restore_run_id': restore_run_id,
            })

    def call(self, args):
        for mode in ['train', 'test']:
            self.extract_feats(mode) 

    def extract_feats(self,  mode):
        feature_file = utils.get_feats_from_artifact_store(mode, self.tracking_uri, self.run_id, self.output_dir)
        if feature_file is None: # features weren't found in artifact store
            dataloader = self.get_readout_dataloader(self.readout_space[mode])
            extracted_feats = []
            for i, data in enumerate(dataloader):
                logging.info('Extract Step: {0:>5}'.format(i+1))
                feats = self.extract_feat_step(data)
                extracted_feats.append(feats)
            feature_file = os.path.join(self.output_dir, mode+'_feat.pkl')
            pickle.dump(extracted_feats, open(feature_file, 'wb')) 
            logging.info('Saved features to {}'.format(feature_file))
            mlflow.log_artifact(feature_file, artifact_path=f'features')

    @abc.abstractmethod
    def extract_feat_step(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def get_readout_dataloader(self, datapaths): # returns object that can be iterated over for batches of data
        raise NotImplementedError

class ReadoutObjectiveBase(PhysOptObjective):
    READOUT_PROTOCOLS = {'observed', 'simulated', 'input'}

    @property
    def phase(self):
        return READOUT_PHASE_NAME

    @property
    def run_id(self):
        readout_run = self.get_run(READOUT_PHASE_NAME)
        return readout_run.info.run_id

    @staticmethod
    def check_feats(feature_file):
        feats_batches = pickle.load(open(feature_file, 'rb'))
        assert isinstance(feats_batches, list), f'Features should be list, but is {type(feats_batches)}'
        for feats in feats_batches:
            assert isinstance(feats, dict)
            required_keys = set(['input_states', 'observed_states', 'simulated_states', 'labels', 'stimulus_name'])
            assert set(feats.keys()) == required_keys, f'{set(feats.keys())} does not match {required_keys}'
            for k,v in feats.items():
                assert isinstance(v, np.ndarray), f'{k} is type {type(v)}, not np.ndarray'

            assert feats['stimulus_name'].ndim == 1
            assert all([isinstance(name, (bytes, str)) for name in feats['stimulus_name']])
            bs = feats['stimulus_name'].size

            assert feats['labels'].ndim == 3
            assert feats['labels'].shape[0] == bs
            assert feats['labels'].shape[2] == 1 # TODO: if labels always scaler this extra dim is unecessary
            T = feats['labels'].shape[1]

            assert feats['input_states'].ndim == 3
            assert feats['input_states'].shape[0] == bs
            T_inp = feats['input_states'].shape[1]
            feat_dim = feats['input_states'].shape[2]

            for k in ['observed_states', 'simulated_states']:
                assert feats[k].ndim == 3
                assert feats[k].shape == (bs, T-T_inp, feat_dim), f'{feats[k].shape} {(bs, T-T_inp, feat_dim)}'

    def setup(self):
        super().setup()
        extraction_run = self.get_run(EXTRACTION_PHASE_NAME, create_new=False)
        restore_run_id = extraction_run.info.run_id
        self.restore_step = int(extraction_run.data.params['restore_step']) # use same restore step as extracted features
        mlflow.log_params({ # log restore settings as params for readout since features and metric results depend on model ckpt
            'restore_step': self.restore_step,
            'restore_run_id': restore_run_id,
            })
        self.train_feature_file = utils.get_feats_from_artifact_store('train', self.tracking_uri, restore_run_id, self.output_dir)
        self.check_feats(self.train_feature_file)
        self.test_feature_file = utils.get_feats_from_artifact_store('test', self.tracking_uri, restore_run_id, self.output_dir)
        self.check_feats(self.test_feature_file)

    def call(self, args):
        # Construct data providers
        assert self.train_feature_file is not None, 'Train features not found'
        logging.info(f'Train feature file: {self.train_feature_file}')
        train_data = metric_utils.build_data(self.train_feature_file)

        assert self.test_feature_file is not None, 'Test features not found'
        logging.info(f'Test feature file: {self.test_feature_file}')
        test_data = metric_utils.build_data(self.test_feature_file)

        # Rebalance data
        logging.info("Rebalancing training data")
        train_data_balanced = metric_utils.rebalance(train_data, metric_utils.label_fn)
        logging.info("Rebalancing testing data")
        test_data_balanced = metric_utils.rebalance(test_data, metric_utils.label_fn)

        # Get stimulus names and labels for test data
        stimulus_names = [d['stimulus_name'] for d in test_data]
        labels = [metric_utils.label_fn(d)[0] for d in test_data]

        for protocol in self.readout_cfg.PROTOCOLS:
            assert protocol in self.READOUT_PROTOCOLS, f'Unknown protocol "{protocol}" -- allowed values {self.READOUT_PROTOCOL}'
            readout_model_file = utils.get_readout_model_from_artifact_store(protocol, self.tracking_uri, self.run_id, self.output_dir)
            if readout_model_file is not None: # using readout model downloaded from artifact store
                logging.info('Loading readout model from: {}'.format(readout_model_file))
                metric_model = joblib.load(readout_model_file)
            else:
                logging.info('Creating new readout model')
                readout_model = self.get_readout_model()
                feature_fn = metric_utils.get_feature_fn(protocol)
                metric_model = metric_utils.MetricModel(readout_model, feature_fn, metric_utils.label_fn, metric_utils.accuracy)

                readout_model_file = os.path.join(self.output_dir, protocol+'_readout_model.joblib')
                logging.info('Training readout model and saving to: {}'.format(readout_model_file))
                metric_model.fit(train_data_balanced)
                joblib.dump(metric_model, readout_model_file)
                mlflow.log_artifact(readout_model_file, artifact_path='readout_models')

            train_acc = metric_model.score(train_data_balanced)
            test_acc = metric_model.score(test_data_balanced)
            test_proba = metric_model.predict(test_data, proba=True)
            logging.info(f'Protocol: {protocol} | Train acc: {train_acc} | Test acc: {test_acc}')

            results = {
                'train_accuracy': train_acc, 
                'test_accuracy': test_acc, 
                'test_proba': test_proba, 
                'stimulus_name': stimulus_names, 
                'labels': labels,
                'protocol': protocol,
                'seed': self.seed,
                'model_name': self.pretraining_cfg.MODEL_NAME,
                'pretraining_name': self.pretraining_name,
                'readout_name': self.readout_name,
                }

            model_info = self.get_readout_model_info(metric_model._readout_model)
            logging.info(f'Model info: {model_info}')
            results.update(model_info)

            # log metrics into mlflow
            mlflow.log_metrics({
                'train_acc_'+protocol: results['train_accuracy'], 
                'test_acc_'+protocol: results['test_accuracy'],
                }, step=self.restore_step)

            metrics_file = os.path.join(self.output_dir, protocol+'_metrics_results.pkl')
            pickle.dump(results, open(metrics_file, 'wb'))
            mlflow.log_artifact(metrics_file, artifact_path='metrics')

    @abc.abstractmethod
    def get_readout_model(self):
        raise NotImplementedError

    @staticmethod
    def get_readout_model_info(readout_model): # split into metrics and artifact?
        return {}
