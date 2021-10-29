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

    def validation(self): # TODO: allow variable agg_func
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
        pretraining_run = self.get_run(PRETRAINING_PHASE_NAME)
        assert pretraining_run is not None, f'Should be exactly 1 pretraining run, but found None'
        assert 'step' in pretraining_run.data.metrics, f'No checkpoint found for pretraining run'

        if self.extraction_cfg.LOAD_STEP is not None: # restore from specified checkpoint
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            metric_history = client.get_metric_history(pretraining_run.info.run_id, 'step')
            assert self.extraction_cfg.LOAD_STEP in [m.value for m in metric_history], f'Checkpoint for step {self.extraction_cfg.LOAD_STEP} not found'
            self.restore_step = self.extraction_cfg.LOAD_STEP
        else: # restore from last checkpoint
            self.restore_step = int(pretraining_run.data.metrics['step'])
            assert self.restore_step == self.pretraining_cfg.TRAIN_STEPS, f'Training not finished - found checkpoint at {self.restore_step} steps, but expected {self.pretraining_cfg.TRAIN_STEPS} steps'
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
                output = self.extract_feat_step(data)
                extracted_feats.append(output)
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
    @property
    def phase(self):
        return READOUT_PHASE_NAME

    @property
    def run_id(self):
        readout_run = self.get_run(READOUT_PHASE_NAME)
        return readout_run.info.run_id

    def setup(self):
        super().setup()
        extraction_run = self.get_run(EXTRACTION_PHASE_NAME)
        assert extraction_run is not None, f'Should be exactly 1 extraction run, but found None'
        restore_run_id = extraction_run.info.run_id
        self.restore_step = int(extraction_run.data.params['restore_step']) # use same restore step as extracted features
        mlflow.log_params({ # log restore settings as params for readout since features and metric results depend on model ckpt
            'restore_step': self.restore_step,
            'restore_run_id': restore_run_id,
            })
        self.train_feature_file = utils.get_feats_from_artifact_store('train', self.tracking_uri, restore_run_id, self.output_dir)
        self.test_feature_file = utils.get_feats_from_artifact_store('test', self.tracking_uri, restore_run_id, self.output_dir)
        assert self.train_feature_file is not None, 'Train features not found'
        assert self.test_feature_file is not None, 'Test features not found'

    def call(self, args):
        # Construct data providers
        logging.info(f'Train feature file: {self.train_feature_file}')
        train_data = metric_utils.build_data(self.train_feature_file)
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

        for protocol in ['observed', 'simulated', 'input']:
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

            # log metrics into mlflow
            mlflow.log_metrics({
                'train_acc_'+protocol: results['train_accuracy'], 
                'test_acc_'+protocol: results['test_accuracy'],
                'step': self.restore_step,
                }, step=self.restore_step)

            if hasattr(metric_model._readout_model, 'best_params_'): # kinda verbose to get the "real" readout model
                assert isinstance(metric_model._readout_model.best_params_, dict)
                prefix = f'best_params_{protocol}_'
                best_params = {prefix+str(k): v for k, v in metric_model._readout_model.best_params_.items()}
                mlflow.log_metrics(best_params, step=self.restore_step)

            if hasattr(metric_model._readout_model, 'cv_results_'):
                logging.info(metric_model._readout_model.cv_results_)
                df = pd.DataFrame(metric_model._readout_model.cv_results_)
                cv_results_file = os.path.join(self.output_dir, protocol+'_cv_results.csv')
                df.to_csv(cv_results_file)
                mlflow.log_artifact(cv_results_file, artifact_path='cv_results')

            # Write every iteration to be safe
            processed_results = self.process_results(results)
            metrics_file = os.path.join(self.output_dir, 'metrics_results.csv')
            metric_utils.write_metrics(processed_results, metrics_file)
            mlflow.log_artifact(metrics_file, artifact_path='metrics')

    @abc.abstractmethod
    def get_readout_model(self):
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