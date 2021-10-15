import os
import abc
import logging
import numpy as np
import mlflow
import pickle

from hyperopt import STATUS_OK, STATUS_FAIL
from physopt.objective.core import PhysOptObjective, PhysOptModel
from physopt.metrics.test_metrics import run_metrics, write_metrics
from physopt.objective import utils
from physopt.objective.utils import PRETRAINING_PHASE_NAME, EXTRACTION_PHASE_NAME, READOUT_PHASE_NAME

class PretrainingObjectiveBase(PhysOptObjective, PhysOptModel):
    @property
    def phase(self):
        return PRETRAINING_PHASE_NAME

    @property
    def run_id(self):
        pretraining_run = utils.get_run(self.tracking_uri, self.experiment.experiment_id, 
            model_name=self.model_name, seed=self.seed, pretraining_name=self.pretraining_name, phase=PRETRAINING_PHASE_NAME)
        return pretraining_run.info.run_id

    def setup(self):
        super().setup() # starts mlflow run and does some logging
        pretraining_run = utils.get_run(self.tracking_uri, self.experiment.experiment_id,
            model_name=self.model_name, seed=self.seed, pretraining_name=self.pretraining_name, phase=PRETRAINING_PHASE_NAME)
        if 'step' in pretraining_run.data.metrics:
            self.restore_run_id = self.run_id # restoring from same run
            self.restore_step = int(pretraining_run.data.metrics['step'])
            self.initial_step = self.restore_step + 1 # start with next step
        else:
            self.restore_run_id = None
            self.restore_step = None
            self.initial_step = 1
        logging.info(f'Set initial step to {self.initial_step}')

        # download ckpt from artifact store and load model, if not doing pretraining from scratch
        if (self.restore_step is not None) and (self.restore_run_id is not None):
            model_file = utils.get_ckpt_from_artifact_store(self.restore_step, self.tracking_uri, self.restore_run_id, self.output_dir)
            self.model = self.load_model(model_file)
            mlflow.set_tags({ # set as tags for pretraining since can resume run multiple times
                'restore_step': self.restore_step,
                'restore_run_id': self.restore_run_id,
                'restore_model_file': model_file,
                })
        else:
            assert self.initial_step == 1, 'Should be doing pretraining from scratch if not loading model'

    def call(self, args):
        if self.cfg.DEBUG == False: # skip initial val when debugging
            logging.info(f'Doing initial validation for step {self.initial_step-1}') 
            self.validation_with_logging(self.initial_step-1) # -1 for "zeroth" step

        if (self.initial_step > self.cfg.TRAIN_STEPS): # loaded ckpt from final step
            assert self.restore_step == self.cfg.TRAIN_STEPS, f'Restore step {self.restore_step} should match train steps {self.cfg.TRAIN_STEPS}'
            logging.info(f'Loaded fully trained model ({self.cfg.TRAIN_STEPS} steps) --  skipping pretraining')
        else:
            trainloader = self.get_pretraining_dataloader(self.pretraining_space['train'], train=True)
            try:
                mlflow.log_param('trainloader_size', len(trainloader))
            except:
                logging.info("Couldn't get trainloader size")

            if self.initial_step == 1: # save initial model
                self.save_model_with_logging(step=0)
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
            if (self.cfg.TRAIN_STEPS % self.cfg.VAL_FREQ != 0):
                self.validation_with_logging(self.cfg.TRAIN_STEPS)
            if (self.cfg.TRAIN_STEPS % self.cfg.CKPT_FREQ != 0):
                self.save_model_with_logging(self.cfg.TRAIN_STEPS)
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

class ExtractionObjectiveBase(PhysOptObjective, PhysOptModel):
    @property
    def phase(self):
        return EXTRACTION_PHASE_NAME

    @property
    def run_id(self):
        extraction_run = utils.get_run(self.tracking_uri, self.experiment.experiment_id,
            model_name=self.model_name, seed=self.seed, pretraining_name=self.pretraining_name, phase=EXTRACTION_PHASE_NAME, readout_name=self.readout_name)
        return extraction_run.info.run_id
    
    def setup(self):
        super().setup()
        pretraining_run = utils.get_run(self.tracking_uri, self.experiment.experiment_id,
            model_name=self.model_name, seed=self.seed, pretraining_name=self.pretraining_name, phase=PRETRAINING_PHASE_NAME)
        assert pretraining_run is not None, f'Should be exactly 1 run with name "{pretraining_run_name}", but found None'
        assert 'step' in pretraining_run.data.metrics, f'No checkpoint found for "{pretraining_run_name}"'

        if self.cfg.READOUT_LOAD_STEP is not None: # restore from specified checkpoint
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            metric_history = client.get_metric_history(pretraining_run.info.run_id, 'step')
            assert self.cfg.READOUT_LOAD_STEP in [m.value for m in metric_history], f'Checkpoint for step {self.cfg.READOUT_LOAD_STEP} not found'
            self.restore_step = self.cfg.READOUT_LOAD_STEP
        else: # restore from last checkpoint
            self.restore_step = int(pretraining_run.data.metrics['step'])
            assert self.restore_step == self.cfg.TRAIN_STEPS, f'Training not finished - found checkpoint at {self.restore_step} steps, but expected {self.cfg.TRAIN_STEPS} steps'
        self.restore_run_id = pretraining_run.info.run_id
        # download ckpt from artifact store and load model
        model_file = utils.get_ckpt_from_artifact_store(self.restore_step, self.tracking_uri, self.restore_run_id, self.output_dir)
        self.model = self.load_model(model_file)
        mlflow.log_params({ # log restore settings as params for readout since features and metric results depend on model ckpt
            'restore_run_id': self.restore_run_id,
            })
        mlflow.log_metric('step', self.restore_step, step=self.restore_step)

    def call(self, args):
        for mode in ['train', 'test']:
            self.extract_feats(mode) 

    def extract_feats(self,  mode):
        feature_file = utils.get_feats_from_artifact_store(mode, self.restore_step, self.tracking_uri, self.run_id, self.output_dir)
        if feature_file is None: # features weren't found in artifact store
            dataloader = self.get_readout_dataloader(self.readout_space[mode])
            extracted_feats = []
            for i, data in enumerate(dataloader):
                output = self.extract_feat_step(data)
                extracted_feats.append(output)
            feature_file = os.path.join(self.output_dir, mode+'_feat.pkl')
            pickle.dump(extracted_feats, open(feature_file, 'wb')) 
            logging.info('Saved features to {}'.format(feature_file))
            mlflow.log_artifact(feature_file, artifact_path=f'step_{self.restore_step}/features')

    @abc.abstractmethod
    def extract_feat_step(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def get_readout_dataloader(self, datapaths): # returns object that can be iterated over for batches of data
        raise NotImplementedError

class ReadoutObjectiveBase(PhysOptObjective):
    def __init__(self,
        seed,
        pretraining_space,
        readout_space,
        output_dir,
        cfg,
        ):
        super().__init__(seed, pretraining_space, readout_space, output_dir, cfg)

    @property
    def phase(self):
        return READOUT_PHASE_NAME

    @property
    def run_id(self):
        readout_run = utils.get_run(self.tracking_uri, self.experiment.experiment_id,
            model_name=self.model_name, seed=self.seed, pretraining_name=self.pretraining_name, phase=READOUT_PHASE_NAME, readout_name=self.readout_name)
        return readout_run.info.run_id

    def setup(self):
        super().setup()
        extraction_run = utils.get_run(self.tracking_uri, self.experiment.experiment_id,
            model_name=self.model_name, seed=self.seed, pretraining_name=self.pretraining_name, phase=EXTRACTION_PHASE_NAME, readout_name=self.readout_name)
        assert extraction_run is not None, f'Should be exactly 1 run with name "{extraction_run_name}", but found None'
        client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        metric_history = client.get_metric_history(extraction_run.info.run_id, 'step')
        metric_values = [m.value for m in metric_history]
        logging.info(f'Steps: {metric_values}')
        if self.cfg.READOUT_LOAD_STEP is not None: # get features from specified step
            assert self.cfg.READOUT_LOAD_STEP in metric_values, f'Features for step {self.cfg.READOUT_LOAD_STEP} not found'
            self.restore_step = self.cfg.READOUT_LOAD_STEP
        else: # restore from lastest features
            self.restore_step = int(max(metric_values))
        self.train_feature_file = utils.get_feats_from_artifact_store('train', self.restore_step, self.tracking_uri, extraction_run.info.run_id, self.output_dir)
        self.test_feature_file = utils.get_feats_from_artifact_store('test', self.restore_step, self.tracking_uri, extraction_run.info.run_id, self.output_dir)
        assert self.train_feature_file is not None, 'Train features not found'
        assert self.test_feature_file is not None, 'Test features not found'

    def call(self, args):
        logging.info('\n\n{}\nStart Compute Metrics:'.format('*'*80))
        logging.info(self.cfg.READOUT)
        metrics_file = os.path.join(self.output_dir, 'metrics_results.csv')
        if os.path.exists(metrics_file): # rename old results, just in case
            dst = os.path.join(self.output_dir, '.metrics_results.csv')
            os.rename(metrics_file, dst)
        protocols = ['observed', 'simulated', 'input']
        for protocol in protocols:
            readout_model_or_file = utils.get_readout_model_from_artifact_store(protocol, self.restore_step, self.tracking_uri, self.run_id, self.output_dir)
            if not self.cfg.READOUT.DO_RESTORE or (readout_model_or_file is None):
                readout_model_or_file = self.get_readout_model()
            results = run_metrics(
                self.seed,
                readout_model_or_file,
                self.output_dir,
                self.train_feature_file,
                self.test_feature_file,
                protocol, 
                self.restore_step,
                )
            results.update({
                'model_name': self.model_name,
                'pretraining_name': self.pretraining_name,
                'readout_name': self.readout_name,
                })
            mlflow.log_metrics({
                'train_acc_'+protocol: results['train_accuracy'], 
                'test_acc_'+protocol: results['test_accuracy'],
                'step': self.restore_step,
                }, step=self.restore_step)
            if 'best_params' in results:
                assert isinstance(results['best_params'], dict)
                prefix = f'best_params_{protocol}_'
                best_params = {prefix+str(k): v for k, v in results['best_params'].items()}
                mlflow.log_metrics(best_params, step=self.restore_step)

            # Write every iteration to be safe
            processed_results = self.process_results(results)
            write_metrics(processed_results, metrics_file)
            mlflow.log_artifact(metrics_file, artifact_path=f'step_{self.restore_step}/metrics')

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
