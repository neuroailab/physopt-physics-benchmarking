import os
import shutil
import abc
import logging
import time
import hashlib
import json
import numpy as np
import mlflow
from hyperopt import STATUS_OK, STATUS_FAIL
from physopt.objective import utils
from physopt.objective.utils import PRETRAINING_PHASE_NAME, EXTRACTION_PHASE_NAME, READOUT_PHASE_NAME

LINE_WIDTH = 120

class PhysOptObjective(metaclass=abc.ABCMeta):
    def __init__(self,
        seed,
        pretraining_space,
        readout_space,
        cfg,
        pretraining_cfg,
        extraction_cfg=None,
        readout_cfg=None,
        ):
        self.seed = seed
        self.pretraining_space = pretraining_space
        self.pretraining_name = pretraining_space['name']
        self.readout_space = readout_space
        self.readout_name = None if readout_space is None else readout_space['name']
        self.cfg = cfg
        self.pretraining_cfg = pretraining_cfg
        self.extraction_cfg = extraction_cfg
        self.readout_cfg = readout_cfg

        experiment_name = utils.get_exp_name(cfg)
        self.output_dir = utils.get_output_dir(os.path.join(cfg.OUTPUT_DIR, cfg.DBNAME), experiment_name, self.pretraining_cfg.MODEL_NAME, self.pretraining_name, self.seed, self.phase, self.readout_name)
        self.log_file = os.path.join(self.output_dir, 'logs', 'output_{}.log'.format(time.strftime("%Y%m%d-%H%M%S")))
        utils.setup_logger(self.log_file, self.cfg.DEBUG) # logger only used for utils.create_experiment
        self.tracking_uri, artifact_location = utils.get_mlflow_backend(cfg.OUTPUT_DIR, cfg.HOSTPORT, cfg.DBNAME)
        self.experiment = utils.create_experiment(self.tracking_uri, experiment_name, artifact_location)

    def get_run(self, phase, create_new=True): # TODO: add list of settings to not use when searching for run (e.g. train_steps, ckpt_freq)
        cfgs = utils.flatten(self.pretraining_cfg, prefix=PRETRAINING_PHASE_NAME) # all phases need pretraining
        cfgs.pop('pretraining_TRAIN_STEPS') # remove since train steps not logged as mlflow param
        if phase != PRETRAINING_PHASE_NAME: # for extraction and readout phases
            assert self.readout_name is not None, f'{phase} should have readout_name, but is None'
            cfgs['readout_name'] = self.readout_name # no readout_name for pretraining phase
            cfgs.update(utils.flatten(self.extraction_cfg, prefix=EXTRACTION_PHASE_NAME))
        if phase == READOUT_PHASE_NAME: # only readout needs all three
            cfgs.update(utils.flatten(self.readout_cfg, prefix=READOUT_PHASE_NAME))
        run = utils.get_run(self.tracking_uri, self.experiment.experiment_id, create_new,
            seed=self.seed, pretraining_name=self.pretraining_name, phase=phase, **cfgs)
        return run

    def setup(self):
        self.init_seed()
        mlflow.set_tracking_uri(self.tracking_uri) # needs to be done (again) in __call__ since might be run by worker on different machine
        mlflow.start_run(run_id=self.run_id) # dynamically searches runs to get run_id, creates new run if none found
        logging.info(f'Starting run id: {mlflow.active_run().info.run_id}')
        logging.info(f'Tracking URI: {mlflow.get_tracking_uri()}')
        logging.info(f'Artifact URI: {mlflow.get_artifact_uri()}')
        logging.info(f'Pretraining Space: {self.pretraining_space}')
        logging.info(f'Readout Space: {self.readout_space}')
        mlflow.set_tag('run_id', mlflow.active_run().info.run_id)
        mlflow.log_params({
            'phase': self.phase,
            'seed': self.seed,
            'pretraining_name': self.pretraining_name,
            'pretraining_train_hash': hashlib.md5(json.dumps(self.pretraining_space['train']).encode()).hexdigest(),
            'pretraining_test_hash': hashlib.md5(json.dumps(self.pretraining_space['test']).encode()).hexdigest(),
            })
        if self.readout_space is not None:
            mlflow.log_params({
                'readout_name': self.readout_name,
                'readout_train_hash': hashlib.md5(json.dumps(self.readout_space['train']).encode()).hexdigest(),
                'readout_test_hash': hashlib.md5(json.dumps(self.readout_space['test']).encode()).hexdigest(),
                })
        for phase in ['pretraining', 'extraction', 'readout']: # log params from cfgs
            cfg = getattr(self, f'{phase}_cfg')
            if cfg is not None:
                cfg = utils.flatten(cfg)
                train_steps = cfg.pop('TRAIN_STEPS', None) # don't log train steps as param
                if train_steps:
                    mlflow.set_tag('TRAIN_STEPS', train_steps)
                mlflow.log_params({
                    f'{phase}_{k}':v for k,v in cfg.items()
                    })

    def teardown(self):
        mlflow.log_artifact(self.log_file) # TODO: log to artifact store more frequently?
        mlflow.end_run()

        if self.cfg.DELETE_LOCAL: # delete locally saved files, since they're already logged as mlflow artifacts -- saves disk storage space
            try:
                logging.info(f'Removing all local files in {self.output_dir}')
                shutil.rmtree(self.output_dir)
            except OSError as e:
                print(f'Error: {e.filename} - {e.strerror}.')

    def init_seed(self):
        np.random.seed(self.seed)

    def __call__(self, args):
        utils.setup_logger(self.log_file, self.cfg.DEBUG)
        logging.info(f'\n\n{"*"*LINE_WIDTH}\n{self.phase} setup:')
        self.setup()
        logging.info(f'\n\n{"*"*LINE_WIDTH}\n{self.phase} call:')
        ret = self.call(args)
        logging.info(f'\n\n{"*"*LINE_WIDTH}\n{self.phase} teardown:')
        self.teardown()
        if ret is None:
            ret = {'loss': 0.0, 'status': STATUS_OK}
        return ret

    @abc.abstractmethod
    def call(self, args):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def run_id(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def phase(self):
        raise NotImplementedError

class PhysOptModel(metaclass=abc.ABCMeta): # Mixin Class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # runs init for PhysOptObjective
        self.model = self.get_model()

    @abc.abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, model_file):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, model_file): # not used in Extraction
        raise NotImplementedError

