import os
import shutil
import abc
import logging
import time
import mlflow
from hyperopt import STATUS_OK, STATUS_FAIL
from physopt.objective import utils
from physopt.objective.utils import PRETRAINING_PHASE_NAME, EXTRACTION_PHASE_NAME, READOUT_PHASE_NAME

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

        experiment_name = utils.get_exp_name(cfg.EXPERIMENT_NAME, cfg.ADD_TIMESTAMP, cfg.DEBUG)
        self.output_dir = utils.get_output_dir(cfg.OUTPUT_DIR, experiment_name, self.model_name, self.pretraining_name, self.seed, self.phase, self.readout_name)
        self.log_file = os.path.join(self.output_dir, 'logs', 'output_{}.log'.format(time.strftime("%Y%m%d-%H%M%S")))
        utils.setup_logger(self.log_file, self.cfg.DEBUG)
        self.tracking_uri, artifact_location = utils.get_mlflow_backend(cfg.OUTPUT_DIR, cfg.POSTGRES.HOST, cfg.POSTGRES.PORT, cfg.POSTGRES.DBNAME)
        self.experiment = utils.create_experiment(self.tracking_uri, experiment_name, artifact_location)
        super().__init__() # runs init for PhysOptModel

    def get_run(self, phase):
        cfgs = utils.flatten(self.pretraining_cfg, prefix=PRETRAINING_PHASE_NAME) # all phases need pretraining
        if phase != PRETRAINING_PHASE_NAME: # for extraction and readout phases
            cfgs.update(utils.flatten(self.extraction_cfg, prefix=EXTRACTION_PHASE_NAME))
        if phase == READOUT_PHASE_NAME: # only readout needs all three
            cfgs.update(utils.flatten(self.readout_cfg, prefix=READOUT_PHASE_NAME))
        run = utils.get_run(self.tracking_uri, self.experiment.experiment_id, model_name=self.model_name, 
            seed=self.seed, pretraining_name=self.pretraining_name, phase=phase, **cfgs)
        return run

    def setup(self):
        mlflow.set_tracking_uri(self.tracking_uri) # needs to be done (again) in __call__ since might be run by worker on different machine
        mlflow.start_run(run_id=self.run_id) # dynamically searches runs to get run_id, creates new run if none found
        logging.info(f'Starting run id: {mlflow.active_run().info.run_id}')
        logging.info(f'Tracking URI: {mlflow.get_tracking_uri()}')
        logging.info(f'Artifact URI: {mlflow.get_artifact_uri()}')
        mlflow.set_tag('run_id', mlflow.active_run().info.run_id)
        mlflow.log_params({
            'phase': self.phase,
            'model_name': self.model_name,
            'seed': self.seed,
            })
        for phase in ['pretraining', 'extraction', 'readout']: # log params from cfgs
            cfg = getattr(self, f'{phase}_cfg')
            if cfg is not None:
                cfg = utils.flatten(cfg)
                mlflow.log_params({
                    f'{phase}_{k}':v for k,v in cfg.items()
                    })
        mlflow.log_params({f'pretraining_{k}':v for k,v in self.pretraining_space.items()})
        if self.readout_space is not None:
            mlflow.log_params({f'readout_{k}':v for k,v in self.readout_space.items()})

    def teardown(self):
        mlflow.log_artifact(self.log_file) # TODO: log to artifact store more frequently?
        mlflow.end_run()

        if self.cfg.DELETE_LOCAL: # delete locally saved files, since they're already logged as mlflow artifacts -- saves disk storage space
            try:
                logging.info(f'Removing all local files in {self.output_dir}')
                shutil.rmtree(self.output_dir)
            except OSError as e:
                print(f'Error: {e.filename} - {e.strerror}.')

    def __call__(self, args):
        utils.setup_logger(self.log_file, self.cfg.DEBUG)
        self.setup()
        ret = self.call(args)
        self.teardown()
        if ret is None:
            ret = {'loss': 0.0, 'status': STATUS_OK}
        return ret

    @property
    def model_name(self):
        return self.pretraining_cfg.MODEL_NAME

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

class PhysOptModel(metaclass=abc.ABCMeta):
    def __init__(self):
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

