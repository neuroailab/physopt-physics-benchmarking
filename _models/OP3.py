import os
import pickle
import numpy as np
from hyperopt import STATUS_OK
from argparse import ArgumentParser

import torch
import torch.nn as nn
import tensorflow as tf

  
from op3.launchers.launcher_util import run_experiment
import op3.torch.op3_modules.op3_model as op3_model
from op3.torch.op3_modules.op3_trainer import TrainingScheduler, OP3Trainer
from op3.torch.data_management.dataset import BlocksDataset, CollideDataset #TODO

def init_seed(seed): # TODO: move to utils in physion package?
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_dataset(data_path, label_key, train=True, size=None, batchsize=8, static=True):
    tf_dataset = CollideDataset(data_path, label_key, size=size)
    dataset = BlocksDataset(tf_dataset, batchsize=batchsize, shuffle=True)
    T = dataset.dataset.seq_len
    print('Dataset Size: {}'.format(len(dataset.dataset)))
    print('Dataloader Size: {}'.format(len(dataset.dataloader)))
    return dataset, T

def train_vae(variant):
    from op3.core import logger

    ######Dataset loading######
    train_path = [os.path.join(path, 'new_tfdata') for path in variant['datapath']]
    test_path = train_path
    bs = variant['training_args']['batch_size']
    train_size = 100 if variant['debug'] == 1 else None
    label_key = variant['label_key']

    static = (variant['schedule_args']['schedule_type'] == 'static_iodine')  # Boolean
    train_dataset, max_T = load_dataset(train_path, label_key, train=True, batchsize=bs, size=train_size, static=static)
    test_dataset, _ = load_dataset(test_path, label_key, train=False, batchsize=bs, size=100, static=static)
    print(logger.get_snapshot_dir())

    ######Model loading######
    op3_args = variant["op3_args"]
    m = op3_model.create_model_v2(op3_args, op3_args['det_repsize'], op3_args['sto_repsize'], action_dim=train_dataset.action_dim)
    if variant['dataparallel']:
        m = torch.nn.DataParallel(m)
    # m.cuda() # TODO

    ######Training######
    scheduler = TrainingScheduler(**variant["schedule_args"], max_T = max_T)
    t = OP3Trainer(train_dataset, test_dataset, m, scheduler, **variant["training_args"])

    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        print('Starting epoch {}'.format(epoch))
        should_save_imgs = (epoch % save_period == 0)
        print('Start training')
        train_stats = t.train_epoch(epoch)
        print('Start testing')
        test_stats = t.test_epoch(epoch, train=False, batches=1, save_reconstruction=should_save_imgs)
        t.test_epoch(epoch, train=True, batches=1, save_reconstruction=should_save_imgs) # TODO: Why do they do this??
        for k, v in {**train_stats, **test_stats}.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        t.save_model()

def run(
    name,
    datasets,
    seed,
    model_dir,
    write_feat='',
    ):
    init_seed(seed)
    if write_feat:
        test()
    else:
        train()

def train():
    parser = ArgumentParser()
    parser.add_argument('-de', '--debug', type=int, default=1)  # Note: Change this to 0 to run on the entire dataset!
    parser.add_argument('-m', '--mode', type=str, default='here_no_doodad')  # Relevant options: 'here_no_doodad', 'local_docker', 'ec2'
    args = parser.parse_args()
    args.variant = 'collide' # TODO, just for exp_prefix since variant is already hardcoded below
    
    variant = dict( # TODO
        op3_args=dict(
            refinement_model_type="size_dependent_conv",  # size_dependent_conv, size_dependent_conv_no_share
            decoder_model_type="reg",  # reg, reg_no_share
            dynamics_model_type="reg_ac32",  # reg_ac32, reg_ac32_no_share
            sto_repsize=64,
            det_repsize=64,
            extra_args=dict(
                beta=1e-2,
                deterministic_sampling=False
            ),
            K=8
        ),
        schedule_args=dict(  # Arguments for TrainingScheduler
            seed_steps=4,
            T=5,  # Max number of steps into the future we want to go or max length of a schedule
            # schedule_type='single_step_physics',  # single_step_physics, curriculum, static_iodine, rprp, next_step, random_alternating
            schedule_type='custom',  # single_step_physics, curriculum, static_iodine, rprp, next_step, random_alternating
        ),
        training_args=dict(  # Arguments for OP3Trainer
            batch_size=4,  # Change to appropriate constant based off dataset size
            lr=3e-4,
        ),
        num_epochs=300,
        save_period=1,
        dataparallel=True, # Use multiple GPUs?
        debug=False,
        datapath=['/mnt/fs4/mrowca/neurips/images/rigid/collide2_new'],
        label_key='is_colliding_dynamic',
    )

    variant['debug'] = args.debug
    run_experiment(
        train_vae,
        exp_prefix='{}'.format(args.variant),
        mode=args.mode,
        variant=variant,
        use_gpu=False,  # Turn on if you have a GPU TODO
        seed=None, # TODO
    )

class Objective():
    def __init__(self,
            exp_key,
            seed,
            train_data,
            feat_data,
            output_dir,
            extract_feat):
        self.exp_key = exp_key
        self.seed = seed
        self.train_data = train_data
        self.feat_data = feat_data
        self.output_dir = output_dir
        self.extract_feat = extract_feat
        self.model_dir = self.get_model_dir()


    def get_model_dir(self):
        model_dir = os.path.join(self.output_dir, self.train_data['name'], str(self.seed), 'model')                                     
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return 
               

    def __call__(self, *args, **kwargs):
        if self.extract_feat: # save out model features from trained model
            write_feat = 'human' if 'human' in self.feat_data['name'] else 'train'
            run(
                name=self.feat_data['name'],
                datasets=self.feat_data['data'],
                seed=self.seed,
                model_dir=self.model_dir,
                write_feat=write_feat,
                ) # TODO: add args

        else: # run model training
            run(
                name=self.train_data['name'],
                datasets=self.train_data['data'],
                seed=self.seed,
                model_dir=self.model_dir,
                ) # TODO: add args

        return {
                'loss': 0.0,
                'status': STATUS_OK,
                'exp_key': self.exp_key,
                'seed': self.seed,
                'train_data': self.train_data,
                'feat_data': self.feat_data,
                'model_dir': self.model_dir,
                }

