import os
import pickle
import numpy as np
from hyperopt import STATUS_OK
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import tensorflow as tf

  
from op3.launchers.launcher_util import run_experiment
import op3.torch.op3_modules.op3_model as op3_model
from op3.torch.op3_modules.op3_trainer import TrainingScheduler, OP3Trainer
from op3.torch.data_management.dataset import BlocksDataset, CollideDataset, CollideHumanDataset # TODO: rename collide
from op3.core import logger

from physion.data.config import get_data_cfg
from physion.utils import init_seed, get_subsets_from_datasets
from physopt.utils import PhysOptObjective

def load_dataset(data_path, data_cfg, train=True, batchsize=8, human=False):
    if human:
        tf_dataset = CollideHumanDataset(data_path, data_cfg)
    else:
        tf_dataset = CollideDataset(data_path, data_cfg, train=train)
    dataset = BlocksDataset(tf_dataset, batchsize=batchsize, shuffle=True) # basically only acts as a wrapper for creating dataloader from dataset and set action_dim
    T = dataset.dataset.seq_len
    print('Dataset Size: {}'.format(len(dataset.dataset)))
    print('Dataloader Size: {}'.format(len(dataset.dataloader)))
    return dataset, T

def train_vae(variant):
    ######Dataset loading######
    datapaths = variant['datapath']
    bs = variant['training_args']['batch_size']
    data_cfg = variant['data_cfg']

    train_dataset, max_T = load_dataset(datapaths, data_cfg, train=True, batchsize=bs)
    test_dataset, _ = load_dataset(datapaths, data_cfg, train=False, batchsize=bs)
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
        t.test_epoch(epoch, train=True, batches=1, save_reconstruction=should_save_imgs) # evals on training set
        for k, v in {**train_stats, **test_stats}.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()

        # save model
        torch.save(t.model.state_dict(), variant['model_file'])
        print('Saved model ckpt to: {}'.format(variant['model_file']))

def test_vae(variant):
    ######Dataset loading######
    datapaths = variant['datapath']
    bs = 2 # variant['training_args']['batch_size'] TODO: reduce gpu memory usage
    data_cfg = variant['data_cfg']

    human = 'human' in variant['name']
    train_dataset, max_T = load_dataset(datapaths, data_cfg, train=True, batchsize=bs, human=human)
    test_dataset, _ = load_dataset(datapaths, data_cfg, train=False, batchsize=bs, human=human)
    print(logger.get_snapshot_dir())

    ######Model loading######
    op3_args = variant["op3_args"]
    m = op3_model.create_model_v2(op3_args, op3_args['det_repsize'], op3_args['sto_repsize'], action_dim=train_dataset.action_dim)

    state_dict = torch.load(variant['model_file'])
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' in k:
            name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    m.load_state_dict(new_state_dict)

    # if variant['dataparallel']:
    #     m = torch.nn.DataParallel(m)
    # m.cuda()

    ######Training######
    variant['schedule_args']['T'] = 10 # overwrite T TODO
    scheduler = TrainingScheduler(**variant["schedule_args"], max_T = max_T)
    t = OP3Trainer(train_dataset, test_dataset, m, scheduler, **variant["training_args"])

    # extracts feature from training dataset
    rollout_hidden_states, encoded_hidden_states, binary_labels = t.test_discriminative_epoch(train=True, batches=len(train_dataset.dataset)//bs) # (N*B, T*K*R) TODO: assumes we're computing features on train dataset
    print('Num samples:{}'.format(binary_labels.shape[0]))
    extracted_feats = [{
        'rollout_states': rollout_hidden_states,
        'encoded_states': encoded_hidden_states,
        'binary_labels': binary_labels,
        }] # list of dicts

    # save out features 
    pickle.dump(extracted_feats, open(variant['feature_file'], 'wb')) 
    print('Saved features to {}'.format(variant['feature_file']))

def run(
    name,
    datasets,
    seed,
    model_dir,
    write_feat='',
    feature_file=None,
    ):
    init_seed(seed)

    parser = ArgumentParser()
    parser.add_argument('-de', '--debug', type=int, default=1)  # Note: Change this to 0 to run on the entire dataset!
    parser.add_argument('-m', '--mode', type=str, default='here_no_doodad')  # Relevant options: 'here_no_doodad', 'local_docker', 'ec2'
    args, _ = parser.parse_known_args()

    subsets = get_subsets_from_datasets(datasets)
    data_cfg = get_data_cfg(subsets, debug=True) # TODO: use subsets to get cfg instead?
    data_cfg.IMSIZE = 64
    data_cfg.freeze()
    
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
            batch_size=16,  # TODO: Change to appropriate constant based off dataset size
            lr=3e-4,
        ),
        num_epochs=1,
        save_period=1,
        dataparallel=True, # Use multiple GPUs?
        debug=False,
        datapath=datasets,
        model_file=os.path.join(model_dir, 'model.pt'),
        feature_file=feature_file,
        name=name,
        data_cfg=data_cfg,
    )

    variant['debug'] = args.debug
    if write_feat:
        run_experiment(
            test_vae,
            exp_prefix='{}'.format(name),
            mode=args.mode,
            variant=variant,
            use_gpu=False,  # Turn on if you have a GPU TODO
            seed=None, # TODO
        )
    else:
        run_experiment(
            train_vae,
            exp_prefix='{}'.format(name),
            mode=args.mode,
            variant=variant,
            use_gpu=False,  # Turn on if you have a GPU TODO
            seed=None, # TODO
        )

class Objective(PhysOptObjective):
    def __call__(self, *args, **kwargs):
        if self.extract_feat: # save out model features from trained model
            write_feat = 'human' if 'human' in self.feat_data['name'] else 'train'
            run(
                name=self.feat_data['name'],
                datasets=self.feat_data['data'],
                seed=self.seed,
                model_dir=self.model_dir,
                write_feat=write_feat,
                feature_file=self.feature_file,
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

