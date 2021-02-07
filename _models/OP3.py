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
from op3.torch.data_management.dataset import BlocksDataset, CollideDataset #TODO

from .SVG_FROZEN import get_label_key # TODO: hacky

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
    label_key = 'object_data' # TODO: unused for training

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
        # save model
        torch.save(t.model.state_dict(), variant['model_file'])
        print('Saved model ckpt to: {}'.format(variant['model_file']))

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

def test_vae(variant):
    from op3.core import logger

    ######Dataset loading######
    train_path = [os.path.join(path, 'new_tfdata') for path in variant['datapath']]
    test_path = [os.path.join(path, 'new_tfvaldata') for path in variant['datapath']]
    bs = 2 # variant['training_args']['batch_size'] TODO: reduce gpu memory usage
    train_size = 4 # TODO
    test_size = 4 # TODO
    label_key = variant['label_key']

    static = (variant['schedule_args']['schedule_type'] == 'static_iodine')  # Boolean
    train_dataset, max_T = load_dataset(train_path, label_key, train=True, batchsize=bs, size=train_size, static=static)
    test_dataset, _ = load_dataset(test_path, label_key, train=False, batchsize=bs, size=test_size, static=static)
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

    # t.test_epoch(0, train=False, batches=1, save_reconstruction=True)
    # t.test_epoch(0, train=True, batches=1, save_reconstruction=True)

    # rollout_hidden_states, encoded_hidden_states, binary_labels = t.test_discriminative_epoch(train=False, batches=test_size//bs) # (N*B, T*K*R)
    # print('Num samples:{}'.format(binary_labels.shape[0]))
    # test_feat = {
    #     'rollout_states': rollout_hidden_states,
    #     'encoded_states': encoded_hidden_states,
    #     'binary_labels': binary_labels,
    #     }
    # pickle.dump(test_feat, open(logger.get_snapshot_dir()+'/test_feat.pkl', 'wb'))

    rollout_hidden_states, encoded_hidden_states, binary_labels = t.test_discriminative_epoch(train=True, batches=train_size//bs) # (N*B, T*K*R)
    print('Num samples:{}'.format(binary_labels.shape[0]))
    extracted_feats = [{
        'rollout_states': rollout_hidden_states,
        'encoded_states': encoded_hidden_states,
        'binary_labels': binary_labels,
        }] # list of dicts

    # save out features, TODO: move this to utils?
    feat_path = os.path.join(variant['model_dir'], 'features', variant['name'])
    if not os.path.exists(feat_path):
        os.makedirs(feat_path, exist_ok=True)
    feat_fn = os.path.join(feat_path, 'feat.pkl')
    pickle.dump(extracted_feats, open(feat_fn, 'wb')) 
    print('Saved features to {}'.format(feat_fn))

def run(
    name,
    datasets,
    seed,
    model_dir,
    write_feat='',
    ):
    init_seed(seed)

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
            batch_size=32,  # Change to appropriate constant based off dataset size
            lr=3e-4,
        ),
        num_epochs=300,
        save_period=1,
        dataparallel=True, # Use multiple GPUs?
        debug=False,
        datapath=datasets,
        label_key=get_label_key(name),
        model_dir=model_dir,
        model_file=os.path.join(model_dir, 'model.pt'),
        name=name,
    )

    variant['debug'] = args.debug
    if write_feat:
        run_experiment(
            test_vae,
            exp_prefix='{}'.format(args.variant),
            mode=args.mode,
            variant=variant,
            use_gpu=False,  # Turn on if you have a GPU TODO
            seed=None, # TODO
        )
    else:
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
        return model_dir
               

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

