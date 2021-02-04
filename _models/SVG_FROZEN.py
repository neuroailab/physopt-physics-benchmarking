import argparse
import datetime
import os
import pickle
import numpy as np
import logging
from hyperopt import STATUS_OK

import physion.modules
from physion.data import TDWDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
import torch.optim as optim
import tensorflow as tf

TRAIN_EPOCHS = 5

COLLIDE_CONFIG = {
        'binary_labels': ['is_colliding_dynamic'],
        'train_shift': [30, 1024, 1],
        'train_len': 32470,
        'test_shift': [30, 1024, 16],
        'test_len': 62,
        }

TOWER_CONFIG = {
        'binary_labels': ['is_stable'],
        'train_shift': [0, 1024, 1],
        'train_len': 63360,
        'test_shift': [2, 1024, 32],
        'test_len': 384,
        }

CONTAIN_CONFIG = {
        'binary_labels': ['object_data'],
        'train_shift': [30, 1024, 1],
        'train_len': 36355,
        'test_shift': [30, 1024, 16],
        'test_len': 28,
        }

CLOTH_CONFIG = {
        'binary_labels': ['object_category'],
        'train_shift': [0, 1024, 1],
        'train_len': 63360,
        'test_shift': [2, 1024, 64],
        'test_len': 192,
        }

ROLL_VS_SLIDE_CONFIG = {
        'binary_labels': ['is_rolling'],
        'train_shift': [32, 1024, 1],
        'train_len': 74572 // 4,
        'test_shift': [32, 1024, 64],
        'test_len': 320 // 4,
        }


def get_config(subset):
    if 'collide' in subset:
        return COLLIDE_CONFIG
    elif 'tower' in subset:
        return TOWER_CONFIG
    elif 'contain' in subset:
        return CONTAIN_CONFIG
    elif 'cloth' in subset:
        return CLOTH_CONFIG
    elif 'roll' in subset:
        return ROLL_VS_SLIDE_CONFIG
    elif 'slide' in subset:
        return ROLL_VS_SLIDE_CONFIG
    else:
        raise ValueError("Unkown config for subset: %s" % subset)

def filter_rule(data, keys):
    assert all(k in keys for k in ['is_moving', 'is_acting']), keys
    return tf.logical_and(data['is_moving'], tf.logical_not(data['is_acting']))

DATA_PARAMS = { # TODO: move to config?
    'enqueue_batch_size': 256,
    'map_pcall_num': 4,
    'sequence_len': 10,
    'buffer_size': 16,
    'batch_size': 1,
    'shift_selector': slice(30, 1024, 1), # remove falling part of sequence TODO: have different settings for each dataset
    'test': False, # True, # determines if tf dataset doesn't loop
    'main_source_key': 'full_particles',
    'sources': ['images', 'reference_ids', 'object_data'],
    'delta_time': 1,
    'filter_rule': (filter_rule, ['is_moving', 'is_acting']),
    'shuffle': False,
    'seed': 0,
    'use_legacy_sequence_mode': 1,
    'subsources': [],
    }

def run(
    name,
    datasets,
    seed,
    model_dir,
    write_feat='',
    ):
    print(name, datasets, model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    device = torch.device('cpu') # TODO
    config = {
        'name': name,
        'datapaths': datasets,
        'encoder': 'deit',
        'regressor': 'lstm',
        'batch_size': 64,
        'model_dir': model_dir,
        'state_len': 4, # number of images as input
        'device': device,
    }
    model =  get_model(config['regressor'], config['encoder']).to(device)
    config['model'] = model
    init_seed(seed)
    if write_feat:
        test(config)
    else:
        train(config)

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_model(regressor, encoder):
    if regressor == 'mlp':
        return physion.modules.Frozen_MLP(encoder)
    elif regressor == 'lstm':
        return physion.modules.Frozen_LSTM(encoder)
    else:
        raise NotImplementedError

def train(config):
    device = config['device']
    model = config['model']
    model_file = os.path.join(config['model_dir'], 'model.pt')

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    dataset = TDWDataset(
        data_root=config['datapaths'],
        label_key='object_data', # just use object_data here since it doesn't really matter
        DATA_PARAMS=DATA_PARAMS,
        )
    trainloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    best_loss = 1e9
    for epoch in range(TRAIN_EPOCHS):
        running_loss = 0.
        for i, data in enumerate(trainloader):
            images = data['images'].to(device)
            inputs = images[:,:4]
            labels = model.get_encoder_feats(images[:,4])
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss/(i+1)
            print(avg_loss)

        # save model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)

def get_label_key(name):
    print(name, get_config(name)['binary_labels'][0])
    return get_config(name)['binary_labels'][0]

def test(config):
    device = config['device']
    encoder = config['encoder']
    state_len = config['state_len']
    model = config['model']

    dataset = TDWDataset(
        data_root=config['datapaths'],
        label_key=get_label_key(config['name']),
        train=False,
        DATA_PARAMS=DATA_PARAMS,
        size=10, # TODO
        )
    testloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    extracted_feats = []
    for i, data in enumerate(testloader):
        images = data['images'].to(device)
        labels = data['binary_labels']

        encoded_states = model.get_seq_enc_feats(images)
        rollout_states = encoded_states[:state_len] # copy over feats for seed frames
        rollout_steps = images.shape[1] - state_len 

        for step in range(rollout_steps):
            input_feats = rollout_states[-state_len:]
            pred_state  = model.dynamics(input_feats) # dynamics model predicts next latent from past latents
            rollout_states.append(pred_state)

        extracted_feats.append({ # TODO: to cpu?
            'encoded_states': encoded_states,
            'rollout_states': rollout_states,
            'binary_labels': labels,
        })

    # Save out features
    feat_path = os.path.join(config['model_dir'], 'features')
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
    feat_fn = os.path.join(feat_path, config['name']+'.pkl')
    pickle.dump(extracted_feats, open(feat_fn, 'wb')) 
    print('Saved features to {}'.format(feat_fn))

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
        return os.path.join(self.output_dir, self.train_data['name'],
                str(self.seed), 'model')

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
