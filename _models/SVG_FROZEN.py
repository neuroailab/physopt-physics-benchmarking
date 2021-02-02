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

def filter_rule(data, keys):
    assert all(k in keys for k in ['is_moving', 'is_acting']), keys
    return tf.logical_and(data['is_moving'], tf.logical_not(data['is_acting']))

DATA_PARAMS = { # TODO: move to config?
    'enqueue_batch_size': 256,
    'map_pcall_num': 4,
    'sequence_len': 10,
    'buffer_size': 16,
    'batch_size': 1,
    'shift_selector': slice(30, 1024, 1), # remove falling part of sequence
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
    if write_feat:
        test()
    else:
        train()

def train():
    device = torch.device('cpu') # TODO
    encoder = 'deit' # TODO
    # model = physion.modules.Frozen_MLP(encoder).to(device)
    model = physion.modules.Frozen_LSTM(encoder).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    dataset = TDWDataset(
        data_root=['/mnt/fs4/mrowca/neurips/images/rigid/collide2_new/new_tfdata'],
        label_key='is_colliding_dynamic',
        DATA_PARAMS=DATA_PARAMS,
        )
    trainloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(5):
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
            print(running_loss/(i+1))

def test():
    device = torch.device('cpu') # TODO
    encoder = 'deit' # TODO
    # model = physion.modules.Frozen_MLP(encoder).to(device)
    model = physion.modules.Frozen_LSTM(encoder).to(device)

    dataset = TDWDataset(
        data_root=['/mnt/fs4/mrowca/neurips/images/rigid/collide2_new/new_tfvaldata'],
        label_key='is_colliding_dynamic',
        DATA_PARAMS=DATA_PARAMS,
        size=10, # TODO
        )
    testloader = DataLoader(dataset, batch_size=64, shuffle=True)

    state_len = 4
    extracted_feats = []
    for i, data in enumerate(testloader):
        images = data['images'].to(device)
        labels = data['binary_labels']

        encoded_states = model.get_seq_enc_feats(images)
        rollout_states = encoded_states[:state_len]
        rollout_steps = images.shape[1] - state_len 

        for step in range(rollout_steps):
            input_feats = rollout_states[-state_len:]
            pred_state  = model.dynamics(input_feats) # TODO: implement rollout instead of using images
            rollout_states.append(pred_state)

        extracted_feats.append({ # TODO: to cpu?
            'encoded_states': encoded_states,
            'rollout_states': rollout_states,
            'binary_labels': labels,
        })

    # Save out features

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
