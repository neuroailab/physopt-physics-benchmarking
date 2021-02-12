import argparse
import datetime
import os
import pickle
import numpy as np
import logging
from hyperopt import STATUS_OK

from physopt.utils import PhysOptObjective
import frozen_physion.modules as modules
from physion.data.pydata import TDWDataset, TDWHumanDataset
from physion.data.config import get_data_cfg
from physion.utils import init_seed, get_subsets_from_datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
import torch.optim as optim
import tensorflow as tf

TRAIN_EPOCHS = 1

def run(
    name,
    datasets,
    seed,
    model_dir,
    write_feat='',
    encoder='vgg',
    regressor='lstm',
    feature_file=None,
    ):
    subsets = get_subsets_from_datasets(datasets)
    data_cfg = get_data_cfg(subsets, debug=True) # TODO: use subsets to get cfg instead?
    data_cfg.freeze()
    print(subsets, data_cfg)

    print(name, datasets, model_dir, encoder, regressor)
    model_file = os.path.join(model_dir, 'model.pt')
    device = torch.device('cpu') # TODO
    config = {
        'name': name,
        'datapaths': datasets,
        'encoder': encoder,
        'regressor': regressor,
        'batch_size': 64,
        'model_file': model_file,
        'feature_file': feature_file,
        'state_len': 4, # number of images as input
        'device': device,
        'data_cfg': data_cfg, # TODO: use merge_from_list/file method
    }
    model =  get_model(config['regressor'], config['encoder']).to(device)
    config['model'] = model
    init_seed(seed)
    if write_feat:
        test(config)
    else:
        train(config)

def get_model(regressor, encoder):
    if regressor == 'mlp':
        return modules.Frozen_MLP(encoder)
    elif regressor == 'lstm':
        return modules.Frozen_LSTM(encoder)
    else:
        raise NotImplementedError

def train(config):
    device = config['device']
    model = config['model']

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    dataset = TDWDataset(
        data_root=config['datapaths'],
        data_cfg=config['data_cfg'],
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
            torch.save(model.state_dict(), config['model_file'])
            print('Saved model checkpoint to: {}'.format(config['model_file']))

def test(config):
    device = config['device']
    encoder = config['encoder']
    state_len = config['state_len']
    model = config['model']

    # load weights
    model.load_state_dict(torch.load(config['model_file']))
    model.eval()

    if 'human' in config['name']:
        dataset = TDWHumanDataset(
            data_root=config['datapaths'],
            data_cfg=config['data_cfg'],
            )
    else:
        dataset = TDWDataset(
            data_root=config['datapaths'],
            data_cfg=config['data_cfg'],
            )
    testloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    extracted_feats = []

    with torch.no_grad():
        for i, data in enumerate(testloader):
            images = data['images'].to(device)

            encoded_states = model.get_seq_enc_feats(images)
            rollout_states = encoded_states[:state_len] # copy over feats for seed frames
            rollout_steps = images.shape[1] - state_len 

            for step in range(rollout_steps):
                input_feats = rollout_states[-state_len:]
                pred_state  = model.dynamics(input_feats) # dynamics model predicts next latent from past latents
                rollout_states.append(pred_state)

            encoded_states = torch.stack(encoded_states, axis=1).cpu().numpy() # TODO: cpu vs detach?
            rollout_states = torch.stack(rollout_states, axis=1).cpu().numpy()
            labels = data['binary_labels'].cpu().numpy()
            print(encoded_states.shape, rollout_states.shape, labels.shape)
            extracted_feats.append({
                'encoded_states': encoded_states,
                'rollout_states': rollout_states,
                'binary_labels': labels,
            })

    pickle.dump(extracted_feats, open(config['feature_file'], 'wb')) 
    print('Saved features to {}'.format(config['feature_file']))

class Objective(PhysOptObjective):
    def __init__(self,
            exp_key,
            seed,
            train_data,
            feat_data,
            output_dir,
            extract_feat,
            encoder,
            regressor,
            ):
        super().__init__(exp_key, seed, train_data, feat_data, output_dir, extract_feat)
        self.encoder = encoder
        self.regressor = regressor

    def __call__(self, *args, **kwargs):
        if self.extract_feat: # save out model features from trained model
            write_feat = 'human' if 'human' in self.feat_data['name'] else 'train'
            run(
                name=self.feat_data['name'],
                datasets=self.feat_data['data'],
                seed=self.seed,
                model_dir=self.model_dir,
                write_feat=write_feat,
                encoder=self.encoder,
                regressor=self.regressor,
                feature_file=self.feature_file,
                ) # TODO: combine args into (YACS) cfg?

        else: # run model training
            run(
                name=self.train_data['name'],
                datasets=self.train_data['data'],
                seed=self.seed,
                model_dir=self.model_dir,
                encoder=self.encoder,
                regressor=self.regressor,
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

class VGGFrozenMLPObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='vgg', regressor='mlp')

class VGGFrozenLSTMObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='vgg', regressor='lstm')

class DEITFrozenMLPObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='deit', regressor='mlp')

class DEITFrozenLSTMObjective(Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, encoder='deit', regressor='lstm')
