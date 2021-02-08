import argparse
import torch
import datetime
import os
import pickle
import numpy as np
import logging
from hyperopt import STATUS_OK
from collections import defaultdict
import pdb

from torch.utils import data
import torch.nn.functional as F

from cswm import modules, utils
from physion.config import get_cfg_defaults
from .SVG_FROZEN import get_label_key # TODO: hacky

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                        help='Learning rate.')

    parser.add_argument('--encoder', type=str, default='medium',
                        help='Object extrator CNN size (e.g., `small`).')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Energy scale.')
    parser.add_argument('--hinge', type=float, default=1.,
                        help='Hinge threshold parameter.')

    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Number of hidden units in transition MLP.')
    parser.add_argument('--embedding-dim', type=int, default=2, # TODO: might want to increase this 
                        help='Dimensionality of embedding.')
    parser.add_argument('--action-dim', type=int, default=4,
                        help='Dimensionality of action space.')
    parser.add_argument('--num-objects', type=int, default=5, # TODO: should we make this consistent across models, e.g. 8?
                        help='Number of object slots in model.')
    parser.add_argument('--ignore-action', action='store_true', default=False,
                        help='Ignore action in GNN transition model.')
    parser.add_argument('--copy-action', action='store_true', default=False,
                        help='Apply same action to all object slots.')

    parser.add_argument('--decoder', action='store_true', default=False,
                        help='Train model using decoder and pixel-based loss.')

    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disable CUDA training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42).')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='How many batches to wait before logging'
                             'training status.')
    parser.add_argument('--dataset', type=str,
                        default='data/shapes_train.h5',
                        help='Path to replay buffer.')
    parser.add_argument('--name', type=str, default='none',
                        help='Experiment name.')
    parser.add_argument('--save-folder', type=str,
                        default='checkpoints',
                        help='Path to checkpoints.')

    args, unknown = parser.parse_known_args()
    return args

def run(
    name,
    datasets,
    seed,
    model_dir,
    write_feat='',
    ):
    args = arg_parse() # TODO: change to cfg obj?
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # overwrite args
    args.seed = seed if not write_feat else 0
    args.epochs = 5

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_file = os.path.join(model_dir, 'model.pt')
    meta_file = os.path.join(model_dir, 'metadata.pkl')
    pickle.dump({'args': args}, open(meta_file, "wb"))

    log_file = os.path.join(model_dir, 'log.txt')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file, 'a'))
    print = logger.info

    cfg = get_cfg_defaults()
    #  TODO: change imsize?
    cfg.freeze()
    config = {
        'name': name,
        'datapaths': datasets,
        'model_dir': model_dir,
        'model_file': model_file,
        'data_cfg': cfg.DATA,
    }

    if write_feat:
        test(args, config)
    else:
        train(args, config)

def train(args, config):
    # TODO: implement loading of saved model
    device = torch.device('cuda' if args.cuda else 'cpu')
    dataset = utils.TDWDataset(
        data_root=config['datapaths'],
        label_key='object_data', # TODO: just use object_data here since it doesn't really matter
        data_cfg=config['data_cfg'],
        # size=100, # TODO
        )
    train_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    # Get data sample
    obs = train_loader.__iter__().next()['obs']
    input_shape = obs[0].size()

    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder).to(device)

    model.apply(utils.weights_init)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate)

    if args.decoder:
        if args.encoder == 'large':
            decoder = modules.DecoderCNNLarge(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape).to(device)
        elif args.encoder == 'medium':
            decoder = modules.DecoderCNNMedium(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape).to(device)
        elif args.encoder == 'small':
            decoder = modules.DecoderCNNSmall(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape).to(device)
        decoder.apply(utils.weights_init)
        optimizer_dec = torch.optim.Adam(
            decoder.parameters(),
            lr=args.learning_rate)


    # Train model.
    print('Starting model training...')
    step = 0
    best_loss = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [data_batch[k] for k in ['obs', 'action', 'next_obs']] # to match format of StateTransitionsDataset
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()

            if args.decoder:
                optimizer_dec.zero_grad()
                obs, action, next_obs = data_batch
                objs = model.obj_extractor(obs)
                state = model.obj_encoder(objs)

                rec = torch.sigmoid(decoder(state))
                loss = F.binary_cross_entropy(
                    rec, obs, reduction='sum') / obs.size(0)

                next_state_pred = state + model.transition_model(state, action)
                next_rec = torch.sigmoid(decoder(next_state_pred))
                next_loss = F.binary_cross_entropy(
                    next_rec, next_obs,
                    reduction='sum') / obs.size(0)
                loss += next_loss
            else:
                loss = model.contrastive_loss(*data_batch)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if args.decoder:
                optimizer_dec.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_batch[0]),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / len(data_batch[0])))

            step += 1

        avg_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.6f}'.format(
            epoch, avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['model_file'])
            print('Saved model checkpoint to: {}'.format(config['model_file']))

def test(args, config):
    device = torch.device('cuda' if args.cuda else 'cpu')
    if 'human' in config['name']:
        dataset = utils.TDWHumanDataset(
            data_root=config['datapaths'],
            label_key=get_label_key(config['name']),
            data_cfg=config['data_cfg'],
            )
    else:
        dataset = utils.TDWDataset(
            data_root=config['datapaths'],
            label_key=get_label_key(config['name']),
            data_cfg=config['data_cfg'],
            train=False, # TODO: this should actually still be true, since we want to extract features from the training set
            size=100, # TODO
            )
    eval_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Get data sample
    obs = eval_loader.__iter__().next()['obs']
    input_shape = obs[0].size()

    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder).to(device)

    model.load_state_dict(torch.load(config['model_file']))
    model.eval()

    with torch.no_grad():

        extracted_feats = []
        for batch_idx, data_batch in enumerate(eval_loader):
            labels = data_batch['binary_labels'].cpu().numpy() # get labels before we overwrite batch data 
            labels = np.split(labels, labels.shape[1], axis=1) # split into list along timesteps dim so we can copy states later, each element in list is (BS, 1, x)
            labels = [np.squeeze(lbl, axis=1) for lbl in labels] # remove dim of 1 after split, each element in list is (BS, x)

            data_batch = [data_batch['all_obs'], [data_batch['action']]] # to match format of PathDataset
            data_batch = [[t.to(
                device) for t in tensor] for tensor in data_batch]
            observations, actions = data_batch

            if observations[0].size(0) != args.batch_size:
                continue

            obs = observations[0]
            next_obs = observations[-1]
            rollout_steps = len(observations) - 1

            state = model.obj_encoder(model.obj_extractor(obs)) # (BS, no, embedding_dim)
            next_state = model.obj_encoder(model.obj_extractor(next_obs))

            encoded_states = [torch.flatten(model.obj_encoder(model.obj_extractor(obs)), start_dim=1) for obs in observations] # each element in list is (BS, n_o * embedding_dim)
            rollout_states = [torch.flatten(state, 1)] 

            pred_state = state
            for i in range(rollout_steps): # TODO
                pred_trans = model.transition_model(pred_state, actions[0]) # just use first action since it's always 0
                pred_state = pred_state + pred_trans
                rollout_states.append(torch.flatten(pred_state, 1))

            # Copy initial state to match seq_len, TODO
            state_len = config['data_cfg'].SEQ_LEN - rollout_steps
            encoded_states = _state_copy_helper(encoded_states, state_len)
            rollout_states = _state_copy_helper(rollout_states, state_len)
            labels = _state_copy_helper(labels, state_len)
            

            # TODO: this is duplicated code
            encoded_states = torch.stack(encoded_states, axis=1).cpu().numpy() # TODO: cpu vs detach?
            rollout_states = torch.stack(rollout_states, axis=1).cpu().numpy()
            labels = np.stack(labels, axis=1)
            print(encoded_states.shape, rollout_states.shape, labels.shape)
            extracted_feats.append({
                'rollout_states': rollout_states, # (BS, seq_len , n_o*embedding_dim)
                'encoded_states': encoded_states,
                'binary_labels': labels, 
            })

        # save out features, TODO: move this to utils?
        feat_path = os.path.join(config['model_dir'], 'features', config['name'])
        if not os.path.exists(feat_path):
            os.makedirs(feat_path, exist_ok=True)
        feat_fn = os.path.join(feat_path, 'feat.pkl')
        pickle.dump(extracted_feats, open(feat_fn, 'wb')) 
        print('Saved features to {}'.format(feat_fn))

def _state_copy_helper(states, state_len):
    return states[:1]*state_len + states[1:]

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
