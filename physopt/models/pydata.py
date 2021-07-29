import os
import copy
import pickle
import numpy as np
import torch
import tensorflow as tf
from physion.data.config import get_data_cfg
from physion.data.tfdata import SequenceNewDataProvider as DataProvider
from physion.utils import get_subsets_from_datasets

def filter_rule(data, keys):
    assert all(k in keys for k in ['is_moving', 'is_acting']), keys
    return tf.logical_and(data['is_moving'], tf.logical_not(data['is_acting']))

DEFAULT_DATA_PARAMS = {
    'sources': ['images', 'reference_ids', 'object_data'],
    'filter_rule': (filter_rule, ['is_moving', 'is_acting']),
    'enqueue_batch_size': 256,
    'buffer_size': 16,
    'map_pcall_num': 1,
    'shuffle': False, # shuffling will be done by pytorch dataprovider
    'use_legacy_sequence_mode': True,
    'main_source_key': 'images',
}

class TDWDatasetBase(object):
    def __init__(
            self,
            imsize,
            seq_len,
            state_len,
            train=True,
            debug=False,
            ):
        self.imsize = imsize
        self.seq_len = seq_len
        self.state_len = state_len # not necessarily always used
        assert self.seq_len > self.state_len, 'Sequence length {} must be greater than state length {}'.format(self.seq_len, self.state_len)
        self.train = train
        self.debug = debug

    def __len__(self):
        return self.N # assumes self.N is set

    def __getitem__(self, index):
        return self.get_seq(index)

    def get_seq(self, index):
        try:
            batch = next(self.data) # assumes self.data is set
        except StopIteration:
            print('End of Dataset')
            raise

        # TODO: check that first dim is 1 before doing [0]
        batch_images = self._to_tensor(batch['images'][0]) # (seq_len, image_size, image_size, 3)
        assert batch_images.shape[0] == self.seq_len, 'size of images {} must match seq_len {}'.format(batch_images.shape, self.seq_len)
        batch_images = batch_images.float().permute(0, 3, 1, 2) # (T, 3, D, D)
        batch_images = torch.nn.functional.interpolate(batch_images, size=self.imsize)

        batch_labels = self._to_tensor(batch[self.label_key][0]) # (seq_len, ...)
        assert batch_labels.shape[0] == self.seq_len, 'size of labels {} must match seq_len {}'.format(batch_labels.shape, self.seq_len)

        sample = {
            'images': batch_images,
            'binary_labels': batch_labels,
            }
        # add human_prob
        if 'human_prob' in batch:
            sample['human_prob'] = batch['human_prob'][0] # (4,)

        return sample

    @staticmethod
    def _to_tensor(arr): # convert to torch tensor
        if not isinstance(arr, np.ndarray):
            try:
                arr = arr.numpy()
            except:
                print('{} type cannot be converted to np.ndarray'.format(type(arr)))
        return torch.from_numpy(arr)

class TDWDataset(TDWDatasetBase):
    """Data handler which loads the TDW images."""

    def __init__(
            self,
            data_root,
            *args,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        self.data = self.build_data(data_root)

    @staticmethod
    def _get_datapaths(data_root, train):
        if not isinstance(data_root, list):
            data_root = [data_root]
        if train:
            datapaths = [os.path.join(dir, 'new_tfdata') for dir in data_root]
        else:
            datapaths = [os.path.join(dir, 'new_tfvaldata') for dir in data_root]
        return datapaths

    def _set_size(self, data_cfg):
        if self.train:
            self.N = data_cfg.DATA.TRAIN_SIZE
        else:
            self.N = data_cfg.DATA.TEST_SIZE
        print('Dataset size: {}'.format(self.N))

    def build_data(self, data_root): # also sets size
        data_cfg = get_data_cfg(get_subsets_from_datasets(data_root), self.debug)
        data_cfg.freeze()
        self._set_size(data_cfg)
        self.label_key = data_cfg.DATA.LABEL_KEY # TODOj

        print('Building TF Dataset')
        tfdata_params = copy.deepcopy(DEFAULT_DATA_PARAMS)
        tfdata_params['sources'].append(self.label_key)
        tfdata_params['data'] = self._get_datapaths(data_root, self.train)
        tfdata_params['sequence_len'] = self.seq_len
        tfdata_params['shift_selector'] = slice(*data_cfg.DATA.SHIFTS)

        data_provider = DataProvider(**tfdata_params)
        batch_size = 1 # only use bs=1 since get_seq gets once sample at a time
        dataset = data_provider.build_datasets(batch_size)
        return iter(dataset)

class TDWHumanDataset(TDWDatasetBase):
    def __init__(
            self,
            data_root, 
            *args,
            **kwargs,
            ):
            super().__init__(*args, **kwargs)
            self.data =  self.build_data(data_root)

    @staticmethod
    def _get_datapaths(data_root):
        if not isinstance(data_root, list):
            data_root = [data_root]
        datapaths = [os.path.join(path, 'raw_data.pickle') for path in data_root]
        return datapaths

    def _set_size(self, data):
        self.N = len(data) 
        print('Dataset size: {}'.format(self.N))

    def build_data(self, data_root): # also sets size
        data_cfg = get_data_cfg(get_subsets_from_datasets(data_root), self.debug)
        data_cfg.freeze()
        self.label_key = data_cfg.DATA.LABEL_KEY # TODO

        data = []
        for path in self._get_datapaths(data_root):
            data.extend(pickle.load(open(path, 'rb')))
        self._set_size(data) # must do before converting to iterator
        return iter(data)

