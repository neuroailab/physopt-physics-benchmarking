import os
import io
import glob
import h5py
import json
from PIL import Image
import numpy as np
import logging
import torch
from  torch.utils.data import Dataset

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.ERROR)

class TDWDataset(Dataset):
    def __init__(
        self,
        data_root,
        imsize,
        seq_len,
        state_len,
        random_seq=True,
        debug=False,
        subsample_factor=6, # TODO: change default to 1 probably
        ):
        assert isinstance(data_root, list)
        self.imsize = imsize
        self.seq_len = seq_len
        self.state_len = state_len # not necessarily always used
        assert self.seq_len > self.state_len, 'Sequence length {} must be greater than state length {}'.format(self.seq_len, self.state_len)
        self.random_seq = random_seq # whether sequence should be sampled randomly from whole video or taken from the beginning
        self.debug = debug
        self.subsample_factor = subsample_factor

        self.hdf5_files = []
        for path in data_root:
            assert '*.hdf5' in path
            files = sorted(glob.glob(path))
            files = [fn for fn in files if 'tfrecords' not in fn]
            self.hdf5_files.extend(files)
            logging.info('Processed {} with {} files'.format(path, len(files)))
        self.N = min(100, len(self.hdf5_files)) if debug else len(self.hdf5_files)
        logging.info('Dataset len: {}'.format(self.N))

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.get_seq(index)

    def get_seq(self, index):
        with h5py.File(self.hdf5_files[index], 'r') as f: # load ith hdf5 file from list
            # extract images and labels
            images = []
            labels = []
            for frame in f['frames']:
                img = f['frames'][frame]['images']['_img'][()]
                img = np.array(Image.open(io.BytesIO(img))) # (256, 256, 3)
                img = (img / 255.).astype(np.float32) # convert from [0, 255] to [0, 1]
                images.append(img)
                lbl = f['frames'][frame]['labels']['target_contacting_zone'][()]
                labels.append(lbl)
            stimulus_name = f['static']['stimulus_name'][()]
        images = torch.from_numpy(np.array(images))
        labels = np.ones_like(labels) if np.any(labels) else np.zeros_like(labels) # Get single label over whole sequence
        labels = torch.from_numpy(np.array(labels))

        images = images[::self.subsample_factor]
        images = images.float().permute(0, 3, 1, 2) # (T, 3, D, D)
        images = torch.nn.functional.interpolate(images, size=self.imsize)

        labels = labels[::self.subsample_factor]
        labels = torch.unsqueeze(labels, -1)

        assert images.shape[0] >= self.seq_len, 'Images must be at least len {}, but are shape {}'.format(self.seq_len, images.shape)
        if self.random_seq: # randomly sample sequence of seq_len
            start_idx = torch.randint(0, images.shape[0]-self.seq_len+1, (1,))[0]
            images = images[start_idx:start_idx+self.seq_len]
            labels = labels[start_idx:start_idx+self.seq_len]
        else: # get first seq_len # of frames
            images = images[:self.seq_len]
            labels = labels[:self.seq_len]

        sample = {
            'images': images,
            'binary_labels': labels,
            'stimulus_name': stimulus_name,
        }
        return sample
