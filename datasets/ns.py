import os.path as osp
import scipy.io as sio
import numpy as np

from h5py import File
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from utils import UnitGaussianNormalizer, GaussianNormalizer
from utils.graph import construct_coordinate


class NavierStokesDataset:
    def __init__(self, data_path, sample_factor=1,
                 in_t=1, out_t=1, duration_t=10, 
                 train_batchsize=10, eval_batchsize=10, 
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, 
                 normalize=True, normalizer_type='PGN', **kwargs):
        self.load_data(data_path=data_path, 
                       train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, 
                       sample_factor=sample_factor,
                       in_t=in_t, out_t=out_t, duration_t=duration_t, 
                       normalize=normalize, normalizer_type=normalizer_type)

        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)

    def load_data(self, data_path, sample_factor,
                  train_ratio, valid_ratio, test_ratio, 
                  in_t, out_t, duration_t, 
                  normalize, normalizer_type):
        process_path = data_path.split('.')[0] + '_processed.pt'
        if osp.exists(process_path):
            print('Loading processed data from ', process_path)
            train_data, valid_data, test_data = torch.load(process_path)
        else:
            print('Processing data...')
            try:
                raw_data = sio.loadmat(data_path)
                data = raw_data['u']
            except:
                raw_data = File(data_path, 'r')
                data = np.transpose(raw_data['u'], (3, 1, 2, 0))
            data_size = data.shape[0]
            train_idx = int(data_size * train_ratio)
            valid_idx = int(data_size * (train_ratio + valid_ratio))
            test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
            
            train_data, normalizer = self.pre_process(data[:train_idx], mode='train', sample_factor=sample_factor,
                                                      in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize, 
                                                      normalizer_type=normalizer_type)
            valid_data = self.pre_process(data[train_idx:valid_idx], mode='valid', sample_factor=sample_factor,
                                          in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                          normalizer=normalizer)
            test_data = self.pre_process(data[valid_idx:test_idx], mode='test', sample_factor=sample_factor,
                                         in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                         normalizer=normalizer)
            print('Saving data...')
            torch.save((train_data, valid_data, test_data), process_path)
            print('Data processed and saved to', process_path)

        self.train_dataset = NavierStokesBase(train_data, mode='train')
        self.valid_dataset = NavierStokesBase(valid_data, mode='valid')
        self.test_dataset = NavierStokesBase(test_data, mode='test')

    def pre_process(self, data, sample_factor,
                    in_t, out_t, duration_t, mode='train', normalize=False, 
                    normalizer_type='PGN', normalizer=None, 
                    **kwargs):
        
        if mode == 'train':
            x = data[:, ::sample_factor, ::sample_factor, :in_t]
            y = data[:, ::sample_factor, ::sample_factor, in_t:in_t+1]
            for i in range(1, duration_t):
                x = np.concatenate((x, data[:, ::sample_factor, ::sample_factor, i:in_t+i]), axis=0)
                y = np.concatenate((y, data[:, ::sample_factor, ::sample_factor, in_t+i:in_t+i+1]), axis=0)
        else:
            x = data[:, ::sample_factor, ::sample_factor, out_t-in_t:out_t]
            y = data[:, ::sample_factor, ::sample_factor, out_t:out_t+1]
            for i in range(1, duration_t):
                x = np.concatenate((x, data[:, ::sample_factor, ::sample_factor, out_t+i-in_t:out_t+i]), axis=0)
                y = np.concatenate((y, data[:, ::sample_factor, ::sample_factor, out_t+i:out_t+i+1]), axis=0)
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        if normalize:
            if mode == 'train':
                if normalizer_type == 'PGN':
                    x_normalizer = UnitGaussianNormalizer(x)
                else:
                    x_normalizer = GaussianNormalizer(x)
                x = x_normalizer.encode(x)
            else:
                x = normalizer.encode(x)
        else:
            x_normalizer = None

        grid_x = torch.linspace(0, 1, x.shape[1])
        grid_x = grid_x.reshape(1, x.shape[1], 1, 1).repeat(x.shape[0], 1, x.shape[2], 1)
        grid_y = torch.linspace(0, 1, x.shape[2])
        grid_y = grid_y.reshape(1, 1, x.shape[2], 1).repeat(x.shape[0], x.shape[1], 1, 1)
        
        pos = torch.stack([grid_x, grid_y], dim=-1)

        length = x.shape[0]
        all_data = [construct_coordinate(Data(x=x[i], y=y[i], pos=pos[i])) for i in tqdm(range(length))]

        if mode == 'train':
            return all_data, x_normalizer
        else:
            return all_data


class NavierStokesBase(Dataset):
    """
    A base class for the Navier-Stokes dataset.

    Args:
        x (list): The input data.
        y (list): The target data.
        mode (str, optional): The mode of the dataset. Defaults to 'train'.
        **kwargs: Additional keyword arguments.

    Attributes:
        mode (str): The mode of the dataset.
        x (list): The input data.
        y (list): The target data.
    """

    def __init__(self, data, mode='train', **kwargs):
        self.mode = mode
        self.all_data = data
        
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]
