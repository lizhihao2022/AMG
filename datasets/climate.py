import os.path as osp

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from utils import UnitGaussianNormalizer
from utils.graph import construct_coordinate


class ClimateDataset:
    def __init__(self, data_path, sample_factor=1, 
                 in_t=1, out_t=1, duration_t=10, 
                 train_batchsize=10, eval_batchsize=10, 
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, 
                 normalize=True, **kwargs):
        self.load_data(data_path=data_path, sample_factor=sample_factor,
                       train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio,
                       in_t=in_t, out_t=out_t, duration_t=duration_t,
                       normalize=normalize)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)
        
    def load_data(self, data_path, 
                  train_ratio, valid_ratio, test_ratio, 
                  sample_factor, in_t, out_t, duration_t, normalize):
        process_path = data_path.split('.')[0] + '_processed.pt'
        if osp.exists(process_path):
            print('Loading processed data from ', process_path)
            train_data, valid_data, test_data = torch.load(process_path)
        else:
            print('Processing data...')
            raw_data = torch.load(data_path)[1]
            data_size = len(raw_data)
            train_idx = int(data_size * train_ratio)
            valid_idx = int(data_size * (train_ratio + valid_ratio))
            test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
            
            train_data, normalizer = self.pre_process(raw_data[:train_idx], mode='train', sample_factor=sample_factor,
                                                      in_t=in_t, out_t=out_t, duration_t=duration_t, 
                                                      normalize=normalize)
            valid_data = self.pre_process(raw_data[train_idx:valid_idx], mode='valid', sample_factor=sample_factor,
                                        in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                        normalizer=normalizer)
            test_data = self.pre_process(raw_data[valid_idx:test_idx], mode='test', sample_factor=sample_factor,
                                         in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                         normalizer=normalizer)
            print('Saving data...')
            torch.save((train_data, valid_data, test_data), process_path)
            print('Data processed and saved to', process_path)
        self.train_dataset = ClimateBase(train_data, mode='train')
        self.valid_dataset = ClimateBase(valid_data, mode='valid')
        self.test_dataset = ClimateBase(test_data, mode='test')
    
    def pre_process(self, raw_data, sample_factor, in_t, out_t, duration_t, 
                    mode='train', normalize=False, normalizer=None, **kwargs):
        if mode == 'train':
            x = raw_data[:, :in_t, ::sample_factor, ::sample_factor, :]
            y = raw_data[:, in_t:in_t+1, ::sample_factor, ::sample_factor, :]
            for i in range(1, duration_t):
                x = torch.cat((x, raw_data[:, i:i+in_t, ::sample_factor, ::sample_factor, :]), dim=0)
                y = torch.cat((y, raw_data[:, i+in_t:i+in_t+1, ::sample_factor, ::sample_factor, :]), dim=0)
        else:
            x = raw_data[:, out_t-in_t:out_t, ::sample_factor, ::sample_factor, :]
            y = raw_data[:, out_t:out_t+1, ::sample_factor, ::sample_factor, :]
            for i in range(1, duration_t):
                x = torch.cat((x, raw_data[:, out_t+i-in_t:out_t+i, ::sample_factor, ::sample_factor, :]), dim=0)
                y = torch.cat((y, raw_data[:, out_t+i:out_t+i+1, ::sample_factor, ::sample_factor, :]), dim=0)
        
        if normalize:
            if mode == 'train':
                x_normalizer = UnitGaussianNormalizer(x)
                x = x_normalizer.encode(x)
            else:
                x = normalizer.encode(x)
        else:
            x_normalizer = None
                    
        latitudes = torch.linspace(-90, 90, x.shape[2])
        longitudes = torch.linspace(-180, 180, x.shape[3])
        
        lon_grid, lat_grid = torch.meshgrid(longitudes, latitudes)
        
        pos = torch.stack([lon_grid, lat_grid], dim=-1)
        
        length = x.shape[0]
        all_data = [construct_coordinate(Data(x=x[i], y=y[i], pos=pos)) for i in tqdm(range(length))]
        
        if mode == 'train':
            return all_data, x_normalizer
        else:
            return all_data

        

class ClimateBase(Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
