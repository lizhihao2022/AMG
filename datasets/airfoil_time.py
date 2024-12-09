import numpy as np
import os.path as osp
import torch
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset

from utils.graph import construct_coordinate
from utils.metrics import StandardDeviationRecord


class AirfoilTimeDataset:
    def __init__(self, data_path, sample_factor=1, 
                 in_t=1, out_t=1, duration_t=10, 
                 train_batchsize=10, eval_batchsize=10, 
                 normalize=True, **kwargs):        
        self.load_data(data_path=data_path, sample_factor=sample_factor,
                       in_t=in_t, out_t=out_t, duration_t=duration_t, 
                       normalize=normalize)

        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)

    def load_data(self, data_path, sample_factor,
                  in_t, out_t, duration_t, normalize, **kwargs):
        process_path = osp.join(data_path, 'processed.pt') 
        if osp.exists(process_path):
            print('Loading processed data from ', process_path)
            train_data, valid_data, test_data = torch.load(process_path)
        else:
            print('Processing data...')
            train_data = np.load(osp.join(data_path, 'train_0.npy'), allow_pickle=True)
            for i in range(1, 5):
                train_sub = np.load(osp.join(data_path, 'train_' + str(i) + '.npy'), allow_pickle=True)
                train_data = np.concatenate((train_data, train_sub), axis=0)
            valid_data = np.load(osp.join(data_path, 'valid.npy'), allow_pickle=True)
            test_data = np.load(osp.join(data_path, 'test.npy'), allow_pickle=True)
            
            train_data, normalizer = self.pre_process(train_data, mode='train', sample_factor=sample_factor,
                                                      in_t=in_t, out_t=out_t, duration_t=duration_t, 
                                                      normalize=normalize)
            valid_data = self.pre_process(valid_data, mode='valid', sample_factor=sample_factor,
                                        in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                        normalizer=normalizer)
            test_data = self.pre_process(test_data, mode='test', sample_factor=sample_factor,
                                         in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                         normalizer=normalizer)
            print('Saving data...')
            torch.save((train_data, valid_data, test_data), process_path)
            print('Data processed and saved to', process_path)

        self.train_dataset = AirfoilTimeBase(train_data, mode='train')
        self.valid_dataset = AirfoilTimeBase(valid_data, mode='valid')
        self.test_dataset = AirfoilTimeBase(test_data, mode='test')

    def pre_process(self, raw_data, sample_factor, in_t, out_t, duration_t, 
                    mode='train', normalize=False, normalizer=None, **kwargs):
        all_data = []
        x_record = StandardDeviationRecord(num_features=4)
        y_record = StandardDeviationRecord(num_features=4)
        
        for idx in tqdm(range(len(raw_data))):
            data = self.single_process(raw_data[idx], in_t, out_t, duration_t, sample_factor, mode)
            all_data.extend(data)
            if normalize and normalizer is None:
                for d in data:
                    x_record.update(d.x.numpy(), n=d.x.size(0))
                    y_record.update(d.y.numpy(), n=d.y.size(0))
        
        if normalize and normalizer is None:
            x_mean, x_std = x_record.avg(), x_record.std()
            y_mean, y_std = y_record.avg(), y_record.std()
            for data in all_data:
                data.x = (data.x - x_mean) / x_std
                data.y = (data.y - y_mean) / y_std
        elif normalize and normalizer is not None:
            for data in all_data:
                data.x = (data.x - normalizer[0]) / normalizer[1]
                data.y = (data.y - normalizer[2]) / normalizer[3]
        
        if mode == 'train':
            return all_data, [x_record.avg(), x_record.std(), y_record.avg(), y_record.std()]
        else:
            return all_data

    def single_process(self, data, in_t, out_t, duration_t, sample_factor, mode='train'):
        desity = torch.from_numpy(data['density'])
        pressure = torch.from_numpy(data['pressure'])
        velocity = torch.from_numpy(data['velocity'])
        attrs = torch.cat([desity, pressure, velocity], dim=-1)
        mesh_pos = torch.from_numpy(data['mesh_pos'][0])
        cell = torch.from_numpy(data['cells'][0])
        
        if mode == 'train':
            x = attrs[:in_t, ::sample_factor, :]
            y = attrs[in_t:in_t+1, ::sample_factor, :]
            for i in range(1, duration_t):
                x = torch.cat([x, attrs[i:in_t+i, ::sample_factor, :]], dim=0)
                y = torch.cat([y, attrs[in_t+i:in_t+i+1, ::sample_factor, :]], dim=0)
        else:
            x = attrs[out_t-in_t:out_t, ::sample_factor, :]
            y = attrs[out_t:out_t+1, ::sample_factor, :]
            for i in range(1, duration_t):
                x = torch.cat([x, attrs[out_t+i-in_t:out_t+i, ::sample_factor, :]], dim=0)
                y = torch.cat([y, attrs[out_t+i:out_t+i+1, ::sample_factor, :]], dim=0)
        
        all_data = [construct_coordinate(Data(x=x[i], y=y[i], pos=mesh_pos, cell=cell)) for i in range(x.shape[0])]
        
        return all_data


class AirfoilTimeBase(Dataset):
    def __init__(self, data, mode='train', **kwargs):
        self.mode = mode
        self.all_data = data
        
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]
