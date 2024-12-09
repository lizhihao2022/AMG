import pickle
import os.path as osp
import torch
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset

from utils.graph import construct_coordinate
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class PoissonDataset:
    def __init__(self, data_path, train_batchsize=10, eval_batchsize=10, 
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, 
                 normalize=True, normalizer_type='PGN', 
                 **kwargs):        
        self.load_data(data_path=data_path, train_ratio=train_ratio, 
                       valid_ratio=valid_ratio, test_ratio=test_ratio, 
                       normalize=normalize, normalizer_type=normalizer_type)

        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)

    def load_data(self, data_path, train_ratio, valid_ratio, test_ratio, normalize, normalizer_type):
        process_path = data_path.split('.')[0] + '_processed.pt'
        if osp.exists(process_path):
            print('Loading processed data from ', process_path)
            train_data, valid_data, test_data = torch.load(process_path)
        else:
            print('Processing data...')
            with open(data_path, 'rb') as f:
                raw_data = pickle.load(f)
            
            f_list = raw_data['f']
            u = torch.from_numpy(raw_data['u']).to(torch.float32).unsqueeze(-1)
            pos = torch.from_numpy(raw_data['point']).to(torch.float32)
            cell = torch.from_numpy(raw_data['cell']).to(torch.long)
            
            f = self.compute_source(f_list, pos)
            data_size = f.shape[0]
            train_idx = int(data_size * train_ratio)
            valid_idx = int(data_size * (train_ratio + valid_ratio))
            test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
            
            train_data, normalizer = self.pre_process(f[:train_idx], u[:train_idx], pos, cell, mode='train', 
                                                      normalize=normalize, normalizer_type=normalizer_type)
            valid_data = self.pre_process(f[train_idx:valid_idx], u[train_idx:valid_idx], pos, cell, mode='valid',
                                          normalize=normalize, normalizer=normalizer)
            test_data = self.pre_process(f[valid_idx:test_idx], u[valid_idx:test_idx], pos, cell, mode='test',
                                         normalize=normalize, normalizer=normalizer)
            print('Saving data...')
            torch.save((train_data, valid_data, test_data), process_path)
            print('Data processed and saved to', process_path)

        self.train_dataset = PoissonBase(train_data, mode='train')
        self.valid_dataset = PoissonBase(valid_data, mode='valid')
        self.test_dataset = PoissonBase(test_data, mode='test')
        
    def compute_source(self, gaussian_list, pos):
        print('Computing source term...')
        f_list = []
        for gaussian in tqdm(gaussian_list):
            terms = torch.from_numpy(gaussian).to(torch.float32).unsqueeze(1)
            terms = terms.repeat(1, pos.shape[0], 1)
            x = pos[:, 0].unsqueeze(0).repeat(terms.shape[0], 1)
            y = pos[:, 1].unsqueeze(0).repeat(terms.shape[0], 1)
            f = torch.exp(-((x - terms[..., 0]) ** 2 + (y - terms[..., 1]) ** 2)/(2 * terms[..., 2] ** 2))
            f = f.sum(0)
            f_list.append(f)

        return torch.stack(f_list, 0).unsqueeze(-1)

    def pre_process(self, f, u, pos, cell,
                    mode='train', normalize=False, 
                    normalizer=None, normalizer_type='PGN', **kwargs):
        if normalize:
            if mode == 'train':
                if normalizer_type == 'PGN':
                    x_normalizer = UnitGaussianNormalizer(f)
                    u_normalizer = UnitGaussianNormalizer(u)
                else:
                    x_normalizer = GaussianNormalizer(f)
                    u_normalizer = GaussianNormalizer(u)
                x = x_normalizer.encode(f)
                u = u_normalizer.encode(u)
                normalizer = (x_normalizer, u_normalizer)
            else:
                x = normalizer[0].encode(f)
                u = normalizer[1].encode(u)
        else:
            normalizer = (None, None)
        
        all_data = [construct_coordinate(Data(x=x[i], y=u[i], pos=pos, cell=cell)) for i in tqdm(range(x.shape[0]))]

        if mode == 'train':
            return all_data, normalizer
        else:
            return all_data


class PoissonBase(Dataset):
    def __init__(self, data, mode='train', **kwargs):
        self.mode = mode
        self.all_data = data
        
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]
