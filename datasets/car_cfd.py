'''
    For more infomation about the original car-cfd data, please refer to:

    - Transolver: A Fast Transformer Solver for PDEs on General Geometries: https://github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar
    - Geometry-Guided Conditional Adaption for Surrogate Models of Large-Scale 3D PDEs on Arbitrary Geometries: https://openreview.net/forum?id=EyQO9RPhwN
    - Learning Three-Dimensional Flow for Interactive Aerodynamic Design: https://dl.acm.org/doi/pdf/10.1145/3197517.3201325

    Here, we follow the preprossing data process in `Transolver` and use their provided preprocessed data.

'''

import torch
import os

import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset

from torch_geometric.loader import DataLoader
from utils.graph import construct_coordinate
from tqdm import tqdm


class CarCFDDataset:
    def __init__(self, data_path, 
                 train_batchsize=256, eval_batchsize=128, valid_dir=1, test_dir=0,      # range(0, 9, 1)
                 normalize=True, **kwargs):
        self.load_data(data_path, valid_dir, test_dir, normalize)

        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)

    def load_data(self, data_path, valid_dir, test_dir, normalize, **kwargs):
        process_path = os.path.join(data_path, 'processed.pt')
        data_dir = os.path.join(data_path, 'preprocessed_data/')
        if os.path.exists(process_path):
            print('Loading processed data from ', process_path)
            train_data, valid_data, test_data = torch.load(process_path)
        else:
            print('Processing data...')

            train_data, valid_data, test_data, _ = self.pre_process(data_dir, valid_dir, test_dir, normalize)

            print('Saving data...')
            torch.save((train_data, valid_data, test_data), process_path)
            print('Data processed and saved to', process_path)
            
        self.train_dataset = CarCFDBase(train_data, mode='train')
        self.valid_dataset = CarCFDBase(valid_data, mode='valid')
        self.test_dataset = CarCFDBase(test_data, mode='test')

    def pre_process(self, data_dir, valid_dir, test_dir, norm=False, **kwargs):
        trainlst = []
        vallst = []
        testlst = []
        
        folds = [f'param{i}' for i in range(9)]
        
        print('loading data...')
        for i, fold in tqdm(enumerate(folds)):
            fold_samples = []
            files = os.listdir(os.path.join(data_dir, fold))        
            for f in files:
                path = os.path.join(data_dir, os.path.join(fold, f))
                if os.path.isdir(path):
                    fold_samples.append(os.path.join(fold, f))
            if i == valid_dir:
                vallst = fold_samples
            elif i == test_dir:
                testlst = fold_samples
            else:
                trainlst += fold_samples

        print('processing data...')
        train_dataset, coef_norm = self.single_process(trainlst, norm=norm, data_dir=data_dir)
        val_dataset = self.single_process(vallst, coef_norm=coef_norm, data_dir=data_dir)
        test_dataset = self.single_process(testlst, coef_norm=coef_norm, data_dir=data_dir)

        train_length = len(train_dataset)
        train_all_data = [construct_coordinate(train_dataset[i]) for i in tqdm(range(train_length))]

        val_length = len(val_dataset)
        val_all_data = [construct_coordinate(val_dataset[i]) for i in tqdm(range(val_length))]

        test_length = len(test_dataset)
        test_all_data = [construct_coordinate(test_dataset[i]) for i in tqdm(range(test_length))]

        return train_all_data, val_all_data, test_all_data, coef_norm

    def single_process(self, samples, norm=False, coef_norm=None, data_dir=None):
        dataset = []
        mean_in, mean_out = 0, 0
        std_in, std_out = 0, 0
        for k, s in tqdm(enumerate(samples)):
            save_path = os.path.join(data_dir, s)
            if not os.path.exists(save_path):
                continue

            init = np.load(os.path.join(save_path, 'x.npy'))[:, 3:]
            target = np.load(os.path.join(save_path, 'y.npy'))
            pos = np.load(os.path.join(save_path, 'pos.npy'))
            surf = np.load(os.path.join(save_path, 'surf.npy'))
            edge_index = np.load(os.path.join(save_path, 'edge_index.npy'))
            drag_coef_numpy = np.load(os.path.join(save_path, 'drag_coef.npy'))

            surf = torch.tensor(surf)
            pos = torch.tensor(pos)
            x = torch.tensor(init)
            y = torch.tensor(target)
            edge_index = torch.tensor(edge_index)
            drag_coef = torch.tensor(drag_coef_numpy)

            if norm and coef_norm is None:
                if k == 0:
                    old_length = init.shape[0]
                    mean_in = init.mean(axis=0)
                    mean_out = target.mean(axis=0)
                    mean_drag_co = drag_coef_numpy
                else:
                    new_length = old_length + init.shape[0]
                    mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                    mean_out += (target.sum(axis=0) - init.shape[0] * mean_out) / new_length
                    old_length = new_length
                    mean_drag_co += (drag_coef_numpy - mean_drag_co) / (k+1)

            data = Data(x=x, y=y, pos=pos, surf=surf.bool(), drag_coef = drag_coef)

            dataset.append(data)

        if norm and coef_norm is None:
            for k, data in tqdm(enumerate(dataset)):
                if k == 0:
                    old_length = data.x.numpy().shape[0]
                    std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                    std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
                    std_drag_co = ((data.drag_coef.numpy() - mean_drag_co) ** 2)
                else:
                    new_length = old_length + data.x.numpy().shape[0]
                    std_in += (((data.x.numpy() - mean_in) ** 2).sum(axis=0) - data.x.numpy().shape[
                        0] * std_in) / new_length
                    std_out += (((data.y.numpy() - mean_out) ** 2).sum(axis=0) - data.x.numpy().shape[
                        0] * std_out) / new_length
                    old_length = new_length
                    std_drag_co += (((data.drag_coef.numpy() - mean_drag_co) ** 2) - std_drag_co) / (k+1)

            std_in = np.sqrt(std_in)
            std_out = np.sqrt(std_out)
            std_drag_co = np.sqrt(std_drag_co)

            for data in dataset:
                data.x = ((data.x - mean_in) / (std_in + 1e-8)).float()
                data.y = ((data.y - mean_out) / (std_out + 1e-8)).float()
                data.drag_coef = ((data.drag_coef - mean_drag_co) / (std_drag_co + 1e-8)).float()

            coef_norm = (mean_in, std_in, mean_out, std_out, mean_drag_co, std_drag_co)
            dataset = (dataset, coef_norm)

        elif coef_norm is not None:
            for data in dataset:
                data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)).float()
                data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)).float()
                data.drag_coef = ((data.drag_coef - coef_norm[4]) / (coef_norm[5] + 1e-8)).float()

        return dataset


class CarCFDBase(Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
