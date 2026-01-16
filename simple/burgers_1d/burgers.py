
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class Burgers1DSimple:
    
    def __init__(self, data_config, train_test_split=0.9):
        self.data_config = data_config
        self.train_test_split = train_test_split
        self.nu = data_config['nu']
        if data_config['nu_divide_by_pi']:
            self.nu = self.nu / np.pi

    def load_burgers_data(self, include_grid=True):
        return self.load_burgers_data_for_path(data_path=self.data_config['data_path'], include_grid=include_grid)

    def load_burgers_data_for_path(self, data_path, include_grid=True):
        print(f"Loading data from {data_path}...")
        
        with h5py.File(data_path, 'r') as f:
            keys = list(f.keys())
            key = 'train' if 'train' in keys else "tensor"
            
            if key == 'train':
                data = f['train']['u'][:] 
            else:
                data = f[key][:]
                
            data = torch.tensor(data, dtype=torch.float32)
            x_grid_source = torch.tensor(f['x-coordinate'][:], dtype=torch.float32)
            if self.data_config['n_samples'] is not None:
                data = data[:self.data_config['n_samples']]
                
            N, T, X = data.shape
            print(f"Original Data Shape (N, T, X): {N}, {T}, {X}")
                        
            u0 = data[:, 0, :].unsqueeze(-1)
            if include_grid:
                grid = x_grid_source.view(1, X, 1).repeat(N, 1, 1)
                x_train = torch.cat([u0, grid], dim=-1)
            else:
                x_train = u0

            u_final = data[:, -1, :]
            y_train = u_final
            
            print(f"Input Shape (N, C, X): {x_train.shape}")
            print(f"Target Shape (N, C, X): {y_train.shape}")

            self.dx = self.data_config['x_domain_size'] / X

            return x_train, y_train
        
    def load_training_data(self, batch_size, include_grid=True):
        x_data, y_data = self.load_burgers_data(include_grid=include_grid)
        train_size = int(self.train_test_split * len(x_data))
        train_loader = DataLoader(TensorDataset(x_data[:train_size], y_data[:train_size]), 
                                batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_data[train_size:], y_data[train_size:]), 
                                batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def print_problem(self, model_config):    
        print(f"--- Configuration Loaded ---")
        print(f"Model Layers: {model_config['n_layers']} (from TOML)")
        print(f"Epochs: {model_config['epochs']} (from CLI)")
        if model_config['nu_divide_by_pi']:
            print(f"Viscosity (nu) divided by PI: {model_config['nu'] / np.pi:.6f}") 
        else:
            print(f"Viscosity (nu): {model_config['nu']:.6f}") 

    def has_pde_loss(self):
        return False

    def pde_loss(self, out, y, x):
        return None
    
    def has_bc_loss(self):
        return False
    
    def bc_loss(self, out, y, x):
        return None