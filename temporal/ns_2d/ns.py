import h5py
import torch
from torch.utils.data import DataLoader, Dataset

class AutoregressiveDataset(Dataset):
    """
    Memory-Efficient Dataset.
    Does NOT pre-calculate windows. Slices them on-the-fly from the main tensor.
    """
    def __init__(self, u_tensor):
        super().__init__()
        self.data = u_tensor # Shape: (N_samples, T_total, X, Y)
        
        self.n_samples = self.data.shape[0]
        self.t_total = self.data.shape[1]
        
        # Calculate how many valid windows exist per trajectory
        # Valid start indices are [0 ... T_total - 1]
        self.windows_per_sample = self.t_total - 1
        
        self.total_windows = self.n_samples * self.windows_per_sample

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # Map global index 'idx' to (sample_idx, time_idx)
        # sample_idx: Which trajectory?
        # time_idx: Which starting time step within that trajectory?
        sample_idx = idx // self.windows_per_sample
        time_idx = idx % self.windows_per_sample
        
        # Slice on-the-fly (Zero memory overhead)
        # Input: [t : t + t_in]
        x = self.data[sample_idx, time_idx : time_idx + 1]
        
        # Target: [t + t_in : t + t_in + t_out]
        y = self.data[sample_idx, time_idx + 1 : time_idx + 2]
        
        return x, y

class RandomizedAutoregressiveDataset(Dataset):
    """
    Randomized Sampling Dataset.
    1. Lazy: Slices data on-the-fly (No memory overhead).
    2. Randomized: Each time you ask for a sample, it picks a random time window.
       This makes 1 Epoch = 1 pass through the trajectories (N=1000), 
       rather than 1 pass through all time steps (N=500,000).
    """
    def __init__(self, u_tensor):
        super().__init__()
        self.data = u_tensor # Shape: (N_samples, T_total, X, Y)
        
        self.n_samples = self.data.shape[0]
        self.t_total = self.data.shape[1]
        
        # The latest possible start time 't' that still leaves room for t_in + t_out
        self.max_t = self.t_total - 1

    def __len__(self):
        # The epoch size is now the number of trajectories (e.g., 1000)
        return self.n_samples

    def __getitem__(self, idx):
        # idx is the Trajectory Index (0 to 999)
        
        # 1. Pick a RANDOM start time 't' for this trajectory
        # We use torch.randint to pick a valid start point
        t = torch.randint(0, self.max_t, (1,)).item()
        
        x = self.data[idx, t : t + 1]
        y = self.data[idx, t + 1 : t + 2]
        
        return x, y

class NavierStokes2D:
    def __init__(self, data_config, train_test_split=0.9):
        self.data_config = data_config
        self.train_test_split = train_test_split
        
        self.visc = data_config.get('viscosity', 1e-3)
        self.dt = data_config.get('dt', 1.0)

    def is_autoregressive(self):
        return True

    def load_data(self):
        data_path = self.data_config['data_path']
        print(f"Loading NS data from {data_path}...")
        
        with h5py.File(data_path, 'r') as f:
            # Load Raw Data
            # If this line alone OOMs (e.g. 32GB+ file), we need to use 'mmap'.
            # But for <16GB files, loading once is faster for training.
            #u_data = f['adversarial/u'][:]
            u_data = f['u'][:]
            
            # Attributes
            if 'viscosity' in f.attrs:
                self.visc = float(f.attrs['viscosity'])
                print(f"Loaded viscosity: {self.visc}")
            if 't' in f:
                t_arr = f['t'][:]
                if len(t_arr) > 1:
                    self.dt = float(t_arr[1] - t_arr[0])
                    print(f"Loaded dt: {self.dt}")

        u_tensor = torch.tensor(u_data, dtype=torch.float32)
        
        # Fix Dimensions: (N, X, Y, T) -> (N, T, X, Y)
        # This permutation is cheap (view operation)
        u_tensor = u_tensor.permute(0, 3, 1, 2)
        
        if self.data_config.get('n_samples') is not None:
            u_tensor = u_tensor[:self.data_config['n_samples']]

        print(f"Data Shape: {u_tensor.shape}")
        return u_tensor

    def load_training_data(self, batch_size, include_grid=True):
        # 1. Load the huge tensor once
        full_tensor = self.load_data()
        
        # 2. Split by Trajectory (Train/Test)
        # We split the *samples*, not the windows, to prevent data leakage.
        total_samples = full_tensor.shape[0]
        train_len = int(self.train_test_split * total_samples)
        
        train_tensor = full_tensor[:train_len]
        test_tensor = full_tensor[train_len:]
        
        print(f"Train Trajectories: {train_tensor.shape[0]} | Test Trajectories: {test_tensor.shape[0]}")
        
        # 3. Create Lazy Datasets
        # These objects are lightweight (store only reference + integers)
        train_dataset = RandomizedAutoregressiveDataset(train_tensor)
        test_dataset = RandomizedAutoregressiveDataset(test_tensor)
        
        # 4. Create Loaders
        # num_workers > 0 is crucial here to pre-fetch slices in background
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        return train_loader, test_loader

    def print_problem(self, model_config):
        print(f"--- NS Configuration ---")
        print(f"Viscosity: {self.visc}")
        print(f"Time Step (dt): {self.dt}")

    def has_pde_loss(self):
        return True

    def pde_loss(self, w_pred, w_target, x_history):
        # (Same implementation as before)
        nu = self.visc
        dt = self.dt
        
        w_prev = x_history[:, -1:, :, :] 
        w_t = (w_pred - w_prev) / dt

        u, v, w_x, w_y, lap_w = self.compute_spectral_physics(w_pred)

        advection = (u * w_x) + (v * w_y)
        diffusion = nu * lap_w

        residual = w_t + advection - diffusion
        return torch.mean(residual**2)

    def compute_spectral_physics(self, w):
        # (Same implementation as before)
        batchsize = w.shape[0]
        nx, ny = w.shape[2], w.shape[3]
        device = w.device

        w_h = torch.fft.fft2(w, dim=(-2, -1))
        
        k_x = 2 * torch.pi * torch.fft.fftfreq(nx, d=1/nx).to(device)
        k_y = 2 * torch.pi * torch.fft.fftfreq(ny, d=1/ny).to(device)
        kx, ky = torch.meshgrid(k_x, k_y, indexing='ij')
        
        kx = kx.reshape(1, 1, nx, ny)
        ky = ky.reshape(1, 1, nx, ny)
        
        k_sq = kx**2 + ky**2
        k_sq[..., 0, 0] = 1.0 # Inverse laplacian

        psi_h = w_h / k_sq
        psi_h[..., 0, 0] = 0.0

        u_h = 1j * ky * psi_h
        v_h = -1j * kx * psi_h
        u = torch.fft.ifft2(u_h, dim=(-2, -1)).real
        v = torch.fft.ifft2(v_h, dim=(-2, -1)).real

        w_x_h = 1j * kx * w_h
        w_y_h = 1j * ky * w_h
        lap_w_h = -k_sq * w_h

        w_x = torch.fft.ifft2(w_x_h, dim=(-2, -1)).real
        w_y = torch.fft.ifft2(w_y_h, dim=(-2, -1)).real
        lap_w = torch.fft.ifft2(lap_w_h, dim=(-2, -1)).real

        return u, v, w_x, w_y, lap_w
    
    def has_bc_loss(self):
        return False