import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class Darcy2DSimple:
    
    def __init__(self, data_config, train_test_split=0.9):
        self.data_config = data_config
        self.train_test_split = train_test_split

    def load_darcy_data(self):
        return self.load_darcy_data_from_path(self.data_config['data_path'])

    def load_darcy_data_from_path(self, data_path):
        """
        Loads 2D Darcy Flow data (Permeability 'a' -> Pressure 'u').
        Returns x_train (Permeability), y_train (Pressure), dx, and dy.
        """
        print(f"Loading Darcy data from {data_path}...")
        
        with h5py.File(data_path, 'r') as f:
            keys = list(f.keys())
            
            # 1. Load Input (Permeability Field 'a') - Assumed to be stored in 'nu'
            x_data_np = f['nu'][:] # Shape (N, Y, X)
            
            # 2. Load Target (Pressure Field 'u') - Assumed to be stored in 'tensor'
            y_data_np = f['tensor'][:] # Shape (N, 1, Y, X)
        
        Y_RES, X_RES = x_data_np.shape[1:]

        # Convert to PyTorch tensors
        x_data = torch.tensor(x_data_np, dtype=torch.float32)
        y_data = torch.tensor(y_data_np, dtype=torch.float32)

        if self.data_config['n_samples'] is not None:
            x_data = x_data[:self.data_config['n_samples']]
            y_data = y_data[:self.data_config['n_samples']]
            
        N, Y, X = x_data.shape
        print(f"Data Shape (N, Y, X): {N}, {Y}, {X}")
        
        # --- Prepare Input (x_train) ---
        # Current shape is (N, Y, X). FNO/PyTorch expects (N, C, Y, X).
        x_train = x_data.unsqueeze(1) # Final shape: (N, 1, Y, X)

        # --- Prepare Target (y_train) ---
        y_train = y_data # Shape (N, 1, Y, X). Already correct.

        # --- Grid params for Physics Loss ---
        self.dx = 1.0 / X_RES
        self.dy = 1.0 / Y_RES
        
        return x_train, y_train
        
    def load_training_data(self, batch_size, include_grid=True):
        x_data, y_data = self.load_darcy_data()
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

    def get_spectral_derivatives_2d(self, u):
        """
        Computes derivatives of u for 2D Darcy Flow using 2D FFT.
        u shape: (Batch, C, Y, X)
        """
        B, C, Y, X = u.shape
        
        dx, dy = self.dx, self.dy

        # 1. FFT to Frequency Domain (Spatial dimensions are -2 and -1)
        # Using 2D FFT
        u_hat = torch.fft.fft2(u, dim=(-2, -1))
        
        # 2. Create Wavenumbers (kx and ky)
        kx = 2 * torch.pi * torch.fft.fftfreq(X, d=dx).to(u.device)
        ky = 2 * torch.pi * torch.fft.fftfreq(Y, d=dy).to(u.device)
        
        # Meshgrid kx and ky for 2D convolution in freq domain
        # Shape: (Y, X)
        Kx, Ky = torch.meshgrid(ky, kx, indexing='ij')

        # Reshape for broadcasting: (1, 1, Y, X)
        Kx = Kx.reshape(1, 1, Y, X)
        Ky = Ky.reshape(1, 1, Y, X)
        
        # 3. Compute Derivatives in Frequency Domain
        # i * kx * u_hat
        u_x_hat = (1j * Kx) * u_hat
        # i * ky * u_hat
        u_y_hat = (1j * Ky) * u_hat
        
        # 4. IFFT back to Physical Domain
        u_x = torch.fft.ifft2(u_x_hat, dim=(-2, -1)).real
        u_y = torch.fft.ifft2(u_y_hat, dim=(-2, -1)).real
        
        return u_x, u_y

    def compute_darcy_pino_loss(self, u_pred, a_input, f_source=1.0):
        """
        Calculates the residual for the Darcy equation: -div(a grad u) = f.
        u_pred: Pressure field u (Batch, 1, Y, X)
        a_input: Permeability field a (Batch, 1, Y, X)
        f_source: Source term (scalar or array)
        """
        # 1. Compute Gradients of u: grad u = (u_x, u_y)
        u_x, u_y = self.get_spectral_derivatives_2d(u_pred)
        
        # 2. Compute Flux q: q = a * grad u = (a * u_x, a * u_y)
        # Since a_input is the permeability, we use it directly
        q_x = a_input * u_x
        q_y = a_input * u_y

        # 3. Compute Divergence of Flux: div(q) = div(a * grad u)
        # Div(q) = q_x_x + q_y_y (using 2D Spectral Derivatives)
        q_x_x, _ = self.get_spectral_derivatives_2d(q_x)
        _, q_y_y = self.get_spectral_derivatives_2d(q_y)
        
        div_a_grad_u = q_x_x + q_y_y
        
        # 4. Compute Residual: R = -div(a grad u) - f
        # Equation: -div(a grad u) = f
        # Residual: R = -div(a grad u) - f
        
        # We assume a constant source term f=1.0 for simplicity, 
        # but the source term f should be passed in a real implementation.
        res = -div_a_grad_u - f_source
        
        return torch.mean(res**2)


    def has_pde_loss(self):
        return True

    def pde_loss(self, out, y, x):
        return self.compute_darcy_pino_loss(out, x)
    
    def has_bc_loss(self):
        return True
    
    def bc_loss(self, out, y, x):
        bc_loss_y = torch.mean((out[:, :, 0, :] - y[:, :, 0, :])**2) + \
                    torch.mean((out[:, :, -1, :] - y[:, :, -1, :])**2)
        
        bc_loss_x = torch.mean((out[:, :, :, 0] - y[:, :, :, 0])**2) + \
                    torch.mean((out[:, :, :, -1] - y[:, :, :, -1])**2)
        
        return bc_loss_x + bc_loss_y
