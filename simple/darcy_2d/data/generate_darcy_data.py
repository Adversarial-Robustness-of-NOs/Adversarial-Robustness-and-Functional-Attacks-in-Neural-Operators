import argparse
import sys
import tomllib
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# IMPORT SHARED SOLVER
import simple.darcy_2d.solver as darcy_solver

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class DarcyDataGenerator:
    def __init__(self, x_resolution=128, y_resolution=128, generation_resolution=None, force_term=1.0):
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        # Generate at the highest resolution to keep features consistent, then downsample
        self.gen_res = generation_resolution if generation_resolution is not None else x_resolution
        self.force = force_term
        
        # Grid Setup (Standard [0,1] box) for TARGET resolution
        self.x = np.linspace(0, 1, x_resolution)
        self.y = np.linspace(0, 1, y_resolution)
        # Note: darcy_solver uses 'ij' indexing internally for physics
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    # --- 1. CORE: Gaussian Blobs (PDEBench Style Clean Data) ---
    def generate_base_permeability(self, seed=None):
        """
        Uses the shared darcy_solver to generate the clean 'Islands' map.
        Values are strictly 0.1 or 1.0.
        """
        # 1. Generate at High Resolution (Master Copy)
        nu_high = darcy_solver.generate_permeability(self.gen_res, self.gen_res, device=DEVICE, seed=seed)
        
        # 2. Downsample if needed
        if self.gen_res != self.x_resolution:
            # Reshape for interpolation: (Batch, Channel, H, W)
            nu_input = nu_high.unsqueeze(0).unsqueeze(0)
            
            # Use 'nearest' to preserve the strict 0.1/1.0 binary nature of the islands
            nu_down = F.interpolate(nu_input, size=(self.x_resolution, self.y_resolution), mode='nearest')
            
            return nu_down.squeeze(0).squeeze(0)
        
        return nu_high

    # --- 2. NOISE TYPE: Spectral (High Freq Jitter) ---
    def apply_spectral_noise(self, nu_clean, noise_level=0.01):
        """
        Adds random Gaussian noise to the permeability map.
        Since PDEBench uses strict 0.1/1.0, this makes the 'islands' noisy.
        """
        nu_cpu = nu_clean.cpu().numpy()
        # Fixed: self.res -> self.x_resolution
        noise = np.random.randn(self.x_resolution, self.y_resolution) * noise_level
        nu_noisy = nu_cpu + noise
        
        # Enforce physical constraints (permeability must be positive)
        # We clip at 0.01 to ensure the solver doesn't crash (div by zero)
        nu_noisy = np.maximum(nu_noisy, 0.01)
        
        return torch.tensor(nu_noisy, device=DEVICE, dtype=torch.float32)

    # --- 3. NOISE TYPE: Spatial (Geometric Inclusions) ---
    def apply_spatial_noise(self, nu_clean, n_inclusions=3):
        """
        Adds random geometric shapes (circles/rectangles) that override the permeability.
        This simulates 'artifacts' or foreign objects in the material.
        """
        nu_mod = nu_clean.cpu().numpy().copy()
        
        for _ in range(n_inclusions):
            cx, cy = random.random(), random.random()
            shape_type = random.choice(['circle', 'rect'])
            
            # Randomly choose one of the valid physics values to inject
            val = random.choice([0.1, 1.0]) 
            
            if shape_type == 'circle':
                r = random.uniform(0.005, 0.015)
                # X and Y are meshgrids
                mask = (self.X - cx)**2 + (self.Y - cy)**2 < r**2
                nu_mod[mask] = val
            else:
                w, h = random.uniform(0.005, 0.02), random.uniform(0.005, 0.02)
                mask = (self.X > cx - w/2) & (self.X < cx + w/2) & \
                       (self.Y > cy - h/2) & (self.Y < cy + h/2)
                nu_mod[mask] = val
                
        return torch.tensor(nu_mod, device=DEVICE, dtype=torch.float32)

    # --- 4. DATASET CREATION ---
    def create_dataset(self, n_samples, mode='clean', filename='data.h5'):
        print(f"Generating {mode} dataset to {filename}...")
        
        input_a_list = []
        target_u_list = []
        
        for i in tqdm(range(n_samples)):
            
            # 1. Generate Base (Clean) with EXPLICIT SEED
            # This ensures Sample #5 is geometrically identical across all resolutions
            nu = self.generate_base_permeability(seed=i)
            
            # 2. Apply Mode-Specific Modifications
            if mode == 'clean':
                pass # nu is already good
            elif mode == 'spectral': 
                np.random.seed(i) # Ensure noise is deterministic
                nu = self.apply_spectral_noise(nu)
            elif mode == 'spatial': 
                random.seed(i) # Ensure inclusions are deterministic
                nu = self.apply_spatial_noise(nu)
            
            # 3. Solve Physics (Ground Truth) using Shared Solver
            # Note: Solver handles padding/BCs internally
            u = darcy_solver.solve_steady_state(nu, force=self.force)
            
            # 4. Store (Move to CPU/Numpy)
            input_a_list.append(nu.cpu().numpy())
            target_u_list.append(u.cpu().numpy())
            
        # Stack and Format
        # Input 'a' shape: (N, H, W)
        # Target 'u' shape: (N, 1, H, W)
        input_a = np.stack(input_a_list)
        target_u = np.stack(target_u_list)[:, None, :, :] 

        # Save H5
        with h5py.File(filename, 'w') as f:
            f.create_dataset('nu', data=input_a)
            f.create_dataset('tensor', data=target_u)
            f.create_dataset('x-coordinate', data=self.x)
            f.create_dataset('y-coordinate', data=self.y)
        
        print(f"Saved: {filename}")
        return input_a[0], target_u[0,0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="simple/darcy_2d/data")
    parser.add_argument("--resolutions", type=int, nargs='+', default=[64, 128, 256], help="List of resolutions")
    parser.add_argument("--force_term", type=float, default=1.0)
    parser.add_argument("--prefix", type=str, default="pdebench_darcy_")
    args = parser.parse_args()
    
    # Calculate the master resolution (highest requested)
    max_res = max(args.resolutions)
    print(f"Master Generation Resolution: {max_res} (All other resolutions will be downsampled from this)")
    
    for r in args.resolutions:
        # Pass generation_resolution=max_res to enforce consistency
        gen = DarcyDataGenerator(x_resolution=r, y_resolution=r, generation_resolution=max_res, force_term=args.force_term)
        
        filename = f"{args.output_dir}/{args.prefix}{r}_force_term_{args.force_term}.h5"
        a_spat, u_spat = gen.create_dataset(args.n_samples, 'clean', filename)
                
        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.imshow(a_spat, cmap='gray', origin='lower')
        plt.title("Permeability")
        plt.colorbar()
        
        plt.subplot(1,2,2)
        plt.imshow(u_spat, cmap='inferno', origin='lower')
        plt.title("Pressure Field (u)")
        plt.colorbar()
        
        preview_path = f"{args.output_dir}/{args.prefix}{r}_force_term_{args.force_term}_preview.png"
        plt.tight_layout()
        plt.savefig(preview_path)
        print(f"Preview saved as {preview_path}")