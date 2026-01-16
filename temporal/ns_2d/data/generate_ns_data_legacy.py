import os
import torch
import torch.nn.functional as F
import math
import scipy.io
import h5py
from timeit import default_timer
from tqdm import tqdm

# --- Monkey Patch for Modern PyTorch Compatibility ---
if not hasattr(torch, 'rfft'):
    def rfft_compat(input, signal_ndim, normalized=False, onesided=True):
        norm = "ortho" if normalized else None
        if signal_ndim == 2:
            if onesided:
                res = torch.fft.rfft2(input, dim=(-2, -1), norm=norm)
            else:
                res = torch.fft.fft2(input, dim=(-2, -1), norm=norm)
            return torch.view_as_real(res)

    def irfft_compat(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None):
        input = torch.view_as_complex(input)
        norm = "ortho" if normalized else None
        if signal_ndim == 2:
            if onesided:
                return torch.fft.irfft2(input, s=signal_sizes, dim=(-2, -1), norm=norm)
            else:
                res = torch.fft.ifft2(input, s=signal_sizes, dim=(-2, -1), norm=norm)
                return res.real

    def ifft_compat(input, signal_ndim, normalized=False):
        if input.shape[-1] == 2:
            input = torch.view_as_complex(input)
        norm = "ortho" if normalized else None
        res = torch.fft.ifftn(input, dim=tuple(range(-signal_ndim, 0)), norm=norm)
        return torch.view_as_real(res)

    torch.rfft = rfft_compat
    torch.irfft = irfft_compat
    torch.ifft = ifft_compat

class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        self.dim = dim
        self.device = device
        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))
        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)
            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)
            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers
            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        self.size = tuple([size]*dim)

    def sample(self, N):
        coeff = torch.randn(N, *self.size, 2, device=self.device)
        coeff[...,0] = self.sqrt_eig*coeff[...,0]
        coeff[...,1] = self.sqrt_eig*coeff[...,1]
        u = torch.ifft(coeff, self.dim, normalized=False)
        u = u[...,0]
        return u

def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    N = w0.size()[-1]
    k_max = math.floor(N/2.0)
    steps = math.ceil(T/delta_t)
    
    # Grid Setup
    w_h = torch.rfft(w0, 2, normalized=False, onesided=False)
    f_h = torch.rfft(f, 2, normalized=False, onesided=False)
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    record_time = math.floor(steps/record_steps)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    k_x = k_y.transpose(0,1)
    
    # Laplacian & Dealiasing
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    c = 0
    t = 0.0
    for j in range(steps):
        # Stream function
        psi_h = w_h.clone()
        psi_h[...,0] = psi_h[...,0]/lap
        psi_h[...,1] = psi_h[...,1]/lap

        # Velocity: q = psi_y, v = -psi_x
        q = psi_h.clone()
        temp = q[...,0].clone()
        q[...,0] = -2*math.pi*k_y*q[...,1]
        q[...,1] = 2*math.pi*k_y*temp
        q = torch.irfft(q, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        v = psi_h.clone()
        temp = v[...,0].clone()
        v[...,0] = 2*math.pi*k_x*v[...,1]
        v[...,1] = -2*math.pi*k_x*temp
        v = torch.irfft(v, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        # Vorticity Gradients
        w_x = w_h.clone()
        temp = w_x[...,0].clone()
        w_x[...,0] = -2*math.pi*k_x*w_x[...,1]
        w_x[...,1] = 2*math.pi*k_x*temp
        w_x = torch.irfft(w_x, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        w_y = w_h.clone()
        temp = w_y[...,0].clone()
        w_y[...,0] = -2*math.pi*k_y*w_y[...,1]
        w_y[...,1] = 2*math.pi*k_y*temp
        w_y = torch.irfft(w_y, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        # Nonlinear Term
        F_h = torch.rfft(q*w_x + v*w_y, 2, normalized=False, onesided=False)
        F_h[...,0] = dealias* F_h[...,0]
        F_h[...,1] = dealias* F_h[...,1]

        # Time Step (Cranck-Nicholson)
        denom = 1.0 + 0.5*delta_t*visc*lap
        w_h[...,0] = (-delta_t*F_h[...,0] + delta_t*f_h[...,0] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,0]) / denom
        w_h[...,1] = (-delta_t*F_h[...,1] + delta_t*f_h[...,1] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,1]) / denom

        t += delta_t
        if (j+1) % record_time == 0:
            sol[...,c] = torch.irfft(w_h, 2, normalized=False, onesided=False, signal_sizes=(N,N))
            sol_t[c] = t
            c += 1
    return sol, sol_t

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Navier-Stokes 2D training data")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="./data_test", help="Output directory")
    parser.add_argument("--resolutions", type=int, nargs='+', default=[32, 64, 128], help="List of resolutions")
    parser.add_argument("--viscosity", type=float, default=1e-3, help="Viscosity coefficient")
    parser.add_argument("--t_final", type=float, default=10.0, help="Final simulation time")
    parser.add_argument("--record_steps", type=int, default=50, help="Number of time steps to record")
    parser.add_argument("--prefix", type=str, default="ns_data_", help="Filename prefix")
    return parser.parse_args()

if __name__ == '__main__':
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # RESOLUTIONS TO GENERATE
    # It will generate ICs at max(resolutions) and downsample for others
    resolutions = args.resolutions
    
    N = args.n_samples
    bsize = args.batch_size
    OUTPUT_DIR = args.output_dir
    
    # Physics
    T_final = args.t_final
    record_steps = args.record_steps
    visc = args.viscosity
    
    # ==========================================
    # 2. GENERATE MASTER INITIAL CONDITIONS
    # ==========================================
    max_res = max(resolutions)
    print(f"1. Generating Master Initial Conditions (GRF) at resolution {max_res}x{max_res}...")
    
    GRF = GaussianRF(2, max_res, alpha=2.5, tau=7, device=device)
    
    # We generate all ICs upfront to ensure consistency. 
    # If N is huge, you might need to chunk this, but for N=1000 it's fine in RAM.
    # Generating in chunks to respect batch size for memory safety
    w0_master_list = []
    n_batches = math.ceil(N / bsize)
    
    for j in range(n_batches):
        curr_bs = min(bsize, N - j*bsize)
        w0_batch = GRF.sample(curr_bs) # Shape (B, max_res, max_res)
        w0_master_list.append(w0_batch.cpu())
    
    w0_master = torch.cat(w0_master_list, dim=0) # (N, max_res, max_res)
    print(f"   -> Master ICs generated. Shape: {w0_master.shape}")

    # ==========================================
    # 3. SOLVE FOR EACH RESOLUTION
    # ==========================================
    for s in resolutions:
        print(f"\n--- Processing Resolution: {s}x{s} ---")
        
        # A. Setup Output Folder
        data_dir = OUTPUT_DIR
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f'{args.prefix}{s}.h5')
        
        # B. Prepare Forcing at Current Resolution
        t_grid = torch.linspace(0, 1, s + 1, device=device)[:-1]
        X, Y = torch.meshgrid(t_grid, t_grid, indexing='ij')
        f_forcing = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

        # C. Prepare Storage
        input_data = torch.zeros(N, s, s)
        output_data = torch.zeros(N, s, s, record_steps)
        time_data = None
        
        t0 = default_timer()
        
        # D. Loop over Batches
        curr_idx = 0
        for j in tqdm(range(n_batches)):
            curr_bs = min(bsize, N - j*bsize)
            
            # 1. Get Master ICs and Resize
            # Input needs to be (Batch, Channels, H, W) for interpolate
            batch_w0_high = w0_master[curr_idx : curr_idx + curr_bs].to(device)
            batch_w0_high = batch_w0_high.unsqueeze(1) 
            
            # Mode='bilinear' or 'bicubic' is good for smooth fields.
            batch_w0_res = F.interpolate(batch_w0_high, size=(s, s), mode='bicubic', align_corners=False)
            batch_w0_res = batch_w0_res.squeeze(1) # Back to (B, s, s)

            # 2. Solve Navier Stokes
            sol, sol_t = navier_stokes_2d(batch_w0_res, f_forcing, visc, T_final, 1e-3, record_steps)
            
            # 3. Store
            input_data[curr_idx : curr_idx + curr_bs] = batch_w0_res.cpu()
            output_data[curr_idx : curr_idx + curr_bs] = sol.cpu()
            
            if curr_idx == 0:
                time_data = sol_t.cpu()
            
            curr_idx += curr_bs

        t1 = default_timer()
        print(f"   -> Solved in {t1 - t0:.2f}s")
        
        # E. Save to H5
        print(f"   -> Saving to {file_path}...")
        with h5py.File(file_path, 'w') as f_h5:
            f_h5.create_dataset('a', data=input_data.numpy())
            f_h5.create_dataset('u', data=output_data.numpy())
            f_h5.create_dataset('t', data=time_data.numpy())
            f_h5.attrs['viscosity'] = visc
            f_h5.attrs['resolution'] = s
            f_h5.attrs['n_samples'] = N
            
    print("\nAll resolutions completed!")