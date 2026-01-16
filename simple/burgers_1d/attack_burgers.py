import argparse
from pathlib import Path
import sys
import tomllib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
from scipy.integrate import odeint
from neuralop.models import FNO
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = 'best_fno_model_pdebench.pth' 
OUTPUT_DIR = 'data/robustness_suite_k32'

# Generation Settings
N_SAMPLES = 200        
NX = 1024
NT = 201
X_DOMAIN_SIZE = 1.0
T_DOMAIN_SIZE = 2.0

# Physics
NU = 0.01 / np.pi

# PGD Parameters
PGD_EPSILON = 0.05     
PGD_STEPS = 200         
PGD_ALPHA = 0.01       
SMOOTH_GRAD = True
KERNEL_SIZE = 15       # Size of smoothing window
SIGMA = 3.0            # Blur strength
LAMBDA_PDE = 1e-2      # Physics residual weight
LAMBDA_BC = 5e-2       # Boundary condition weight
AMPLITUDE_CLAMP = 2.0  # limits maximum distortion in amplitude attack
PHASE_CLAMP = 0.1      # limits maximum distortion in phase attack

# Batch Size for Attack
ATTACK_BATCH_SIZE = 10 

# Model Params (Must match train.py)
MODES = 32
WIDTH = 32
IN_CHANNELS = 1
OUT_CHANNELS = 1
NUM_LAYERS = 8

def init_globals(args, toml_config):
    """
    Merges Command Line Args (Attack/Gen) and TOML Config (Model Arch)
    into Global Scope, then calculates Physics Grids.
    """
    # 1. Update TOML Globals (Model Architecture)
    global NU, X_DOMAIN_SIZE, T_DOMAIN_SIZE, IN_CHANNELS, OUT_CHANNELS
    global MODES, WIDTH, NUM_LAYERS

    model_conf = toml_config['config']
    
    # Handle Pi logic
    if model_conf.get('nu_divide_by_pi', False):
        NU = model_conf['nu'] / np.pi
    else:
        NU = model_conf['nu']

    X_DOMAIN_SIZE = model_conf['x_domain_size']
    T_DOMAIN_SIZE = model_conf['t_domain_size']
    IN_CHANNELS   = model_conf['in_channels']
    OUT_CHANNELS  = model_conf['out_channels']
    MODES         = tuple(model_conf['num_modes'])
    WIDTH         = model_conf['width']
    NUM_LAYERS    = model_conf['num_layers']

    # 2. Update Argparse Globals (Attack Settings)
    global MODEL_PATH, OUTPUT_DIR, N_SAMPLES, NX, NT, ATTACK_BATCH_SIZE
    global PGD_EPSILON, PGD_STEPS, PGD_ALPHA, SMOOTH_GRAD, KERNEL_SIZE, SIGMA
    global LAMBDA_PDE, LAMBDA_BC
    global AMPLITUDE_CLAMP, PHASE_CLAMP

    MODEL_PATH = args.model_path
    OUTPUT_DIR = args.output_dir
    N_SAMPLES = args.n_samples
    NX = args.nx
    NT = args.nt
    ATTACK_BATCH_SIZE = args.attack_batch_size
    
    # PGD
    PGD_EPSILON = args.pgd_epsilon
    PGD_STEPS = args.pgd_steps
    PGD_ALPHA = args.pgd_alpha
    SMOOTH_GRAD = args.smooth_grad # Boolean
    KERNEL_SIZE = args.kernel_size
    SIGMA = args.sigma
    LAMBDA_PDE = args.lambda_pde
    LAMBDA_BC = args.lambda_bc
    AMPLITUDE_CLAMP = args.amplitude_clamp
    PHASE_CLAMP = args.phase_clamp

    # 3. Calculate Physics Grids (Dependent on NX/NT/X_DOMAIN_SIZE)
    global DX, K_WAVENUMBERS, T_EVAL, X_GRID
    
    DX = X_DOMAIN_SIZE / NX
    K_WAVENUMBERS = 2 * np.pi * np.fft.rfftfreq(NX, d=DX)
    T_EVAL = np.linspace(0, T_DOMAIN_SIZE, NT)
    X_GRID = np.linspace(0, X_DOMAIN_SIZE, NX, endpoint=False)

    print(f"--- Attack Configuration Loaded ---")
    print(f"Model: {MODEL_PATH}")
    print(f"PGD Epsilon: {PGD_EPSILON}, Steps: {PGD_STEPS}")
    print(f"Smoothing Enabled: {SMOOTH_GRAD}")
    print(f"Resolution: NX={NX}, NT={NT}")

def burgers_rhs(u, t, k, nu):
    u_hat = np.fft.rfft(u)
    u_x = np.fft.irfft(1j * k * u_hat)
    u_xx = np.fft.irfft(-(k**2) * u_hat)
    return -u * u_x + nu * u_xx

def solve_batch_ground_truth(u0_batch):
    solutions = []
    print(f"Solving Physics for {len(u0_batch)} adversarial samples...")
    if isinstance(u0_batch, torch.Tensor):
        u0_batch = u0_batch.cpu().numpy()
        
    for u0 in tqdm(u0_batch):
        sol = odeint(burgers_rhs, u0, T_EVAL, args=(K_WAVENUMBERS, NU))
        solutions.append(sol)
    return np.stack(solutions, axis=0).astype(np.float32)

# --- 2. GENERATOR ---
def generate_clean_ics(n):
    rng = np.random.default_rng(42)
    ics = []
    for _ in range(n):
        u = np.zeros(NX)
        k_tot = 4
        selected_indices = rng.integers(0, k_tot, size=2)
        active_modes = np.zeros(k_tot)
        for idx in selected_indices: active_modes[idx] += 1
        
        L_domain = 1.0
        for k_idx, count in enumerate(active_modes):
            if count == 0: continue
            k = k_idx + 1
            amp = rng.uniform(0, 1)
            phs = 2 * np.pi * rng.uniform(0, 1)
            u += count * amp * np.sin(2 * np.pi * k * X_GRID / L_domain + phs)
        
        if np.max(np.abs(u)) > 1e-9: u = u / np.max(np.abs(u))
        ics.append(u)
    return torch.tensor(np.stack(ics), dtype=torch.float32).to(DEVICE)

# --- 3. ATTACK LOGIC ---

def compute_pde_residual_loss(u_pred, nu, dx, dt):
    u_t = (u_pred[..., 2:, :] - u_pred[..., :-2, :]) / (2 * dt)
    u_x = (u_pred[..., :, 2:] - u_pred[..., :, :-2]) / (2 * dx)
    u_xx = (u_pred[..., :, 2:] - 2*u_pred[..., :, 1:-1] + u_pred[..., :, :-2]) / (dx**2)
    
    u_core = u_pred[..., 1:-1, 1:-1]
    u_t = u_t[..., :, 1:-1]
    u_x = u_x[..., 1:-1, :]
    u_xx = u_xx[..., 1:-1, :]
    
    res = u_t + (u_core * u_x) - (nu * u_xx)
    return torch.mean(res**2)

def compute_bc_loss(u_pred):
    """
    Simple periodic boundary penalty: enforce u(x=0) ~= u(x=1).
    """
    left = u_pred[..., 0]
    right = u_pred[..., -1]
    return torch.mean((left - right) ** 2)

# --- 1. UTILS ---
def get_smoothing_kernel(kernel_size=15, sigma=3.0, device='cpu'):
    """Creates a 1D Gaussian kernel for smoothing gradients."""
    x_coord = torch.arange(kernel_size)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(np.sqrt(2.*np.pi*variance))) * \
                      torch.exp(-(x_coord - mean)**2. / (2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel.view(1, 1, kernel_size).to(device)

def spatial_pgd(model, u0_full_data):
    """Batched Spatial PGD Attack."""
    model.eval()
    adversarial_batches = []
        
    total_samples = len(u0_full_data)
    print(f"Running Spatial PGD on {total_samples} samples (Batch Size: {ATTACK_BATCH_SIZE})...")

    if SMOOTH_GRAD:
        kernel = get_smoothing_kernel(KERNEL_SIZE, SIGMA, DEVICE)
        pad = KERNEL_SIZE // 2

    # Iterate in Mini-Batches
    for i in range(0, total_samples, ATTACK_BATCH_SIZE):
        u0_batch = u0_full_data[i : i + ATTACK_BATCH_SIZE]
        
        with torch.no_grad():
            input_clean = u0_batch.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
            clean_pred = model(input_clean).detach()

        # Init Delta for this batch
        delta = torch.zeros_like(u0_batch).uniform_(-PGD_EPSILON, PGD_EPSILON)
        delta.requires_grad = True
                
        # PGD Loop
        for step in range(PGD_STEPS):
            # Use delta directly in computation graph
            perturbed_u0 = torch.clamp(u0_batch + delta, -2.0, 2.0)
            
            # Expand to (N, 1, T, X) - No Grid Injection
            input_u0 = perturbed_u0.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
            
            pred = model(input_u0)
            loss = -1.0 * torch.nn.MSELoss()(pred, clean_pred)
            
            # Zero gradients if they exist
            if delta.grad is not None:
                delta.grad.zero_()
                
            loss.backward()

            with torch.no_grad():
                grad = delta.grad
                
                # --- SMOOTHING STEP ---
                if SMOOTH_GRAD:
                    # Apply 1D Convolution to blur the gradient
                    # Reshape to (N, 1, W) for conv1d
                    grad_input = grad.unsqueeze(1) 
                    grad_smooth = F.conv1d(grad_input, kernel, padding=pad)
                    grad = grad_smooth.squeeze(1)
                
                # Update
                delta.data -= PGD_ALPHA * grad.sign()
                delta.data = torch.clamp(delta.data, -PGD_EPSILON, PGD_EPSILON)
                delta.grad.zero_()
        
        # Store result
        adversarial_batches.append((u0_batch + delta).detach())
        
        if (i // ATTACK_BATCH_SIZE) % 5 == 0:
            print(f"   Processed batch {i} to {i+len(u0_batch)}...")

    return torch.cat(adversarial_batches, dim=0)

def spectral_pgd(model, u0_full_data):
    """Batched Spectral PGD Attack."""
    model.eval()
    adversarial_batches = []

    total_samples = len(u0_full_data)
    print(f"Running Spectral PGD on {total_samples} samples (Batch Size: {ATTACK_BATCH_SIZE})...")

    for i in range(0, total_samples, ATTACK_BATCH_SIZE):
        u0_batch = u0_full_data[i : i + ATTACK_BATCH_SIZE]
        with torch.no_grad():
            input_clean = u0_batch.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
            clean_pred = model(input_clean).detach()

        u_hat = torch.fft.rfft(u0_batch, dim=1)
        
        delta_re = torch.zeros_like(u_hat.real).uniform_(-0.1, 0.1).requires_grad_(True)
        delta_im = torch.zeros_like(u_hat.imag).uniform_(-0.1, 0.1).requires_grad_(True)
        
        for step in range(PGD_STEPS):
            delta_c = torch.complex(delta_re, delta_im)
            u_adv = torch.fft.irfft(u_hat + delta_c, n=NX, dim=1)
            
            input_u0 = u_adv.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
            
            pred = model(input_u0)
            loss = -1.0 * torch.nn.MSELoss()(pred, clean_pred)
            
            if delta_re.grad is not None:
                delta_re.grad.zero_()
                delta_im.grad.zero_()
                
            loss.backward()
            
            with torch.no_grad():
                delta_re.data += PGD_ALPHA * delta_re.grad.sign()
                delta_im.data += PGD_ALPHA * delta_im.grad.sign()
                
                delta_re.data = torch.clamp(delta_re.data, -0.1, 0.1)
                delta_im.data = torch.clamp(delta_im.data, -0.1, 0.1)
                
                delta_re.grad.zero_()
                delta_im.grad.zero_()
                
        final_delta = torch.complex(delta_re, delta_im)
        res_batch = torch.fft.irfft(u_hat + final_delta, n=NX, dim=1).detach()
        adversarial_batches.append(res_batch)
        
        if (i // ATTACK_BATCH_SIZE) % 5 == 0:
            print(f"   Processed batch {i} to {i+len(u0_batch)}...")

    return torch.cat(adversarial_batches, dim=0)

def spectral_pgd_targeted(model, u0_full_data):
    """
    Batched & Targeted Spectral PGD Attack.
    Attacks only the low-frequency modes (k <= MODES) that the FNO relies on.
    """
    model.eval()
    adversarial_batches = []
    
    total_samples = len(u0_full_data)
    cutoff = MODES[0]  # We only perturb the modes the model can "see"
    
    print(f"Running Targeted Spectral PGD on {total_samples} samples...")
    print(f"   Targeting Wavenumbers k <= {cutoff}")

    # 2. Iterate in Mini-Batches
    for i in range(0, total_samples, ATTACK_BATCH_SIZE):
        u0_batch = u0_full_data[i : i + ATTACK_BATCH_SIZE]
        current_bs = len(u0_batch)
        
        # --- A. PREPARE CLEAN TARGET ---
        # We need the clean prediction to maximize divergence from it
        with torch.no_grad():
            clean_in = u0_batch.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)                
            target_pred = model(clean_in).detach()

        # --- B. INIT SPECTRAL DELTAS ---
        # Convert to freq domain
        u_hat = torch.fft.rfft(u0_batch, dim=1)
        
        # Initialize perturbation ONLY for low frequencies
        # Shape: (Batch, cutoff)
        delta_re = torch.zeros(current_bs, cutoff, device=DEVICE).uniform_(-0.1, 0.1)
        delta_im = torch.zeros(current_bs, cutoff, device=DEVICE).uniform_(-0.1, 0.1)
        
        delta_re.requires_grad = True
        delta_im.requires_grad = True
        
        # --- C. PGD LOOP ---
        for step in range(PGD_STEPS):
            # 1. Reconstruct Full Spectrum Delta
            # Create a full-size zero tensor and fill in the low-freq corner
            full_delta_re = torch.zeros_like(u_hat.real)
            full_delta_im = torch.zeros_like(u_hat.imag)
            
            full_delta_re[:, :cutoff] = delta_re
            full_delta_im[:, :cutoff] = delta_im
            
            delta_c = torch.complex(full_delta_re, full_delta_im)
            
            # 2. IFFT to Physical Domain
            u_adv = torch.fft.irfft(u_hat + delta_c, n=NX, dim=1)
            
            # 3. Prepare Model Input
            input_u0 = u_adv.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
                        
            pred = model(input_u0)
            
            # 4. Loss: Maximize Divergence (Negative MSE)
            loss = -1.0 * torch.nn.MSELoss()(pred, target_pred)
            
            if delta_re.grad is not None:
                delta_re.grad.zero_()
                delta_im.grad.zero_()
                
            loss.backward()
            
            # 5. Update (In-Place)
            with torch.no_grad():
                # Gradient Ascent
                delta_re.data -= PGD_ALPHA * delta_re.grad.sign()
                delta_im.data -= PGD_ALPHA * delta_im.grad.sign()
                
                # Constraint: Clamp spectral coefficients
                # This keeps the perturbation magnitude reasonable
                delta_re.data.clamp_(-0.2, 0.2)
                delta_im.data.clamp_(-0.2, 0.2)
                
                delta_re.grad.zero_()
                delta_im.grad.zero_()
                
        # --- D. FINALIZE BATCH ---
        with torch.no_grad():
            full_delta_re = torch.zeros_like(u_hat.real)
            full_delta_im = torch.zeros_like(u_hat.imag)
            full_delta_re[:, :cutoff] = delta_re
            full_delta_im[:, :cutoff] = delta_im
            final_delta = torch.complex(full_delta_re, full_delta_im)
            
            res_batch = torch.fft.irfft(u_hat + final_delta, n=NX, dim=1).detach()
            adversarial_batches.append(res_batch)
        
        if (i // ATTACK_BATCH_SIZE) % 5 == 0:
            print(f"   Processed batch {i} to {i+len(u0_batch)}...")

    return torch.cat(adversarial_batches, dim=0)

def spectral_pi_pgd(model, u0_full_data):
    """
    Physics-informed spectral PGD:
    maximize prediction divergence while penalizing PDE residuals
    and boundary-condition violations (low-frequency perturbations).
    """
    model.eval()
    adversarial_batches = []
    total_samples = len(u0_full_data)
    cutoff = MODES[0] if isinstance(MODES, (list, tuple)) else MODES
    dt = T_EVAL[1] - T_EVAL[0] if len(T_EVAL) > 1 else 1.0

    print(f"Running Physics-Informed Spectral PGD on {total_samples} samples...")
    print(f"   Targeting Wavenumbers k <= {cutoff}")
    print(f"   lambda_pde={LAMBDA_PDE}, lambda_bc={LAMBDA_BC}")

    for i in range(0, total_samples, ATTACK_BATCH_SIZE):
        u0_batch = u0_full_data[i : i + ATTACK_BATCH_SIZE]
        current_bs = len(u0_batch)

        with torch.no_grad():
            clean_in = u0_batch.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
            target_pred = model(clean_in).detach()

        u_hat = torch.fft.rfft(u0_batch, dim=1)
        delta_re = torch.zeros(current_bs, cutoff, device=DEVICE).uniform_(-PGD_EPSILON, PGD_EPSILON)
        delta_im = torch.zeros(current_bs, cutoff, device=DEVICE).uniform_(-PGD_EPSILON, PGD_EPSILON)
        delta_re.requires_grad = True
        delta_im.requires_grad = True

        for _ in range(PGD_STEPS):
            full_delta_re = torch.zeros_like(u_hat.real)
            full_delta_im = torch.zeros_like(u_hat.imag)
            full_delta_re[:, :cutoff] = delta_re
            full_delta_im[:, :cutoff] = delta_im
            delta_c = torch.complex(full_delta_re, full_delta_im)

            u_adv = torch.fft.irfft(u_hat + delta_c, n=NX, dim=1)
            input_u0 = u_adv.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
            pred = model(input_u0)

            mse_loss = F.mse_loss(pred, target_pred)
            pde_loss = compute_pde_residual_loss(pred, NU, DX, dt)
            bc_loss = compute_bc_loss(pred)
            loss = -mse_loss + LAMBDA_PDE * pde_loss + LAMBDA_BC * bc_loss

            if delta_re.grad is not None:
                delta_re.grad.zero_()
                delta_im.grad.zero_()

            loss.backward()

            with torch.no_grad():
                delta_re.data -= PGD_ALPHA * delta_re.grad.sign()
                delta_im.data -= PGD_ALPHA * delta_im.grad.sign()

                delta_re.data.clamp_(-PGD_EPSILON, PGD_EPSILON)
                delta_im.data.clamp_(-PGD_EPSILON, PGD_EPSILON)

                delta_re.grad.zero_()
                delta_im.grad.zero_()

        with torch.no_grad():
            full_delta_re = torch.zeros_like(u_hat.real)
            full_delta_im = torch.zeros_like(u_hat.imag)
            full_delta_re[:, :cutoff] = delta_re
            full_delta_im[:, :cutoff] = delta_im
            final_delta = torch.complex(full_delta_re, full_delta_im)
            res_batch = torch.fft.irfft(u_hat + final_delta, n=NX, dim=1).detach()
            adversarial_batches.append(res_batch)

        if (i // ATTACK_BATCH_SIZE) % 5 == 0:
            print(f"   Processed batch {i} to {i+len(u0_batch)}...")

    return torch.cat(adversarial_batches, dim=0)

def spatial_pi_pgd(model, u0_full_data):
    """
    Physics-informed spatial PGD:
    maximize prediction divergence while penalizing PDE residuals
    and boundary-condition violations in the physical domain.
    """
    model.eval()
    adversarial_batches = []
    total_samples = len(u0_full_data)
    dt = T_EVAL[1] - T_EVAL[0] if len(T_EVAL) > 1 else 1.0

    print(f"Running Physics-Informed Spatial PGD on {total_samples} samples...")
    print(f"   lambda_pde={LAMBDA_PDE}, lambda_bc={LAMBDA_BC}")

    if SMOOTH_GRAD:
        kernel = get_smoothing_kernel(KERNEL_SIZE, SIGMA, DEVICE)
        pad = KERNEL_SIZE // 2

    for i in range(0, total_samples, ATTACK_BATCH_SIZE):
        u0_batch = u0_full_data[i : i + ATTACK_BATCH_SIZE]

        with torch.no_grad():
            input_clean = u0_batch.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
            target_pred = model(input_clean).detach()

        delta = torch.zeros_like(u0_batch).uniform_(-PGD_EPSILON, PGD_EPSILON)
        delta.requires_grad = True

        for _ in range(PGD_STEPS):
            perturbed_u0 = torch.clamp(u0_batch + delta, -2.0, 2.0)
            input_u0 = perturbed_u0.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
            pred = model(input_u0)

            mse_loss = F.mse_loss(pred, target_pred)
            pde_loss = compute_pde_residual_loss(pred, NU, DX, dt)
            bc_loss = compute_bc_loss(pred)
            loss = -mse_loss + LAMBDA_PDE * pde_loss + LAMBDA_BC * bc_loss

            if delta.grad is not None:
                delta.grad.zero_()

            loss.backward()

            with torch.no_grad():
                grad = delta.grad
                if SMOOTH_GRAD:
                    grad_input = grad.unsqueeze(1)
                    grad_smooth = F.conv1d(grad_input, kernel, padding=pad)
                    grad = grad_smooth.squeeze(1)

                delta.data -= PGD_ALPHA * grad.sign()
                delta.data = torch.clamp(delta.data, -PGD_EPSILON, PGD_EPSILON)
                delta.grad.zero_()

        adversarial_batches.append((u0_batch + delta).detach())

        if (i // ATTACK_BATCH_SIZE) % 5 == 0:
            print(f"   Processed batch {i} to {i+len(u0_batch)}...")

    return torch.cat(adversarial_batches, dim=0)

def spectral_pgd_amplitude(model, u0_full_data):
    """
    Batched & Targeted Spectral PGD Attack.
    Attacks only the low-frequency modes (k <= MODES) that the FNO relies on.
    Only changes the amplitudes of the spectral decomposition.

    """
    model.eval()
    adversarial_batches = []
    
    total_samples = len(u0_full_data)
    cutoff = MODES[0]  # We only perturb the modes the model can "see"
    
    print(f"Running Amplitude Spectral PGD on {total_samples} samples...")
    print(f"   Targeting Wavenumbers k <= {cutoff}")

    # 2. Iterate in Mini-Batches
    for i in tqdm(range(0, total_samples, ATTACK_BATCH_SIZE)):
        u0_batch = u0_full_data[i : i + ATTACK_BATCH_SIZE]
        current_bs = len(u0_batch)
        
        # --- A. PREPARE CLEAN TARGET ---
        # We need the clean prediction to maximize divergence from it
        with torch.no_grad():
            clean_in = u0_batch.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)                
            target_pred = model(clean_in).detach()

        # --- B. INIT SPECTRAL DELTAS ---
        # Convert to freq domain
        u_hat = torch.fft.rfft(u0_batch, dim=1)
        
        # Initialize perturbation ONLY for low frequencies
        # Shape: (Batch, cutoff)
        direction = u_hat / (u_hat.abs() + 1e-18)
        delta = torch.zeros(current_bs, cutoff, device=DEVICE).uniform_(-0.1, 0.1)
        
        delta.requires_grad = True
        
        # --- C. PGD LOOP ---
        for step in range(PGD_STEPS):
            # 1. Reconstruct Full Spectrum Delta
            # Create a full-size zero tensor and fill in the low-freq corner
            full_delta = torch.zeros_like(u_hat.real, device=DEVICE)
            
            full_delta[:, :cutoff] = delta
            
            delta_c = direction * full_delta
            
            # 2. IFFT to Physical Domain
            u_adv = torch.fft.irfft(u_hat + delta_c, n=NX, dim=1)
            
            # 3. Prepare Model Input
            input_u0 = u_adv.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
                        
            pred = model(input_u0)
            
            # 4. Loss: Maximize Divergence (Negative MSE)
            loss = -1.0 * torch.nn.MSELoss()(pred, target_pred)
            
            if delta.grad is not None:
                delta.grad.zero_()
                
            loss.backward()
            
            # 5. Update (In-Place)
            with torch.no_grad():
                # Gradient Ascent
                delta.data -= PGD_ALPHA * delta.grad.sign()
                
                # Constraint: Clamp spectral coefficients
                # This keeps the perturbation magnitude reasonable
                delta.data.clamp_(-AMPLITUDE_CLAMP, AMPLITUDE_CLAMP)
                
                delta.grad.zero_()
                
        # --- D. FINALIZE BATCH ---
        with torch.no_grad():
            full_delta = torch.zeros_like(u_hat.real)
            full_delta[:, :cutoff] = delta
            final_delta = direction * full_delta
            
            res_batch = torch.fft.irfft(u_hat + final_delta, n=NX, dim=1).detach()
            adversarial_batches.append(res_batch)

    return torch.cat(adversarial_batches, dim=0)


def spectral_pgd_phase(model, u0_full_data):
    """
    Batched & Targeted Spectral PGD Attack.
    Attacks only the low-frequency modes (k <= MODES) that the FNO relies on.
    Only changes the phase of the spectral decomposition.

    """
    model.eval()
    adversarial_batches = []
    
    total_samples = len(u0_full_data)
    cutoff = MODES[0]  # We only perturb the modes the model can "see"
    
    print(f"Running Phase Spectral PGD on {total_samples} samples...")
    print(f"   Targeting Wavenumbers k <= {cutoff}")

    # 2. Iterate in Mini-Batches
    for i in tqdm(range(0, total_samples, ATTACK_BATCH_SIZE)):
        u0_batch = u0_full_data[i : i + ATTACK_BATCH_SIZE]
        current_bs = len(u0_batch)
        
        # --- A. PREPARE CLEAN TARGET ---
        # We need the clean prediction to maximize divergence from it
        with torch.no_grad():
            clean_in = u0_batch.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
            target_pred = model(clean_in).detach()

        # --- B. INIT SPECTRAL DELTAS ---
        # Convert to freq domain
        u_hat = torch.fft.rfft(u0_batch, dim=1)

        # Initialize perturbation ONLY for low frequencies
        # Shape: (Batch, cutoff)
        delta_phi = torch.zeros(current_bs, cutoff, device=DEVICE).uniform_(-0.1, 0.1)

        delta_phi.requires_grad = True

        # --- C. PGD LOOP ---
        for step in range(PGD_STEPS):
            # 1. Reconstruct Full Spectrum Delta
            # Create a full-size zero tensor and fill in the low-freq corner
            full_delta_re = torch.zeros_like(u_hat.real)
            full_delta_im = torch.zeros_like(u_hat.imag)
            
            full_delta_re[:, :cutoff] = torch.cos(delta_phi)
            full_delta_im[:, :cutoff] = torch.sin(delta_phi)
            
            delta_c = torch.complex(full_delta_re, full_delta_im)
            
            # 2. IFFT to Physical Domain
            u_adv = torch.fft.irfft(u_hat * delta_c, n=NX, dim=1)
            
            # 3. Prepare Model Input
            input_u0 = u_adv.unsqueeze(1).unsqueeze(2).repeat(1, 1, NT, 1)
                        
            pred = model(input_u0)
            
            # 4. Loss: Maximize Divergence (Negative MSE)
            loss = -1.0 * torch.nn.MSELoss()(pred, target_pred)
            
            if delta_phi.grad is not None:
                delta_phi.grad.zero_()
                
            loss.backward()
            
            # 5. Update (In-Place)
            with torch.no_grad():
                # Gradient Ascent
                delta_phi.data -= PGD_ALPHA * delta_phi.grad.sign()
                # Constraint: Clamp spectral coefficients
                # This keeps the perturbation magnitude reasonable
                delta_phi.data.clamp_(-0.1, 0.1)
                delta_phi.grad.zero_()
                
        # --- D. FINALIZE BATCH ---
        with torch.no_grad():
            full_delta_re = torch.zeros_like(u_hat.real)
            full_delta_im = torch.zeros_like(u_hat.imag)
            full_delta_re[:, :cutoff] = torch.cos(delta_phi)
            full_delta_im[:, :cutoff] = torch.sin(delta_phi)
            final_delta = torch.complex(full_delta_re, full_delta_im)
            
            res_batch = torch.fft.irfft(u_hat * final_delta, n=NX, dim=1).detach()
            adversarial_batches.append(res_batch)

    return torch.cat(adversarial_batches, dim=0)


# --- 4. MAIN ---

def save_dataset(filename, u_sols):
    path = os.path.join(OUTPUT_DIR, filename)
    with h5py.File(path, 'w') as f:
        f.create_dataset('train/u', data=u_sols)
        f.create_dataset('x-coordinate', data=X_GRID)
        f.create_dataset('t-coordinate', data=T_EVAL)
        f.attrs['nu'] = NU
    print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description="Run PGD Attack on FNO Model")

    # -- Config & Paths --
    parser.add_argument("--config", type=str, default="simple_model.toml", help="Path to TOML model config")
    parser.add_argument("--model_path", type=str, default='best_fno_model_pdebench.pth')
    parser.add_argument("--output_dir", type=str, default='data/robustness_suite_k32')

    # -- Generation Settings --
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--nt", type=int, default=201)
    parser.add_argument("--attack_batch_size", type=int, default=10)

    # -- PGD Parameters --
    parser.add_argument("--pgd_epsilon", type=float, default=0.05)
    parser.add_argument("--pgd_steps", type=int, default=200)
    parser.add_argument("--pgd_alpha", type=float, default=0.01)
    parser.add_argument("--kernel_size", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--lambda_pde", type=float, default=0.0, help="Physics penalty weight")
    parser.add_argument("--lambda_bc", type=float, default=0.0, help="Boundary condition penalty weight")
    parser.add_argument("--amplitude_clamp", type=float, default=2.0)
    parser.add_argument("--phase_clamp", type=float, default=0.1)

    # -- Boolean for Smoothing (Default is True, so we create a flag to turn it OFF) --
    parser.add_argument("--no_smoothing", dest="smooth_grad", action="store_false", default=True,
                        help="Disable gradient smoothing (enabled by default)")
    # Parse args
    args = parser.parse_args()

    # Load TOML
    try:
        with open(args.config, "rb") as f:
            toml_config = tomllib.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    # Initialize Globals
    init_globals(args, toml_config)
    
    # Create Output Directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading Model from {MODEL_PATH}...")
    try:
        model = FNO(
            n_modes=MODES,
            hidden_channels=WIDTH,
            in_channels=IN_CHANNELS, 
            n_layers=NUM_LAYERS,
            out_channels=OUT_CHANNELS,
        ).to(DEVICE)
        
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Generating Clean Base Samples...")
    u0_clean = generate_clean_ics(N_SAMPLES)
    
    # The batched functions now handle memory safety
    u0_spatial_adv = spatial_pgd(model, u0_clean)
    u0_spectral_adv = spectral_pgd_targeted(model, u0_clean)
    u0_spatial_pi_adv = spatial_pi_pgd(model, u0_clean)
    u0_spectral_pi_adv = spectral_pi_pgd(model, u0_clean)
    
    print("\n--- Processing Spatial Adversarial Dataset ---")
    sol_spatial = solve_batch_ground_truth(u0_spatial_adv)
    save_dataset('eval_pgd_spatial.h5', sol_spatial)
    
    print("\n--- Processing Spectral Adversarial Dataset ---")
    sol_spectral = solve_batch_ground_truth(u0_spectral_adv)
    save_dataset('eval_pgd_spectral.h5', sol_spectral)

    print("\n--- Processing Physics-Informed Spatial Adversarial Dataset ---")
    sol_spatial_pi = solve_batch_ground_truth(u0_spatial_pi_adv)
    save_dataset('eval_pgd_spatial_pi.h5', sol_spatial_pi)

    print("\n--- Processing Physics-Informed Spectral Adversarial Dataset ---")
    sol_spectral_pi = solve_batch_ground_truth(u0_spectral_pi_adv)
    save_dataset('eval_pgd_spectral_pi.h5', sol_spectral_pi)
    
    print("\nDONE.")

if __name__ == "__main__":
    main()
