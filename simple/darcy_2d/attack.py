import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tomllib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import random
import itertools
import copy

# IMPORT SHARED SOLVER
import simple.darcy_2d.solver as solver
from models.model_factory import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. UTILITIES & LOSS FUNCTIONS
# ==========================================

def resize_tensor(x, target_res, mode='bilinear'):
    if x.shape[-1] == target_res: return x
    return F.interpolate(x, size=(target_res, target_res), mode=mode, align_corners=True)

class DarcyPhysicsLoss(nn.Module):
    def __init__(self, res, force=1.0, reduction='mean'):
        super().__init__()
        self.h = 1.0 / (res - 1)
        self.force = force
        self.reduction = reduction
        
        # Central Difference Kernels
        self.dx_kernel = torch.tensor([[[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]]], 
                                      device=DEVICE, dtype=torch.float32) / self.h
        self.dy_kernel = torch.tensor([[[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]]], 
                                      device=DEVICE, dtype=torch.float32) / self.h

    def compute_flux(self, u, a):
        u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate') 
        du_dx = F.conv2d(u_pad, self.dx_kernel)
        du_dy = F.conv2d(u_pad, self.dy_kernel)
        return a * du_dx, a * du_dy

    def forward(self, u, a):
        # Calculate Residual: -div(a * grad(u)) - f = 0
        flux_x, flux_y = self.compute_flux(u, a)
        d_flux_x_dx = F.conv2d(F.pad(flux_x, (1, 1, 1, 1), mode='replicate'), self.dx_kernel)
        d_flux_y_dy = F.conv2d(F.pad(flux_y, (1, 1, 1, 1), mode='replicate'), self.dy_kernel)
        
        residual = -(d_flux_x_dx + d_flux_y_dy) - self.force
        
        loss_pde = torch.mean(residual**2, dim=(1, 2, 3))
        
        # Dirichlet BC (u=0 at boundaries)
        loss_bc = (torch.mean(u[..., 0, :]**2, dim=(1, 2)) + 
                   torch.mean(u[..., -1, :]**2, dim=(1, 2)) + 
                   torch.mean(u[..., :, 0]**2, dim=(1, 2)) + 
                   torch.mean(u[..., :, -1]**2, dim=(1, 2)))
                   
        if self.reduction == 'mean':
            return loss_pde.mean(), loss_bc.mean()
        elif self.reduction == 'sum':
            return loss_pde.sum(), loss_bc.sum()
        else: 
            return loss_pde, loss_bc

def load_or_generate_data(cfg, data_conf, force, device):
    """
    Attempts to load data from the file specified in data_config.
    Falls back to generation ONLY if file is missing.
    """
    file_path = data_conf['config']['data_path']
    
    n_samples = cfg['general']['n_samples']
    res = cfg['general']['resolution']
    
    if file_path and os.path.exists(file_path):
        print(f"    -> Loading data from {file_path}")
        with h5py.File(file_path, 'r') as f:
            if 'nu' in f and 'tensor' in f:
                # Load a (Permeability) and u (Pressure)
                a_data = f['nu'][:n_samples]
                u_data = f['tensor'][:n_samples]
                
                a = torch.tensor(a_data, dtype=torch.float32).to(device)
                u = torch.tensor(u_data, dtype=torch.float32).to(device)
                
                # Add channel dim if missing (N, H, W) -> (N, 1, H, W)
                if a.ndim == 3: a = a.unsqueeze(1)
                if u.ndim == 3: u = u.unsqueeze(1)
                
                # Resize if resolution in config differs from file
                if a.shape[-1] != res:
                    print(f"    -> Resizing data from {a.shape[-1]} to {res}")
                    a = resize_tensor(a, res)
                    u = resize_tensor(u, res)
                    
                return a, u
            else:
                print("    -> H5 file structure mismatch. Generating...")
    else:
        print(f"    -> File not found ({file_path}). Generating...")

    # Fallback: Generate and Solve on-the-fly
    print("    -> Generating seeds and solving ground truth (this may take time)...")
    batch = []
    for _ in range(n_samples):
        batch.append(solver.generate_permeability(res, res, device=device))
    a = torch.stack(batch).unsqueeze(1)
    
    # Solve u (Ground Truth)
    u_list = []
    chunk_size = 16
    for i in range(0, n_samples, chunk_size):
        a_chunk = a[i:i+chunk_size]
        chunk_u = []
        for j in range(len(a_chunk)):
            nu_sample = torch.maximum(a_chunk[j, 0], torch.tensor(0.1, device=device))
            u_sol = solver.solve_steady_state(nu_sample, force=force)
            chunk_u.append(u_sol)
        u_list.append(torch.stack(chunk_u))
    u = torch.cat(u_list).unsqueeze(1)
    
    return a, u

# ==========================================
# 2. NOISE BASELINES
# ==========================================

def normalize_perturbation(delta, target_l2, x_ref):
    """
    Scales delta so that ||delta||_2 = target_l2 * ||x_ref||_2 (relative L2)
    target_l2: relative L2 norm (e.g., 0.05 means 5% of input norm)
    """
    if target_l2 is None: return delta
    
    delta_flat = delta.view(delta.shape[0], -1)
    delta_norms = torch.norm(delta_flat, p=2, dim=1)
    
    ref_flat = x_ref.view(x_ref.shape[0], -1)
    ref_norms = torch.norm(ref_flat, p=2, dim=1)
    
    # Calculate target magnitude (Relative)
    target_norms = target_l2 * ref_norms
    
    scaler = target_norms / (delta_norms + 1e-12)
    scaler = scaler.view(-1, 1, 1, 1)
    return delta * scaler


def compute_relative_bounds(x_ref, rel_eps):
    """
    Computes per-sample L-inf bounds based on relative epsilon.
    rel_eps: relative epsilon (e.g., 0.01 means 1% of max|x|)
    Returns: eps_bounds tensor of shape (B, 1, 1, 1) for broadcasting
    """
    # Compute per-sample max absolute value
    x_flat = x_ref.view(x_ref.shape[0], -1)
    max_vals = torch.max(torch.abs(x_flat), dim=1)[0]  # (B,)
    
    # Relative epsilon: fraction of max value
    eps_bounds = rel_eps * max_vals
    return eps_bounds.view(-1, 1, 1, 1)


def project_perturbation_relative(x_adv, x_clean, rel_eps):
    """
    Projects adversarial examples to satisfy relative L-inf constraint.
    ||x_adv - x_clean||_inf <= rel_eps * max|x_clean|
    """
    eps_bounds = compute_relative_bounds(x_clean, rel_eps)  # (B, 1, 1, 1)
    delta = x_adv - x_clean
    delta_clamped = torch.clamp(delta, -eps_bounds, eps_bounds)
    return x_clean + delta_clamped

def run_spectral_noise(x_seed, config, target_l2=0.05):
    print(f"    -> Running Spectral Noise (Target L2: {target_l2})")
    noise_level = config.get('noise', {}).get('spectral_std', 0.01)
    noise = torch.randn_like(x_seed) * noise_level
    
    if target_l2 is not None:
        noise = normalize_perturbation(noise, target_l2, x_seed)

    x_noisy = x_seed + noise
    return torch.clamp(x_noisy, min=0.01)

def run_spatial_noise(x_seed, config, target_l2=0.05):
    print(f"    -> Running Spatial Noise (Target L2: {target_l2})")
    n_inclusions = config.get('noise', {}).get('spatial_inclusions', 3)
    
    batch_size = 8
    num_samples = x_seed.shape[0]
    out_list = []

    for i in range(0, num_samples, batch_size):
        batch = x_seed[i : i+batch_size].clone()
        delta = torch.zeros_like(batch)
        B, C, H, W = batch.shape
        
        x_lin = torch.linspace(0, 1, H, device=x_seed.device)
        y_lin = torch.linspace(0, 1, W, device=x_seed.device)
        X, Y = torch.meshgrid(x_lin, y_lin, indexing='ij')
        
        for b in range(B):
            for _ in range(n_inclusions):
                cx, cy = random.random(), random.random()
                shape_type = random.choice(['circle', 'rect'])
                val = random.choice([0.1, 1.0])
                if shape_type == 'circle':
                    r = random.uniform(0.005, 0.015)
                    mask = ((X - cx)**2 + (Y - cy)**2) < r**2
                else:
                    w, h = random.uniform(0.005, 0.02), random.uniform(0.005, 0.02)
                    mask = (X > cx - w/2) & (X < cx + w/2) & (Y > cy - h/2) & (Y < cy + h/2)
                delta[b, 0, mask] = val
        
        if target_l2 is not None:
            delta = normalize_perturbation(delta, target_l2, batch)

        out_list.append(torch.clamp(batch + delta, min=0.01))
        
    return torch.cat(out_list)

# ==========================================
# 3. ATTACK A: PGD (Chunked)
# ==========================================

def run_pgd_attack(model, x_seed, y_target, config, mode='spatial', force=1.0, rel_eps=0.01, lambda_pde=0.0, lambda_bc=0.0, batch_size=8):
    """
    PGD attack with relative epsilon constraint.
    rel_eps: relative L-inf bound as fraction of max|x| per sample (e.g., 0.01 = 1%)
    """
    print(f"    -> Running PGD (Mode: {mode} | Rel Eps: {rel_eps:.2%} | Physics: {lambda_pde} | Batch: {batch_size})")
    model.eval()
    
    steps = config['pgd']['steps']
    alpha = config['pgd']['alpha']
    physics_loss_fn = DarcyPhysicsLoss(res=config['general']['resolution'], 
                                       force=force, reduction='none')

    num_samples = x_seed.shape[0]
    final_adv = []

    # OUTER LOOP: Mini-Batching to save Memory
    for i in range(0, num_samples, batch_size):
        x_chunk = x_seed[i : i+batch_size].clone().to(DEVICE)
        y_chunk = y_target[i : i+batch_size].to(DEVICE)
        
        # Compute per-sample relative epsilon bounds
        eps_bounds = compute_relative_bounds(x_chunk, rel_eps)  # (B, 1, 1, 1)
        
        # Init Parameter for this chunk
        if mode == 'spatial':
            param = x_chunk.clone().requires_grad_(True)
        elif mode == 'spectral':
            x_ft = torch.fft.rfft2(x_chunk)
            param = torch.zeros_like(x_ft, requires_grad=True)
            # For spectral mode, compute eps bound in frequency domain
            spectral_eps = rel_eps * torch.abs(x_ft).max()
        elif mode == 'amplitude':
            x_ft = torch.fft.rfft2(x_chunk)
            clean_amp = torch.abs(x_ft)
            clean_phase = torch.angle(x_ft)
            param = torch.zeros_like(clean_amp, requires_grad=True)
            spectral_eps = rel_eps * clean_amp.max()
        elif mode == 'phase':
            x_ft = torch.fft.rfft2(x_chunk)
            clean_amp = torch.abs(x_ft)
            clean_phase = torch.angle(x_ft)
            param = torch.zeros_like(clean_phase, requires_grad=True)
            spectral_eps = rel_eps * np.pi  # Phase bounded relative to pi
            
        # INNER LOOP: Optimization Steps
        for step in range(steps):
            # 1. Reconstruct Input from Parameter
            if mode == 'spatial': curr_x = param
            elif mode == 'spectral': curr_x = torch.fft.irfft2(x_ft + param, s=x_chunk.shape[-2:])
            elif mode == 'amplitude': curr_x = torch.fft.irfft2((clean_amp + param) * torch.exp(1j * clean_phase), s=x_chunk.shape[-2:])
            elif mode == 'phase': curr_x = torch.fft.irfft2(clean_amp * torch.exp(1j * (clean_phase + param)), s=x_chunk.shape[-2:])

            # 2. Forward Pass
            out = model(curr_x)
            
            # 3. Loss Calculation
            # Targeted against ORIGINAL ground truth (Robustness/Stability)
            mse_vec = F.mse_loss(out, y_chunk, reduction='none').mean(dim=(1,2,3))
            pde_vec, bc_vec = physics_loss_fn(out, curr_x)
            
            # Maximize Error, Minimize Physics Violation
            loss_vec = mse_vec - (lambda_pde * pde_vec) - (lambda_bc * bc_vec)
            loss = -loss_vec.sum() * 1e9 # Scale up for gradient stability
            
            # 4. Update
            model.zero_grad()
            if param.grad is not None: param.grad.zero_()
            loss.backward()
            
            grad = param.grad.data
            if grad.is_complex(): 
                param.data -= alpha * torch.sgn(grad)
            else: 
                param.data -= alpha * grad.sign()       
            
            # 5. Projection (Relative)
            if mode == 'spatial':
                # Per-sample relative L-inf projection
                delta = param.data - x_chunk.data
                delta_clamped = torch.clamp(delta, -eps_bounds, eps_bounds)
                param.data = x_chunk.data + delta_clamped
            elif mode == 'spectral':
                norm = torch.abs(param.data); mask = norm > spectral_eps
                param.data[mask] = param.data[mask] / norm[mask] * spectral_eps
            elif mode in ['amplitude', 'phase']:
                param.data = torch.clamp(param.data, -spectral_eps, spectral_eps)
            param.grad.zero_()
            
        # Final reconstruction for this chunk
        if mode == 'spatial': chunk_res = param.detach()
        elif mode == 'spectral': chunk_res = torch.fft.irfft2(x_ft + param, s=x_chunk.shape[-2:]).detach()
        elif mode == 'amplitude': chunk_res = torch.fft.irfft2((clean_amp + param) * torch.exp(1j * clean_phase), s=x_chunk.shape[-2:]).detach()
        elif mode == 'phase': chunk_res = torch.fft.irfft2(clean_amp * torch.exp(1j * (clean_phase + param)), s=x_chunk.shape[-2:]).detach()
        
        final_adv.append(chunk_res.cpu())
        
        del param, x_chunk, y_chunk, loss
        torch.cuda.empty_cache()

    return torch.cat(final_adv).to(DEVICE)

# ==========================================
# 4. ATTACK B: MVMO (Chunked)
# ==========================================

def run_mvmo_attack(model, x_seed, y_target, config, force=1.0, rel_eps=0.01, batch_size=8):
    """
    MVMO attack with relative epsilon constraint.
    rel_eps: relative L-inf bound as fraction of max|x| per sample
    """
    print(f"    -> Running MVMO Attack (Rel Eps: {rel_eps:.2%} | Batch: {batch_size})")
    model.eval()
    
    steps = config['pgd']['steps']
    alpha = config['pgd']['alpha']
    pde_threshold = config['physics'].get('mvmo_threshold', 1e-4)
    physics_loss_fn = DarcyPhysicsLoss(res=config['general']['resolution'], 
                                       force=force, reduction='none')

    num_samples = x_seed.shape[0]
    final_adv = []

    for i in range(0, num_samples, batch_size):
        x_chunk = x_seed[i : i+batch_size].clone().to(DEVICE)
        y_chunk = y_target[i : i+batch_size].to(DEVICE)
        param = x_chunk.clone().requires_grad_(True)
        
        # Compute per-sample relative epsilon bounds
        eps_bounds = compute_relative_bounds(x_chunk, rel_eps)  # (B, 1, 1, 1)
        
        for step in range(steps):
            out = model(param)
            mse_vec = F.mse_loss(out, y_chunk, reduction='none').mean(dim=(1, 2, 3))
            pde_vec, bc_vec = physics_loss_fn(out, param)
            total_physics_vec = pde_vec + bc_vec
            
            # Switch Loss based on Physics Violation
            fix_physics_mask = total_physics_vec > pde_threshold
            targets = torch.where(fix_physics_mask, total_physics_vec * 1e5, -mse_vec * 1e5)
            loss = targets.sum()
                
            model.zero_grad()
            if param.grad is not None: param.grad.zero_()
            loss.backward()
            
            grad = param.grad.data
            param.data -= alpha * grad.sign()
            
            # Relative projection
            delta = param.data - x_chunk.data
            delta_clamped = torch.clamp(delta, -eps_bounds, eps_bounds)
            param.data = x_chunk.data + delta_clamped
            param.grad.zero_()
            
        final_adv.append(param.detach().cpu())
        del param, x_chunk, y_chunk, loss
        torch.cuda.empty_cache()

    return torch.cat(final_adv).to(DEVICE)

# ==========================================
# 5. ATTACK C: BOUNDARY (Chunked)
# ==========================================

def run_boundary_attack(model, x_seed, y_target, config, rel_eps=0.01, batch_size=8):
    """
    Boundary attack with relative epsilon constraint.
    rel_eps: relative L-inf bound as fraction of max|x| per sample
    """
    print(f"    -> Running Boundary Attack (Rel Eps: {rel_eps:.2%})")
    model.eval()
    
    steps = config['pgd']['steps']
    alpha = config['pgd']['alpha']
    
    num_samples = x_seed.shape[0]
    final_adv = []
    
    # Boundary mask is independent of batch
    mask = torch.zeros(1, 1, x_seed.shape[-2], x_seed.shape[-1], device=DEVICE)
    mask[..., 0, :] = 1.0; mask[..., -1, :] = 1.0 
    mask[..., :, 0] = 1.0; mask[..., :, -1] = 1.0
    
    for i in range(0, num_samples, batch_size):
        x_chunk = x_seed[i : i+batch_size].clone().to(DEVICE)
        y_chunk = y_target[i : i+batch_size].to(DEVICE)
        param = x_chunk.clone().requires_grad_(True)
        
        # Compute per-sample relative epsilon bounds
        eps_bounds = compute_relative_bounds(x_chunk, rel_eps)  # (B, 1, 1, 1)
        
        for _ in range(steps):
            out = model(param)
            mse_vec = F.mse_loss(out, y_chunk, reduction='none').mean(dim=(1, 2, 3))
            loss = -mse_vec.sum() * 1e9 
            
            model.zero_grad()
            if param.grad is not None: param.grad.zero_()
            loss.backward()
            
            grad = param.grad.data * mask 
            param.data -= alpha * grad.sign()
            
            # Relative projection (masked to boundary)
            delta = param.data - x_chunk.data
            delta_clamped = torch.clamp(delta, -eps_bounds, eps_bounds)
            delta_clamped = delta_clamped * mask 
            param.data = x_chunk.data + delta_clamped
            param.grad.zero_()
            
        final_adv.append(param.detach().cpu())
        del param, x_chunk, y_chunk, loss
        torch.cuda.empty_cache()

    return torch.cat(final_adv).to(DEVICE)

# ==========================================
# 7. MAIN EXECUTION & EVALUATION
# ==========================================
def ensure_list(x):
    return x if isinstance(x, list) else [x]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Darcy Attack Suite (Grid Search)")
    parser.add_argument("--attack_config", type=str, default="attack_config.toml")
    args = parser.parse_args()

    if not os.path.exists(args.attack_config):
        sys.exit(f"Config not found: {args.attack_config}")
    
    with open(args.attack_config, "rb") as f: cfg = tomllib.load(f)
    
    model_conf_path = cfg['model']['model_config_path']
    with open(model_conf_path, "rb") as f: model_conf = tomllib.load(f)["config"]

    data_conf_path = cfg['model']['data_config_path']
    with open(data_conf_path, "rb") as f: data_conf = tomllib.load(f)

    Path(cfg['general']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    model = create_model(model_conf, DEVICE)
    model.load_state_dict(torch.load(cfg['model']['model_path'], map_location=DEVICE)['model_state_dict'])
    model.eval()

    print(f"2. Loading Data...")
    a_clean, u_clean = load_or_generate_data(cfg, data_conf, force=data_conf['config']['force_term'], device=DEVICE)
    
    eval_physics_fn = DarcyPhysicsLoss(res=cfg["general"]["resolution"], force=data_conf['config']['force_term'], reduction='mean')
    atk_bs = cfg['pgd']['batch_size']

    # ---------------------------------------
    # GRID SEARCH LOOP
    # ---------------------------------------
    
    # Now using relative epsilon (as fraction of max|x|) and relative L2 (as fraction of ||x||)
    rel_eps_list = ensure_list(cfg['pgd'].get('rel_eps', cfg['pgd'].get('epsilon', 0.01)))
    rel_l2_list = ensure_list(cfg['pgd'].get('rel_l2', cfg['pgd'].get('target_l2', 0.05)))
    pde_list = ensure_list(cfg['physics'].get('lambda_pde', 0.0))
    bc_list = ensure_list(cfg['physics'].get('lambda_bc', 0.0))

    def get_suffix(p_rel_eps, p_rel_l2, p_pde, p_bc, is_noise=False):
        s = ""
        if not is_noise:
            if len(rel_eps_list) > 1: s += f"_eps{p_rel_eps}"
            if len(pde_list) > 1 or p_pde > 0: s += f"_pde{p_pde}"
            if len(bc_list) > 1 or p_bc > 0: s += f"_bc{p_bc}"
        else:
            if len(rel_l2_list) > 1: s += f"_l2{p_rel_l2}"
        return s

    def save_h5(filename, a_adv):
        out_file = f"{cfg['general']['output_dir']}/{filename}"
        with h5py.File(out_file, 'w') as f:
            f.create_dataset('nu', data=a_adv.squeeze(1).cpu().numpy())
            # Save the ORIGINAL Clean Ground Truth
            f.create_dataset('tensor', data=u_clean.squeeze(1).cpu().numpy())
            f.create_dataset('original_nu', data=a_clean.squeeze(1).cpu().numpy())
        print(f"Saved: {out_file}")

    def save_attack_params(filename, p_rel_eps, p_rel_l2, p_pde, p_bc):
        with open(f"{cfg['general']['output_dir']}/{filename}", 'w') as f:
            json.dump({
                "rel_eps": p_rel_eps,
                "rel_l2": p_rel_l2,
                "lambda_pde": p_pde,
                "lambda_bc": p_bc
            }, f)
        print(f"Saved: {filename}")

    # --- LOOP A: OPTIMIZATION ATTACKS ---
    for rel_eps in rel_eps_list:
        suffix = get_suffix(rel_eps, None, 0.0, 0.0)
        if cfg['active_attacks'].get('spatial_pure', False):
            print(f"\n>>> EXECUTE: Spatial PGD (rel_eps={rel_eps:.2%})")
            a_adv = run_pgd_attack(model, a_clean, u_clean, cfg, mode='spatial', force=data_conf['config']['force_term'], rel_eps=rel_eps,
                                   lambda_pde=0.0, lambda_bc=0.0, batch_size=atk_bs)
            save_h5(f'attack_pgd_spatial{suffix}.h5', a_adv)
            save_attack_params(f'attack_pgd_spatial{suffix}.json', rel_eps, None, 0.0, 0.0)

        if cfg['active_attacks'].get('spectral_pure', False):
            print(f"\n>>> EXECUTE: Spectral PGD (rel_eps={rel_eps:.2%})")
            a_adv = run_pgd_attack(model, a_clean, u_clean, cfg, mode='spectral', force=data_conf['config']['force_term'], rel_eps=rel_eps,
                                   lambda_pde=0.0, lambda_bc=0.0, batch_size=atk_bs)
            save_h5(f'attack_pgd_spectral{suffix}.h5', a_adv)
            save_attack_params(f'attack_pgd_spectral{suffix}.json', rel_eps, None, 0.0, 0.0)

        if cfg['active_attacks'].get('mvmo', False):
            print(f"\n>>> EXECUTE: MVMO Attack (rel_eps={rel_eps:.2%})")
            a_adv = run_mvmo_attack(model, a_clean, u_clean, cfg, force=data_conf['config']['force_term'], rel_eps=rel_eps, batch_size=atk_bs)
            save_h5(f'attack_mvmo{suffix}.h5', a_adv)
            save_attack_params(f'attack_mvmo{suffix}.json', rel_eps, None, 0.0, 0.0)

        if cfg['active_attacks'].get('boundary', False):
            print(f"\n>>> EXECUTE: Boundary Attack (rel_eps={rel_eps:.2%})")
            a_adv = run_boundary_attack(model, a_clean, u_clean, cfg, rel_eps=rel_eps, batch_size=atk_bs)
            save_h5(f'attack_boundary{suffix}.h5', a_adv)
            save_attack_params(f'attack_boundary{suffix}.json', rel_eps, None, 0.0, 0.0)

        for pde, bc in zip(pde_list, bc_list):
            suffix = get_suffix(rel_eps, None, pde, bc)
            if cfg['active_attacks'].get('spatial_stealth', False):
                print(f"\n>>> EXECUTE: Spatial Stealth (rel_eps={rel_eps:.2%}, pde={pde})")
                a_adv = run_pgd_attack(model, a_clean, u_clean, cfg, mode='spatial', force=data_conf['config']['force_term'], rel_eps=rel_eps,
                                    lambda_pde=pde, lambda_bc=bc, batch_size=atk_bs)
                save_h5(f'attack_pgd_spatial_stealth{suffix}.h5', a_adv)
                save_attack_params(f'attack_pgd_spatial_stealth{suffix}.json', rel_eps, None, pde, bc)

            if cfg['active_attacks'].get('spectral_stealth', False):
                print(f"\n>>> EXECUTE: Spectral Stealth (rel_eps={rel_eps:.2%}, pde={pde})")
                a_adv = run_pgd_attack(model, a_clean, u_clean, cfg, mode='spectral', force=data_conf['config']['force_term'], rel_eps=rel_eps,
                                    lambda_pde=pde, lambda_bc=bc, batch_size=atk_bs)
                save_h5(f'attack_pgd_spectral_stealth{suffix}.h5', a_adv)
                save_attack_params(f'attack_pgd_spectral_stealth{suffix}.json', rel_eps, None, pde, bc)

    # --- LOOP B: NOISE BASELINES ---
    for rel_l2 in rel_l2_list:
        run_cfg = copy.deepcopy(cfg)
        run_cfg['pgd']['rel_l2'] = rel_l2
        suffix = get_suffix(None, rel_l2, None, None, is_noise=True)

        if cfg['active_attacks'].get('spectral_noise', False):
            print(f"\n>>> EXECUTE: Spectral Noise (rel_l2={rel_l2:.2%})")
            a_adv = run_spectral_noise(a_clean, run_cfg, target_l2=rel_l2)
            save_h5(f'attack_noise_spectral{suffix}.h5', a_adv)
            save_attack_params(f'attack_noise_spectral{suffix}.json', None, rel_l2, None, None)

        if cfg['active_attacks'].get('spatial_noise', False):
            print(f"\n>>> EXECUTE: Spatial Noise (rel_l2={rel_l2:.2%})")
            a_adv = run_spatial_noise(a_clean, run_cfg, target_l2=rel_l2)
            save_h5(f'attack_noise_spatial{suffix}.h5', a_adv)
            save_attack_params(f'attack_noise_spatial{suffix}.json', None, rel_l2, None, None)

    print("\nAll requested attacks completed.")