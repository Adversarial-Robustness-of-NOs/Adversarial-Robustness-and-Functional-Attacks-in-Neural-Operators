import argparse
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
from tqdm import tqdm

# Import your existing modules
from models.model_factory import create_model
from temporal.ns_2d.ns import NavierStokes2D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. UTILITIES & LOSS FUNCTIONS
# ==========================================

def get_spectral_mask(spatial_shape, config, device):
    """
    Creates a boolean mask for FFT frequencies based on config.
    Returns: Tensor of shape (H, W//2 + 1) for RFFT masking.
    """
    H, W = spatial_shape
    
    # 1. Check if targeting is enabled
    freq_conf = config.get('frequency', {})
    if not freq_conf.get('spectral_targeted', False):
        return None # No masking
        
    min_k = freq_conf.get('target_min', 0)
    max_k = freq_conf.get('target_max', 1000) # Default to high if not set
    
    # 2. Generate Frequency Grid (RFFT standard)
    kx = torch.fft.fftfreq(H, d=1.0, device=device) * H
    ky = torch.fft.rfftfreq(W, d=1.0, device=device) * W
    
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    R = torch.sqrt(KX**2 + KY**2)
    
    # 3. Create Mask
    mask = (R >= min_k) & (R <= max_k)
    return mask.float()

def resize_tensor(x, target_res, mode='bilinear'):
    if x.shape[-1] == target_res: return x
    b, t, h, w = x.shape
    x_reshaped = x.view(b, t, h, w)
    out = F.interpolate(x_reshaped, size=(target_res, target_res), mode=mode, align_corners=True)
    return out


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


def normalize_perturbation(delta, target_l2, x_ref):
    """
    Scales delta so that ||delta||_2 = target_l2 * ||x_ref||_2 (relative L2)
    target_l2: relative L2 norm (e.g., 0.05 means 5% of input norm)
    """
    if target_l2 is None: 
        return delta
    
    delta_flat = delta.view(delta.shape[0], -1)
    delta_norms = torch.norm(delta_flat, p=2, dim=1)
    
    ref_flat = x_ref.view(x_ref.shape[0], -1)
    ref_norms = torch.norm(ref_flat, p=2, dim=1)
    
    # Calculate target magnitude (Relative)
    target_norms = target_l2 * ref_norms
    
    scaler = target_norms / (delta_norms + 1e-12)
    scaler = scaler.view(-1, 1, 1, 1)
    return delta * scaler

def rollout(model, initial_data, in_channels, future_steps):
    model.eval()
    curr_x = initial_data
    preds = []
    preds.append(curr_x)
    for _ in range(future_steps):
        pred = model(curr_x)
        preds.append(pred)
        curr_x = torch.cat([curr_x[:, 1:], pred], dim=1)
    return torch.cat(preds, dim=1)

class NSPhysicsLoss(nn.Module):
    def __init__(self, problem_instance, reduction='mean'):
        super().__init__()
        self.problem = problem_instance
        self.reduction = reduction

    def forward(self, trajectory):
        # 1. PDE Loss
        if trajectory.shape[1] < 2: 
            if self.reduction == 'none':
                pde_loss = torch.zeros(trajectory.shape[0], device=DEVICE)
            else:
                pde_loss = torch.tensor(0.0, device=DEVICE)
        else:
            dt = self.problem.dt
            pde_losses_per_sample = []
            for t in range(1, trajectory.shape[1]):
                w_curr = trajectory[:, t:t+1]
                w_prev = trajectory[:, t-1:t]
                residual = compute_ns_residual(self.problem, w_curr, w_prev, dt)
                pde_loss_t = torch.mean(residual**2, dim=(1, 2, 3))
                pde_losses_per_sample.append(pde_loss_t)
            pde_loss_stack = torch.stack(pde_losses_per_sample, dim=1)
            if self.reduction == 'none':
                pde_loss = pde_loss_stack.mean(dim=1)
            else:
                pde_loss = pde_loss_stack.mean()
        return pde_loss

def compute_ns_residual(problem, w_curr, w_prev, dt):
    if w_curr.ndim == 3: w_curr = w_curr.unsqueeze(1)
    if w_prev.ndim == 3: w_prev = w_prev.unsqueeze(1)
    w_t = (w_curr - w_prev) / dt
    u, v, w_x, w_y, lap_w = problem.compute_spectral_physics(w_curr)
    advection = (u * w_x) + (v * w_y)
    diffusion = problem.visc * lap_w
    residual = w_t + advection - diffusion
    return residual

# ==========================================
# 2. NOISE BASELINES
# ==========================================

def run_spectral_noise(u0_clean, cfg, rel_l2):
    """Spectral noise with relative L2 norm constraint."""
    print(f"    -> Running Spectral Noise (Rel L2: {rel_l2:.2%})")
    B, T, H, W = u0_clean.shape
    noise = torch.randn_like(u0_clean)
    noise_fft = torch.fft.rfft2(noise, dim=(-2, -1))
    
    kx = torch.fft.fftfreq(H, d=1.0, device=u0_clean.device) * H
    ky = torch.fft.rfftfreq(W, d=1.0, device=u0_clean.device) * W
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    R = torch.sqrt(KX**2 + KY**2)
    
    high_pass = (R > 6).float()
    noise_fft = noise_fft * high_pass.unsqueeze(0).unsqueeze(0)
    noise = torch.fft.irfft2(noise_fft, s=(H, W), dim=(-2, -1))
    
    # Use relative L2 normalization
    noise = normalize_perturbation(noise, rel_l2, u0_clean)
    
    return (u0_clean + noise).cpu()

def run_spatial_noise(u0_clean, cfg, rel_l2):
    """Spatial noise with relative L2 norm constraint."""
    print(f"    -> Running Spatial Noise (Rel L2: {rel_l2:.2%})")
    B, T, H, W = u0_clean.shape
    noise = torch.randn_like(u0_clean)
    
    # Use relative L2 normalization
    noise = normalize_perturbation(noise, rel_l2, u0_clean)
    
    return (u0_clean + noise).cpu()

# ==========================================
# 3. PGD ATTACKS
# ==========================================

def run_pgd_attack(model, u0_clean, u_clean, cfg, rel_eps, mode='spatial', lambda_pde=0.0, lambda_bc=0.0):
    """
    PGD attack with relative L-inf epsilon constraint.
    rel_eps: relative epsilon (e.g., 0.01 means 1% of max|x|)
    
    For spectral mode: optimizes directly in frequency domain (like attack_darcy.py)
    For spatial mode: optimizes in spatial domain
    """
    print(f"    -> Running PGD (Mode: {mode} | Rel Eps: {rel_eps:.2%} | Physics: {lambda_pde})")
    model.eval()
    pgd_cfg = cfg['pgd']
    
    B, T_in, H, W = u0_clean.shape
    T_total = u_clean.shape[1]
    future_steps = T_total - T_in
    in_channels = T_in
    
    print(f"    [DEBUG] PGD Config: steps={pgd_cfg.get('steps', 100)}, lr={pgd_cfg.get('lr', 0.01)}")
    print(f"    [DEBUG] Data shapes: u0_clean={u0_clean.shape}, u_clean={u_clean.shape}")
    print(f"    [DEBUG] future_steps={future_steps}")
    
    # Compute per-sample relative epsilon bounds (L-inf) for spatial mode
    eps_bounds = compute_relative_bounds(u0_clean, rel_eps)  # (B, 1, 1, 1)
    print(f"    [DEBUG] eps_bounds range: [{eps_bounds.min().item():.6f}, {eps_bounds.max().item():.6f}]")
    
    n_steps = pgd_cfg.get('steps', 100)
    lr = pgd_cfg.get('lr', 0.01)
    
    # Mode-specific initialization (matching attack_darcy.py for spectral)
    if mode == 'spectral':
        # Optimize directly in frequency domain (like attack_darcy.py)
        u0_fft = torch.fft.rfft2(u0_clean, dim=(-2, -1))
        spectral_eps = rel_eps * torch.abs(u0_fft).max()
        
        # Get optional frequency mask
        spec_mask = get_spectral_mask((H, W), cfg, u0_clean.device)
        
        # Initialize delta in frequency domain with small random values
        delta = torch.zeros_like(u0_fft)  # Complex tensor
        # Add small random perturbation for symmetry breaking
        delta_init_real = torch.randn_like(delta.real) * (spectral_eps.item() * 0.01)
        delta_init_imag = torch.randn_like(delta.imag) * (spectral_eps.item() * 0.01)
        delta = torch.complex(delta_init_real, delta_init_imag)
        delta = delta.requires_grad_(True)
        
        print(f"    [DEBUG] spectral_eps: {spectral_eps.item():.6f}")
        print(f"    [DEBUG] delta is complex: {delta.is_complex()}")
        print(f"    [DEBUG] spec_mask is None: {spec_mask is None}")
        
        optimizer = torch.optim.Adam([delta], lr=lr)
    else:
        # Spatial mode: optimize in spatial domain with random init
        clean_flat = u0_clean.view(B, -1)
        clean_norm = torch.norm(clean_flat, p=2, dim=1, keepdim=True)
        max_norm = rel_eps * clean_norm
        
        # Random init
        delta = torch.randn_like(u0_clean)
        
        # Normalize init to be small (1% of budget) to break symmetry
        delta_flat = delta.view(B, -1)
        init_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
        scale_init = (max_norm * 0.01) / (init_norm + 1e-12)
        delta = delta * scale_init.view(B, 1, 1, 1)
        delta = delta.detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([delta], lr=lr)
    
    print(f"    [DEBUG] delta requires_grad: {delta.requires_grad}")
    
    # Physics loss setup
    physics_loss_fn = None
    if lambda_pde > 0:
        problem = NavierStokes2D(cfg)
        physics_loss_fn = NSPhysicsLoss(problem, reduction='mean')
    
    # Main optimization loop
    for step in tqdm(range(n_steps), desc=f"PGD {mode} rel_eps={rel_eps:.2%}"):
        optimizer.zero_grad()
        
        # Reconstruct adversarial input
        if mode == 'spectral':
            # Apply frequency mask if specified
            if spec_mask is not None:
                delta_masked = delta * spec_mask.unsqueeze(0).unsqueeze(0)
            else:
                delta_masked = delta
            # Reconstruct spatial domain: x_adv = irfft(x_fft + delta)
            u0_adv = torch.fft.irfft2(u0_fft + delta_masked, s=(H, W), dim=(-2, -1))
        else:
            u0_adv = u0_clean + delta
        
        # Debug: Check if u0_adv differs from u0_clean (step 0 only)
        if step == 0:
            adv_clean_diff = (u0_adv - u0_clean).abs().max().item()
            print(f"    [DEBUG] Step 0: max|u0_adv - u0_clean| = {adv_clean_diff:.8f}")
        
        # Rollout
        pred_adv = rollout(model, u0_adv, in_channels, future_steps)
        
        # Use no_grad for pred_clean to avoid double backward issues
        with torch.no_grad():
            pred_clean = rollout(model, u0_clean, in_channels, future_steps)
        
        # Debug: Check if predictions differ (step 0 only)
        if step == 0:
            pred_diff = (pred_adv - pred_clean).abs().max().item()
            pred_diff_mean = (pred_adv - pred_clean).abs().mean().item()
            print(f"    [DEBUG] Step 0: max|pred_adv - pred_clean| = {pred_diff:.8f}")
            print(f"    [DEBUG] Step 0: mean|pred_adv - pred_clean| = {pred_diff_mean:.8f}")
        
        # Maximize prediction error
        pred_loss = -F.mse_loss(pred_adv, pred_clean)
        
        # Physics regularization
        phys_loss = torch.tensor(0.0, device=DEVICE)
        if physics_loss_fn is not None and lambda_pde > 0:
            phys_loss = physics_loss_fn(pred_adv)
        
        total_loss = pred_loss + lambda_pde * phys_loss
        total_loss.backward()
        
        # Debug: Check gradient on first few steps
        if step < 3 or step == n_steps - 1:
            if delta.grad is not None:
                grad_abs = torch.abs(delta.grad) if delta.is_complex() else delta.grad.abs()
                grad_max = grad_abs.max().item()
                grad_mean = grad_abs.mean().item()
            else:
                grad_max = 0
                grad_mean = 0
            delta_abs = torch.abs(delta.data) if delta.is_complex() else delta.data.abs()
            delta_max = delta_abs.max().item()
            print(f"    [DEBUG] Step {step}: loss={total_loss.item():.6f}, grad_max={grad_max:.8f}, grad_mean={grad_mean:.8f}, delta_max={delta_max:.8f}")
        
        optimizer.step()
        
        # Project to constraint set
        with torch.no_grad():
            if mode == 'spectral':
                # Spectral projection (like attack_darcy.py)
                norm = torch.abs(delta.data)
                mask = norm > spectral_eps
                delta.data[mask] = delta.data[mask] / norm[mask] * spectral_eps
            else:
                # Spatial projection: per-sample relative L-inf clamping
                delta.data = torch.clamp(delta.data, -eps_bounds, eps_bounds)
    
    # Final reconstruction
    with torch.no_grad():
        if mode == 'spectral':
            if spec_mask is not None:
                delta_masked = delta * spec_mask.unsqueeze(0).unsqueeze(0)
            else:
                delta_masked = delta
            u0_adv = torch.fft.irfft2(u0_fft + delta_masked, s=(H, W), dim=(-2, -1))
            delta_final = u0_adv - u0_clean  # Compute spatial delta for stats
        else:
            delta_final = delta
            u0_adv = u0_clean + delta_final
        
        # Debug: Show perturbation statistics
        delta_abs_max = delta_final.abs().max().item()
        delta_abs_mean = delta_final.abs().mean().item()
        delta_l2 = torch.norm(delta_final).item()
        clean_l2 = torch.norm(u0_clean).item()
        rel_pert = delta_l2 / (clean_l2 + 1e-8)
        
        print(f"    [DEBUG] Perturbation Stats:")
        print(f"       Delta max abs: {delta_abs_max:.6f}")
        print(f"       Delta mean abs: {delta_abs_mean:.6f}")  
        print(f"       Delta L2: {delta_l2:.6f}")
        print(f"       Clean L2: {clean_l2:.6f}")
        print(f"       Relative L2: {rel_pert:.2%}")
        
        # Verify u0_adv is different from u0_clean
        diff = (u0_adv - u0_clean).abs().max().item()
        print(f"       Verify: max|u0_adv - u0_clean| = {diff:.6f}")
    
    return u0_adv.cpu()

# ==========================================
# 4. SEQUENTIAL PGD ATTACKS
# ==========================================

def run_sequential_pgd_attack(model, u0_clean, u_clean, cfg, mode='spatial', rel_eps=0.01, lambda_pde=0.0, lambda_bc=0.0):
    """
    Sequential PGD attack with relative L-inf epsilon constraint.
    rel_eps: relative epsilon (e.g., 0.01 means 1% of max|x|)
    
    For spectral mode: optimizes directly in frequency domain
    For spatial mode: optimizes in spatial domain
    """
    print(f"    -> Running Sequential PGD (Mode: {mode} | Rel Eps: {rel_eps:.2%} | Physics: {lambda_pde})")
    model.eval()
    pgd_cfg = cfg['pgd']
    
    B, T_in, H, W = u0_clean.shape
    T_total = u_clean.shape[1]
    future_steps = T_total - T_in
    in_channels = T_in
    
    # Compute per-sample relative epsilon bounds (L-inf)
    eps_bounds = compute_relative_bounds(u0_clean, rel_eps)  # (B, 1, 1, 1)
    
    n_steps = pgd_cfg.get('steps', 100)
    lr = pgd_cfg.get('lr', 0.01)
    
    # Mode-specific setup
    if mode == 'spectral':
        u0_fft = torch.fft.rfft2(u0_clean, dim=(-2, -1))
        spectral_eps = rel_eps * torch.abs(u0_fft).max()
        spec_mask = get_spectral_mask((H, W), cfg, u0_clean.device)
        
        # Initialize deltas in frequency domain for each time step
        # Shape: (B, future_steps + 1, T_in, H, W//2+1) complex
        fft_shape = u0_fft.shape  # (B, T_in, H, W//2+1)
        deltas = torch.zeros(B, future_steps + 1, *fft_shape[1:], dtype=torch.complex64, device=u0_clean.device)
        # Small random init
        deltas_init_real = torch.randn_like(deltas.real) * (spectral_eps.item() * 0.01)
        deltas_init_imag = torch.randn_like(deltas.imag) * (spectral_eps.item() * 0.01)
        deltas = torch.complex(deltas_init_real, deltas_init_imag)
        deltas = deltas.requires_grad_(True)
    else:
        spec_mask = None
        spectral_eps = None
        # Initialize deltas in spatial domain for each time step
        deltas = torch.zeros(B, future_steps + 1, 1, H, W, device=u0_clean.device)
        # Small random init
        deltas = torch.randn_like(deltas) * (eps_bounds.max().item() * 0.01)
        deltas = deltas.requires_grad_(True)
    
    optimizer = torch.optim.Adam([deltas], lr=lr)
    
    physics_loss_fn = None
    if lambda_pde > 0:
        problem = NavierStokes2D(cfg)
        physics_loss_fn = NSPhysicsLoss(problem, reduction='mean')
    
    for step in tqdm(range(n_steps), desc=f"Sequential {mode} rel_eps={rel_eps:.2%}"):
        optimizer.zero_grad()
        
        # Apply perturbations at each time step
        curr_x = u0_clean.clone()
        all_preds = [curr_x]
        
        for t in range(future_steps):
            # Apply delta to input
            if mode == 'spectral':
                curr_fft = torch.fft.rfft2(curr_x, dim=(-2, -1))
                if spec_mask is not None:
                    delta_masked = deltas[:, t] * spec_mask.unsqueeze(0).unsqueeze(0)
                else:
                    delta_masked = deltas[:, t]
                curr_x_pert = torch.fft.irfft2(curr_fft + delta_masked, s=(H, W), dim=(-2, -1))
            else:
                delta_t = deltas[:, t]
                curr_x_pert = curr_x + delta_t
            
            pred = model(curr_x_pert)
            all_preds.append(pred)
            curr_x = torch.cat([curr_x[:, 1:], pred], dim=1)
        
        pred_adv = torch.cat(all_preds, dim=1)
        
        with torch.no_grad():
            pred_clean = rollout(model, u0_clean, in_channels, future_steps)
        
        # Maximize prediction error
        pred_loss = -F.mse_loss(pred_adv, pred_clean)
        
        # Physics regularization
        phys_loss = torch.tensor(0.0, device=DEVICE)
        if physics_loss_fn is not None and lambda_pde > 0:
            phys_loss = physics_loss_fn(pred_adv)
        
        total_loss = pred_loss + lambda_pde * phys_loss
        total_loss.backward()
        optimizer.step()
        
        # Project each delta to constraint set
        with torch.no_grad():
            for t in range(future_steps + 1):
                if mode == 'spectral':
                    # Spectral projection (like attack_darcy.py)
                    norm = torch.abs(deltas.data[:, t])
                    mask = norm > spectral_eps
                    deltas.data[:, t][mask] = deltas.data[:, t][mask] / norm[mask] * spectral_eps
                else:
                    # Spatial projection: per-sample relative L-inf clamping
                    deltas.data[:, t] = torch.clamp(deltas.data[:, t], -eps_bounds, eps_bounds)
    
    # Reconstruct full adversarial trajectory
    with torch.no_grad():
        curr_x = u0_clean.clone()
        all_preds = [curr_x]
        all_spatial_deltas = []
        
        for t in range(future_steps):
            if mode == 'spectral':
                curr_fft = torch.fft.rfft2(curr_x, dim=(-2, -1))
                if spec_mask is not None:
                    delta_masked = deltas[:, t] * spec_mask.unsqueeze(0).unsqueeze(0)
                else:
                    delta_masked = deltas[:, t]
                curr_x_pert = torch.fft.irfft2(curr_fft + delta_masked, s=(H, W), dim=(-2, -1))
                delta_spatial = curr_x_pert - curr_x
            else:
                delta_t = deltas[:, t]
                curr_x_pert = curr_x + delta_t
                delta_spatial = delta_t
            
            all_spatial_deltas.append(delta_spatial.unsqueeze(1))
            pred = model(curr_x_pert)
            all_preds.append(pred)
            curr_x = torch.cat([curr_x[:, 1:], pred], dim=1)
        
        u_adv = torch.cat(all_preds, dim=1)
        spatial_deltas = torch.cat(all_spatial_deltas, dim=1)
    
    return u_adv.cpu(), spatial_deltas.cpu()

# ==========================================
# 5. MAIN
# ==========================================

def ensure_list(val):
    if isinstance(val, list): return val
    return [val]

def ensure_zipped_list(a, b):
    a_list = ensure_list(a)
    b_list = ensure_list(b)
    if len(a_list) == 1: a_list = a_list * len(b_list)
    if len(b_list) == 1: b_list = b_list * len(a_list)
    return list(zip(a_list, b_list))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.attack_config, "rb") as f:
        cfg = tomllib.load(f)
    
    # Flatten config
    cfg = {**cfg.get('general', {}), **cfg.get('model', {}), **cfg}
    
    # Load model
    with open(cfg['model_config_path'], "rb") as f:
        model_conf = tomllib.load(f)["config"]
    
    model = create_model(model_conf, DEVICE)
    model.load_state_dict(torch.load(cfg['model_path'], map_location=DEVICE)['model_state_dict'], strict=False)
    model.eval()
    
    # Load data
    with open(cfg['data_config_path'], "rb") as f:
        data_conf = tomllib.load(f)["config"]
    
    problem = NavierStokes2D(data_conf)
    u_tensor = problem.load_data()
    cfg['visc'] = problem.visc
    cfg['dt'] = problem.dt
    
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    n_samples = cfg['n_samples']
    u_clean = u_tensor[:n_samples].to(DEVICE)
    in_channels = model_conf['in_channels']
    u0_clean = u_clean[:, :in_channels]
    
    def save_results(u_adv, deltas, filename, attack_params=None):
        """Save results with attack parameters as attributes."""
        out_path = os.path.join(cfg['output_dir'], filename)
        
        # Debug: Print what we're about to save
        print(f"    [DEBUG] Saving: u_adv shape={u_adv.shape}, deltas shape={deltas.shape if deltas is not None else None}")
        print(f"    [DEBUG] u_adv range: [{u_adv.min():.4f}, {u_adv.max():.4f}]")
        print(f"    [DEBUG] u0_clean range: [{u0_clean.min().item():.4f}, {u0_clean.max().item():.4f}]")
        if deltas is not None:
            print(f"    [DEBUG] deltas range: [{deltas.min():.4f}, {deltas.max():.4f}]")
        
        with h5py.File(out_path, 'w') as f:
            if u_adv.shape[1] > u0_clean.shape[1]:  # Sequential
                f.create_dataset('adversarial/u', data=u_adv.permute(0, 2, 3, 1).cpu().numpy())
                print(f"    [DEBUG] Saved as adversarial/u (sequential)")
            else:
                f.create_dataset('adversarial/x_init', data=u_adv.permute(0, 2, 3, 1).cpu().numpy())
                print(f"    [DEBUG] Saved as adversarial/x_init (IC-only)")
            f.create_dataset('train/u', data=u_clean.permute(0, 2, 3, 1).cpu().numpy())
            if deltas is not None:
                if deltas.ndim == 5:  # Sequential: (B, T, 1, H, W)
                    f.create_dataset('adversarial/deltas', data=deltas.permute(0, 1, 3, 4, 2).cpu().numpy())
                else:
                    f.create_dataset('adversarial/deltas', data=deltas.permute(0, 2, 3, 1).cpu().numpy())
            
            # Save physics params
            f.attrs['viscosity'] = cfg['visc']
            f.attrs['dt'] = cfg['dt']
            
            # Save attack parameters
            if attack_params:
                for key, val in attack_params.items():
                    if val is not None:
                        f.attrs[key] = val
        
        print(f"Saved: {out_path}")

    # --- BASELINES (Relative L2) ---
    if cfg['active_attacks'].get('spectral_noise', False):
        for rel_l2 in ensure_list(cfg["pgd"].get("rel_l2", cfg["pgd"].get("target_l2", 0.05))):
            print(f"\n>>> EXECUTE: Spectral Noise Baseline (rel_l2={rel_l2:.2%})")
            u_res = run_spectral_noise(u0_clean, cfg, rel_l2)
            attack_params = {
                'attack_type': 'noise',
                'attack_domain': 'spectral',
                'rel_l2': rel_l2,
                'rel_eps': None,
                'lambda_pde': 0.0,
                'lambda_bc': 0.0,
            }
            save_results(u_res, u_res - u0_clean.cpu(), f'attack_noise_spectral_l2_{rel_l2}.h5', attack_params)

    if cfg['active_attacks'].get('spatial_noise', False):
        for rel_l2 in ensure_list(cfg["pgd"].get("rel_l2", cfg["pgd"].get("target_l2", 0.05))):
            print(f"\n>>> EXECUTE: Spatial Noise Baseline (rel_l2={rel_l2:.2%})")
            u_res = run_spatial_noise(u0_clean, cfg, rel_l2)
            attack_params = {
                'attack_type': 'noise',
                'attack_domain': 'spatial',
                'rel_l2': rel_l2,
                'rel_eps': None,
                'lambda_pde': 0.0,
                'lambda_bc': 0.0,
            }
            save_results(u_res, u_res - u0_clean.cpu(), f'attack_noise_spatial_l2_{rel_l2}.h5', attack_params)

    # 2. PGD ATTACKS (Relative L-inf epsilon)
    pgd_types = [
        ('spatial_pure', 'spatial', 'rel_eps_spatial', 'epsilon_spatial'),
        ('spectral_pure', 'spectral', 'rel_eps_spectral', 'epsilon_spectral'),
    ]
    for key, mode, rel_eps_key, old_eps_key in pgd_types:
        if cfg['active_attacks'].get(key, False):
            # Support both new (rel_eps_*) and old (epsilon_*) config keys
            eps_values = cfg['pgd'].get(rel_eps_key, cfg['pgd'].get(old_eps_key, 0.01))
            for rel_eps in ensure_list(eps_values):
                print(f"\n>>> EXECUTE: PGD {key} (rel_eps={rel_eps:.2%})")
                u_adv = run_pgd_attack(model, u0_clean, u_clean, cfg, rel_eps=rel_eps, mode=mode)
                deltas = u_adv - u0_clean.cpu()
                
                # Debug: verify deltas are non-zero
                delta_max = deltas.abs().max().item()
                delta_mean = deltas.abs().mean().item()
                print(f"    [DEBUG] Computed deltas: max={delta_max:.6f}, mean={delta_mean:.6f}")
                
                attack_params = {
                    'attack_type': 'pgd_pure',
                    'attack_domain': mode,
                    'rel_eps': rel_eps,
                    'rel_l2': None,
                    'lambda_pde': 0.0,
                    'lambda_bc': 0.0,
                }
                save_results(u_adv, deltas, f'attack_{key}_eps{rel_eps}.h5', attack_params)

    # 3. STEALTH ATTACKS (PGD with physics constraints)
    stealth_types = [
        ('spatial_stealth', 'spatial', 'rel_eps_spatial', 'epsilon_spatial'),
        ('spectral_stealth', 'spectral', 'rel_eps_spectral', 'epsilon_spectral')
    ]
    for key, mode, rel_eps_key, old_eps_key in stealth_types:
        if cfg['active_attacks'].get(key, False):
            eps_values = cfg['pgd'].get(rel_eps_key, cfg['pgd'].get(old_eps_key, 0.01))
            for rel_eps in ensure_list(eps_values):
                for lambda_pde, lambda_bc in ensure_zipped_list(cfg['physics']['lambda_pde'], cfg['physics']['lambda_bc']):
                    print(f"\n>>> EXECUTE: Stealth {key} (rel_eps={rel_eps:.2%}, pde={lambda_pde}, bc={lambda_bc})")
                    u_adv = run_pgd_attack(model, u0_clean, u_clean, cfg, mode=mode, rel_eps=rel_eps, lambda_pde=lambda_pde, lambda_bc=lambda_bc)
                    deltas = u_adv - u0_clean.cpu()
                    attack_params = {
                        'attack_type': 'pgd_stealth',
                        'attack_domain': mode,
                        'rel_eps': rel_eps,
                        'rel_l2': None,
                        'lambda_pde': lambda_pde,
                        'lambda_bc': lambda_bc,
                    }
                    save_results(u_adv, deltas, f'attack_{key}_eps{rel_eps}_pde{lambda_pde}_bc{lambda_bc}.h5', attack_params)

    # 4. SEQUENTIAL ATTACKS
    seq_types = [
        ('sequential_spatial', 'spatial', 'rel_eps_spatial', 'epsilon_spatial'), 
        ('sequential_spectral', 'spectral', 'rel_eps_spectral', 'epsilon_spectral')
    ]
    for key, mode, rel_eps_key, old_eps_key in seq_types:
        if cfg['active_attacks'].get(key, False):
            eps_values = cfg['pgd'].get(rel_eps_key, cfg['pgd'].get(old_eps_key, 0.01))
            for rel_eps in ensure_list(eps_values):
                for lambda_pde, lambda_bc in ensure_zipped_list(cfg['physics']['lambda_pde'], cfg['physics']['lambda_bc']):
                    print(f"\n>>> EXECUTE: Sequential {key} (rel_eps={rel_eps:.2%}, pde={lambda_pde}, bc={lambda_bc})")
                    u_adv, deltas = run_sequential_pgd_attack(model, u0_clean, u_clean, cfg, mode=mode, rel_eps=rel_eps, lambda_pde=lambda_pde, lambda_bc=lambda_bc)
                    attack_params = {
                        'attack_type': 'sequential',
                        'attack_domain': mode,
                        'rel_eps': rel_eps,
                        'rel_l2': None,
                        'lambda_pde': lambda_pde,
                        'lambda_bc': lambda_bc,
                    }
                    save_results(u_adv, deltas.detach().squeeze(2).cpu(), f'attack_{key}_eps{rel_eps}_pde{lambda_pde}_bc{lambda_bc}.h5', attack_params)

    print("\nDone.")

if __name__ == "__main__":
    main()