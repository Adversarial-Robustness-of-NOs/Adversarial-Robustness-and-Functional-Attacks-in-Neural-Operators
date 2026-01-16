import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import tomllib
import h5py
import json
from tqdm import tqdm
from pathlib import Path

# Adjust imports to match your folder structure
from temporal.ns_2d.ns import NavierStokes2D
from models.model_factory import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. UTILS
# ==========================================

def calculate_l2_metrics(pred, true):
    if pred.shape != true.shape:
        pred = F.interpolate(pred, size=true.shape[-2:], mode='bicubic', align_corners=False)
    diff_norm = torch.norm(pred - true)
    true_norm = torch.norm(true)
    return diff_norm.item(), (diff_norm / (true_norm + 1e-8)).item()

def calculate_spacetime_l2(pred, true):
    if pred.shape != true.shape:
        pred = F.interpolate(pred, size=true.shape[-2:], mode='bicubic', align_corners=False)
    diff_norm = torch.norm(pred - true)
    true_norm = torch.norm(true)
    return (diff_norm / (true_norm + 1e-8)).item()

def calculate_spacetime_mse(pred, true):
    if pred.shape != true.shape:
        pred = F.interpolate(pred, size=true.shape[-2:], mode='bicubic', align_corners=False)
    return torch.mean((pred - true)**2).item()

def calculate_per_step_l2(pred, true):
    """Calculate relative L2 error at each time step."""
    # pred, true: (B, T, H, W) or (B, T, C, H, W)
    if pred.shape != true.shape:
        pred = F.interpolate(pred.view(-1, 1, pred.shape[-2], pred.shape[-1]), 
                            size=true.shape[-2:], mode='bicubic', align_corners=False)
        pred = pred.view(true.shape)
    
    T = pred.shape[1]
    errors = []
    for t in range(T):
        diff_norm = torch.norm(pred[:, t] - true[:, t])
        true_norm = torch.norm(true[:, t]) + 1e-8
        errors.append((diff_norm / true_norm).item())
    return errors

def calculate_frequency_error(pred, true, cutoff_radius=12.0):
    if pred.shape != true.shape:
        pred = F.interpolate(pred, size=true.shape[-2:], mode='bicubic', align_corners=False)
    
    diff = pred - true
    diff_ft = torch.fft.rfft2(diff)
    H, W = diff.shape[-2], diff.shape[-1]
    
    kx = torch.fft.fftfreq(H, d=1.0).to(pred.device) * H
    ky = torch.fft.rfftfreq(W, d=1.0).to(pred.device) * W
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    R = torch.sqrt(KX**2 + KY**2)
    
    low_mask = (R <= cutoff_radius).float()
    high_mask = (R > cutoff_radius).float()
    
    low_freq_error = torch.norm(diff_ft * low_mask).item()
    high_freq_error = torch.norm(diff_ft * high_mask).item()
    return low_freq_error, high_freq_error

def compute_physics_residual_tensor(problem, w_curr, w_prev, dt):
    if w_curr.ndim == 3: w_curr = w_curr.unsqueeze(1)
    if w_prev.ndim == 3: w_prev = w_prev.unsqueeze(1)
    w_t = (w_curr - w_prev) / dt
    u, v, w_x, w_y, lap_w = problem.compute_spectral_physics(w_curr)
    advection = (u * w_x) + (v * w_y)
    diffusion = problem.visc * lap_w
    residual = w_t + advection - diffusion
    return residual


def get_log_spectrum(img_2d):
    f = np.fft.fft2(img_2d)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude


# ==========================================
# 1b. SPECTRAL ANALYSIS (from Darcy)
# ==========================================

def get_radial_spectrum(batch_img):
    """Computes Radial Average Power Spectrum (RAPS). Returns (B, max_r)."""
    # Handle different input shapes
    if isinstance(batch_img, np.ndarray):
        batch_img = torch.tensor(batch_img, dtype=torch.float32)
    
    if batch_img.ndim == 2:
        batch_img = batch_img.unsqueeze(0)  # Add batch dim
    if batch_img.ndim == 4:
        batch_img = batch_img.squeeze(1)  # Remove channel dim
    
    # Move to CPU and ensure float dtype for all operations
    batch_img = batch_img.detach().cpu().float()
    
    B, H, W = batch_img.shape
    f = torch.fft.fft2(batch_img)
    fshift = torch.fft.fftshift(f, dim=(-2, -1))
    magnitude = torch.abs(fshift) ** 2 
    
    # Create coordinate grids on CPU
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    r = torch.sqrt((x.float() - center_x)**2 + (y.float() - center_y)**2).long()
    max_r = min(H, W) // 2
    
    spectra = []
    for b in range(B):
        mag = magnitude[b].float()  # Ensure float
        t_r, t_mag = r.flatten(), mag.flatten()
        mask = t_r < max_r
        t_r, t_mag = t_r[mask], t_mag[mask]
        energy_sums = torch.bincount(t_r, weights=t_mag)
        pixel_counts = torch.bincount(t_r).float()
        spectra.append((energy_sums / (pixel_counts + 1e-8)).numpy())
    return np.array(spectra)


def analyze_bands(spectrum, split_ratio=0.25):
    """Splits spectrum into Low (0-split_ratio%) and High (split_ratio-100%) frequency magnitudes."""
    k_max = len(spectrum)
    split_idx = int(k_max * split_ratio)
    low_mag = np.sum(spectrum[:split_idx])
    high_mag = np.sum(spectrum[split_idx:])
    return low_mag, high_mag


def plot_spectral_comparison_ns(input_pert, output_pert, save_path, time_idx=None):
    """
    Plot spectral comparison between input perturbation and output perturbation.
    Similar to Darcy's spectral plot.
    
    Args:
        input_pert: Input perturbation (adv IC - clean IC), shape (H, W) or (T, H, W)
        output_pert: Output perturbation (adv pred - clean pred), shape (H, W) or (T, H, W)
        save_path: Path to save the figure
        time_idx: Optional time index label for multi-step data
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Handle time-series data - use last timestep or specific index
    if input_pert.ndim == 3:
        input_pert = input_pert[-1]  # Use last timestep of IC
    if output_pert.ndim == 3:
        output_pert = output_pert[-1]  # Use last timestep
    
    # Compute radial spectra
    input_spectrum = get_radial_spectrum(input_pert)[0]
    output_spectrum = get_radial_spectrum(output_pert)[0]
    
    # Frequency axis
    k = np.arange(len(input_spectrum))
    
    # Plot both spectra
    ax.semilogy(k, input_spectrum + 1e-10, 'b-', linewidth=2, label='Input Perturbation (IC)')
    ax.semilogy(k, output_spectrum + 1e-10, 'r-', linewidth=2, label='Output Perturbation')
    
    # Mark low/high frequency boundary (25%)
    boundary = int(len(k) * 0.25)
    ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7, label='Low/High Boundary')
    
    # Shade regions
    ax.axvspan(0, boundary, alpha=0.1, color='blue')
    ax.axvspan(boundary, len(k), alpha=0.1, color='red')
    
    # Compute energy ratios for annotation
    in_low, in_high = analyze_bands(input_spectrum)
    out_low, out_high = analyze_bands(output_spectrum)
    in_total = in_low + in_high + 1e-10
    out_total = out_low + out_high + 1e-10
    
    in_low_pct = in_low / in_total * 100
    in_high_pct = in_high / in_total * 100
    out_low_pct = out_low / out_total * 100
    out_high_pct = out_high / out_total * 100
    
    ax.set_xlabel('Frequency (k)', fontsize=12)
    ax.set_ylabel('Magnitude (log)', fontsize=12)
    
    title = f'Spectral Comparison\nInput: {in_low_pct:.0f}%L/{in_high_pct:.0f}%H → Output: {out_low_pct:.0f}%L/{out_high_pct:.0f}%H'
    if time_idx is not None:
        title = f't={time_idx}: ' + title
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, len(k)-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return {
        'input_low_pct': in_low_pct,
        'input_high_pct': in_high_pct,
        'output_low_pct': out_low_pct,
        'output_high_pct': out_high_pct,
        'input_spectrum': input_spectrum,
        'output_spectrum': output_spectrum
    }


def plot_global_curves(clean_rel, adv_rel, div_rel, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(clean_rel, marker='o', markersize=4, label='Baseline Error (Rel)', color='blue', alpha=0.7)
    if adv_rel is not None:
        plt.plot(adv_rel, marker='x', markersize=4, label='Attack Error (Rel)', color='red', linewidth=2)
    if div_rel is not None:
        plt.plot(div_rel, marker='s', markersize=4, label='Trajectory Divergence (Rel)', color='purple', linestyle='--', alpha=0.7)
    
    plt.title("Global Average Error (Relative)")
    plt.xlabel("Time Step")
    plt.ylabel("Relative L2 Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=100)
    plt.close()

def plot_dynamics_sample(clean_err_rel, adv_err_rel, clean_phys, adv_phys, save_path, drift_rel=None):
    """
    Plot dynamics for a single sample.
    
    Args:
        clean_err_rel: Per-step clean prediction error vs ground truth
        adv_err_rel: Per-step adversarial prediction error vs ground truth (or drift if legacy)
        clean_phys: Per-step clean physics residual
        adv_phys: Per-step adversarial physics residual
        save_path: Output path
        drift_rel: Optional drift (adv vs clean) - for backward compatibility
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Panel 1: Trajectory Errors vs Ground Truth
    ax1.plot(clean_err_rel, marker='.', label='Clean Error (vs GT)', color='blue')
    if adv_err_rel is not None:
        ax1.plot(adv_err_rel, marker='x', label='Adversarial Error (vs GT)', color='red', linewidth=2)
    if drift_rel is not None:
        ax1.plot(drift_rel, marker='s', label='Drift (Adv - Clean)', color='purple', linewidth=1, alpha=0.7)
    ax1.set_ylabel("Relative L2 Error")
    ax1.set_title("Prediction Error vs Ground Truth")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Physics Residual
    ax2.plot(clean_phys, label='Clean Physics', color='blue', linestyle='--')
    if adv_phys is not None:
        ax2.plot(adv_phys, label='Adv Physics', color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel("PDE Residual")
    ax2.set_xlabel("Time Step")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_snapshots_comparison(u_true, u_clean, u_adv, phys_map, time_indices, save_path, 
                               pert_info=None, x_clean_ic=None, x_adv_ic=None):
    """
    Plot snapshot comparison with optional spectral analysis column.
    
    Args:
        u_true: Ground truth trajectory (T, H, W)
        u_clean: Clean prediction trajectory (T, H, W)
        u_adv: Adversarial prediction trajectory (T, H, W)
        phys_map: Physics residual maps for each time index
        time_indices: List of time indices to plot
        save_path: Output path
        pert_info: Optional perturbation info string
        x_clean_ic: Clean initial condition (for spectral comparison)
        x_adv_ic: Adversarial initial condition (for spectral comparison)
    """
    # Determine if we have data for spectral comparison
    has_spectral = (x_clean_ic is not None and x_adv_ic is not None and u_adv is not None)
    
    if has_spectral:
        cols = ['Ground Truth', 'Baseline Pred', 'Adversarial Pred', 'Adv - Baseline', 
                'Physics Residual', 'Error Spectrum', 'Spectral Comparison']
        n_cols = 7
    else:
        cols = ['Ground Truth', 'Baseline Pred', 'Adversarial Pred', 'Adv - Baseline', 
                'Physics Residual', 'Error Spectrum']
        n_cols = 6
    
    n_rows = len(time_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    if pert_info: 
        fig.suptitle(pert_info, fontsize=20, fontweight='bold', y=0.98)
    
    vmin, vmax = u_true.min(), u_true.max()
    
    # Compute input perturbation once (for spectral comparison)
    if has_spectral:
        input_pert = x_adv_ic - x_clean_ic  # IC perturbation
        input_spectrum = get_radial_spectrum(input_pert)[0]
        in_low, in_high = analyze_bands(input_spectrum)
        in_total = in_low + in_high + 1e-10
    
    for row_idx, t in enumerate(time_indices):
        if t >= u_true.shape[0]: 
            break
        
        output_diff = u_adv[t] - u_clean[t] if u_adv is not None else np.zeros_like(u_true[t])
        
        row_data = [
            u_true[t], 
            u_clean[t], 
            u_adv[t] if u_adv is not None else np.zeros_like(u_true[t]),
            np.abs(output_diff),
            phys_map[row_idx] if row_idx < len(phys_map) else np.zeros_like(u_true[t]),
            get_log_spectrum(output_diff)
        ]
        
        cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'magma', 'twilight', 'inferno']
        
        for i, (data, cmap) in enumerate(zip(row_data, cmaps)):
            ax = axes[row_idx, i]
            kw = {'vmin': vmin, 'vmax': vmax} if i < 3 else {}
            im = ax.imshow(data, cmap=cmap, origin='lower', **kw)
            
            if row_idx == 0: 
                ax.set_title(cols[i], fontsize=12)
            if i == 0: 
                ax.set_ylabel(f"t={t}", fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row_idx != n_rows - 1: 
                ax.set_xticks([])
            ax.set_yticks([])
        
        # Add spectral comparison plot in the last column
        if has_spectral:
            ax_spec = axes[row_idx, n_cols - 1]
            
            # Compute output perturbation spectrum at this timestep
            output_pert = u_adv[t] - u_clean[t]
            output_spectrum = get_radial_spectrum(output_pert)[0]
            
            out_low, out_high = analyze_bands(output_spectrum)
            out_total = out_low + out_high + 1e-10
            
            # Frequency axis
            k = np.arange(len(input_spectrum))
            
            # Plot both spectra
            ax_spec.semilogy(k, input_spectrum + 1e-10, 'b-', linewidth=2, label='Input Pert')
            ax_spec.semilogy(k, output_spectrum + 1e-10, 'r-', linewidth=2, label='Output Pert')
            
            # Mark low/high frequency boundary (25%)
            boundary = int(len(k) * 0.25)
            ax_spec.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)
            
            # Shade regions
            ax_spec.axvspan(0, boundary, alpha=0.1, color='blue')
            ax_spec.axvspan(boundary, len(k), alpha=0.1, color='red')
            
            # Compute percentages
            in_low_pct = in_low / in_total * 100
            in_high_pct = in_high / in_total * 100
            out_low_pct = out_low / out_total * 100
            out_high_pct = out_high / out_total * 100
            
            ax_spec.set_xlabel('k', fontsize=9)
            if row_idx == 0:
                ax_spec.set_title('Spectral Comparison', fontsize=12)
            
            # Compact annotation
            ax_spec.text(0.02, 0.98, f'In: {in_low_pct:.0f}%L/{in_high_pct:.0f}%H\nOut: {out_low_pct:.0f}%L/{out_high_pct:.0f}%H',
                        transform=ax_spec.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax_spec.legend(loc='upper right', fontsize=7)
            ax_spec.grid(True, alpha=0.3)
            ax_spec.set_xlim([0, len(k)-1])

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_spectral_evolution(u_clean, u_adv, save_path):
    if u_adv.shape != u_clean.shape:
        u_adv_t = torch.tensor(u_adv).unsqueeze(0).unsqueeze(0)
        u_adv_t = F.interpolate(u_adv_t, size=u_clean.shape, mode='bicubic', align_corners=False)
        u_adv = u_adv_t.squeeze().numpy()

    diff = u_adv - u_clean
    T, H, W = diff.shape
    k_max = H // 2
    evolution_matrix = np.zeros((T, k_max))
    
    kx = np.fft.fftfreq(H, d=1.0) * H
    ky = np.fft.fftfreq(W, d=1.0) * W
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    R = np.sqrt(KX**2 + KY**2)
    
    for t in range(T):
        diff_ft = np.fft.fft2(diff[t])
        energy_spectrum = np.abs(diff_ft)**2
        for k in range(k_max):
            mask = (R >= k) & (R < k+1)
            if mask.sum() > 0:
                evolution_matrix[t, k] = energy_spectrum[mask].mean()
                
    plt.figure(figsize=(10, 6))
    log_matrix = np.log10(evolution_matrix.T + 1e-12)
    plt.imshow(log_matrix, aspect='auto', origin='lower', cmap='inferno', extent=[0, T, 0, k_max])
    plt.colorbar(label='Log Error Energy')
    plt.xlabel('Time Step')
    plt.ylabel('Wavenumber k')
    plt.title('Evidence of Inverse Energy Cascade')
    plt.axhline(y=6, color='white', linestyle='--', alpha=0.5, label='Cutoff k=6')
    plt.savefig(save_path, dpi=100)
    plt.close()

# ==========================================
# 2. DATA LOADING
# ==========================================

def load_data_file(path, n_samples):
    print(f"Loading Reference {path}...")
    with h5py.File(path, 'r') as f:
        if 'train/u' in f: u = f['train/u'][:]
        elif 'u' in f: u = f['u'][:]
        else: raise ValueError(f"Could not find 'u' in {path}")
        
        visc = f.attrs.get('viscosity', None)
        dt_val = None
        if 't-coordinate' in f:
            t_coord = f['t-coordinate'][:]
            if len(t_coord) > 1: dt_val = t_coord[1] - t_coord[0]
            
    t_u = torch.tensor(u, dtype=torch.float32)
    if t_u.ndim == 4 and t_u.shape[-1] != t_u.shape[-2]: 
         if t_u.shape[1] == t_u.shape[2]: t_u = t_u.permute(0, 3, 1, 2)
             
    start = max(0, t_u.shape[0] - n_samples)
    return t_u[start:], visc, dt_val

def load_attack_file(path, n_samples):
    print(f"Loading Attack {path}...")
    with h5py.File(path, 'r') as f:
        u_adv = None
        deltas = None
        
        # Load Deltas (Critical for Sequential)
        if 'adversarial/deltas' in f: deltas = f['adversarial/deltas'][:]
        
        if 'adversarial/u' in f: u_adv = f['adversarial/u'][:]
        elif 'adversarial/x_init' in f: u_adv = f['adversarial/x_init'][:]
        
        u_clean = None
        if 'train/u' in f: u_clean = f['train/u'][:]
        elif 'u' in f: u_clean = f['u'][:]
        
        visc = f.attrs.get('viscosity', None)
        dt_val = None

    def to_tensor(arr):
        if arr is None: return None
        t = torch.tensor(arr, dtype=torch.float32)
        if t.ndim == 4 and t.shape[-1] != t.shape[-2]: 
             if t.shape[1] == t.shape[2]: t = t.permute(0, 3, 1, 2)
        return t

    t_adv = to_tensor(u_adv)
    t_clean = to_tensor(u_clean)
    
    t_deltas = None
    if deltas is not None:
        t_deltas = torch.tensor(deltas, dtype=torch.float32)
        if t_deltas.ndim == 4 and t_clean is not None and t_deltas.shape[1] == t_clean.shape[1]:
             pass
        elif t_deltas.ndim == 4 and t_deltas.shape[-1] != t_deltas.shape[-2]:
             if t_deltas.shape[1] == t_deltas.shape[2]: t_deltas = t_deltas.permute(0, 3, 1, 2)

    total = t_adv.shape[0] if t_adv is not None else (t_clean.shape[0] if t_clean is not None else 0)
    start = max(0, total - n_samples)
    
    if t_adv is not None: t_adv = t_adv[start:]
    if t_clean is not None: t_clean = t_clean[start:]
    if t_deltas is not None: t_deltas = t_deltas[start:]
    
    return t_clean, t_adv, t_deltas, visc, dt_val

# ==========================================
# 4. MAIN EVALUATION
# ==========================================

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="temporal/ns_2d/fno_model.toml")
    parser.add_argument("--data_config", type=str, default="temporal/ns_2d/attack_data.toml")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--data_path", type=str, required=True, help="Path to Attack H5 (Source)")
    parser.add_argument("--ref_data_path", type=str, default=None, 
                        help="Optional: Path to Ground Truth H5 (Target Res).")
    parser.add_argument("--output_dir", type=str, default="temporal/ns_2d/benchmark_results")
    parser.add_argument("--num_test_samples", type=int, default=100)
    parser.add_argument("--num_samples_to_plot", type=int, default=10)
    args = parser.parse_args()
    
    with open(args.model_config, "rb") as f: model_conf = tomllib.load(f)["config"]
    with open(args.data_config, "rb") as f: data_conf_toml = tomllib.load(f)["config"]
    
    # 1. Load Model
    model = create_model(model_conf, DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE)['model_state_dict'], strict=False)
    model.eval()

    in_channels = model_conf['in_channels']

    # 2. Load Data
    t_clean_src, t_adv_src, t_deltas, visc, dt_val = load_attack_file(args.data_path, args.num_test_samples)
    
    if args.ref_data_path:
        t_true, visc_ref, dt_ref = load_data_file(args.ref_data_path, args.num_test_samples)
        if t_true.shape[0] > t_clean_src.shape[0]:
            t_true = t_true[:t_clean_src.shape[0]]
    else:
        t_true = t_clean_src

    # Instantiate Problem for Physics Calc
    data_conf_toml['viscosity'] = visc if visc else data_conf_toml.get('viscosity', 1e-3)
    data_conf_toml['dt'] = dt_val if dt_val else data_conf_toml.get('dt', 1.0)
    problem = NavierStokes2D(data_conf_toml)

    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Sizes
    n_eval = min(t_clean_src.shape[0], t_true.shape[0])
    src_res = t_clean_src.shape[-1]
    tgt_res = t_true.shape[-1]
    total_time = t_clean_src.shape[1]
    future_steps = total_time - in_channels

    # 4. Metric Collection
    metrics = {
        'ic_pert_abs': [], 'ic_pert_rel': [],
        'ic_low_freq': [], 'ic_high_freq': [],
        'clean_st_l2': [], 'adv_st_l2': [],
        'clean_st_mse': [], 'adv_st_mse': [],
        'clean_low_freq': [], 'clean_high_freq': [],
        'adv_low_freq': [], 'adv_high_freq': [],
        # Temporal curves
        'inj_curve': [], 'drift_curve': [],
        'clean_phys_curve': [], 'adv_phys_curve': [],
        # NEW: Per-step L2 errors vs ground truth
        'clean_error_curve': [], 'adv_error_curve': [],
        # NEW: Spectral transfer analysis
        'spectral_input_low_pct': [], 'spectral_input_high_pct': [],
        'spectral_output_low_pct': [], 'spectral_output_high_pct': [],
    }

    print(f"\nEvaluating {n_eval} samples...")
    for i in tqdm(range(n_eval)):
        # A. Load Sample
        x_clean_src = t_clean_src[i:i+1, :in_channels].to(DEVICE)
        
        # Prepare Deltas (Injection)
        sample_deltas = None
        if t_deltas is not None:
            sample_deltas = t_deltas[i:i+1].to(DEVICE)

        pert_str = ""

        if t_adv_src is not None:
            if t_adv_src.shape[1] == total_time: 
                x_adv_src = t_adv_src[i:i+1, :in_channels].to(DEVICE)
            else:
                x_adv_src = t_adv_src[i:i+1, :in_channels].to(DEVICE)
            
            ic_abs, ic_rel = calculate_l2_metrics(x_adv_src, x_clean_src)
            metrics['ic_pert_abs'].append(ic_abs)
            metrics['ic_pert_rel'].append(ic_rel)
            
            ic_lf, ic_hf = calculate_frequency_error(x_adv_src, x_clean_src, cutoff_radius=6.0)
            metrics['ic_low_freq'].append(ic_lf)
            metrics['ic_high_freq'].append(ic_hf)
            pert_str = f"Sample {i} | IC Perturbation: L2={ic_abs:.4f} ({ic_rel:.1%})"
        else:
            x_adv_src = x_clean_src
            metrics['ic_pert_abs'].append(0.0); metrics['ic_pert_rel'].append(0.0)
            metrics['ic_low_freq'].append(0.0); metrics['ic_high_freq'].append(0.0)
            pert_str = f"Sample {i} | Clean Baseline"

        y_true = t_true[i:i+1, in_channels:].to(DEVICE)

        # B. Resolution Handling
        if src_res != tgt_res:
            x_clean = F.interpolate(x_clean_src, size=(tgt_res, tgt_res), mode='bicubic', align_corners=False)
            x_adv = F.interpolate(x_adv_src, size=(tgt_res, tgt_res), mode='bicubic', align_corners=False)
            if sample_deltas is not None:
                B, T, H, W = sample_deltas.shape
                deltas_reshaped = sample_deltas.view(B*T, 1, H, W)
                deltas_resized = F.interpolate(deltas_reshaped, size=(tgt_res, tgt_res), mode='bicubic', align_corners=False)
                sample_deltas = deltas_resized.view(B, T, tgt_res, tgt_res)
        else:
            x_clean = x_clean_src
            x_adv = x_adv_src

        # C. Model Rollout
        def rollout(x_init, injections=None):
            curr = x_init
            target_res = x_init.shape[-1]
            preds = []
            with torch.no_grad():
                for t in range(future_steps):
                    if injections is not None and t < injections.shape[1]:
                        inj_t = injections[:, t].unsqueeze(1)
                        curr[:, -1:] = curr[:, -1:] + inj_t
                        
                    pred = model(curr)
                    pred = F.interpolate(pred, size=(target_res, target_res), 
                         mode='bicubic', align_corners=False)
                    preds.append(pred)
                    curr = torch.cat([curr[:, 1:], pred], dim=1)
            return torch.cat(preds, dim=1)

        y_pred_clean = rollout(x_clean)
        y_pred_adv = rollout(x_adv, injections=sample_deltas)

        # D. Global Metrics
        metrics['clean_st_l2'].append(calculate_spacetime_l2(y_pred_clean, y_true))
        metrics['adv_st_l2'].append(calculate_spacetime_l2(y_pred_adv, y_true))
        metrics['clean_st_mse'].append(calculate_spacetime_mse(y_pred_clean, y_true))
        metrics['adv_st_mse'].append(calculate_spacetime_mse(y_pred_adv, y_true))

        mid_t = future_steps // 2
        lf_c, hf_c = calculate_frequency_error(y_pred_clean[:, mid_t], y_true[:, mid_t], cutoff_radius=6.0)
        metrics['clean_low_freq'].append(lf_c)
        metrics['clean_high_freq'].append(hf_c)
        
        lf_a, hf_a = calculate_frequency_error(y_pred_adv[:, mid_t], y_true[:, mid_t], cutoff_radius=6.0)
        metrics['adv_low_freq'].append(lf_a)
        metrics['adv_high_freq'].append(hf_a)

        # E. Temporal Curves
        curr_inj, curr_drift, curr_c_phys, curr_a_phys = [], [], [], []
        curr_clean_err, curr_adv_err = [], []
        
        w_prev_c = x_clean[:, -1:]
        w_prev_a = x_adv[:, -1:]

        for t in range(future_steps):
            # Injection L2
            inj_val = 0.0
            if sample_deltas is not None and t < sample_deltas.shape[1]:
                inj_val = torch.norm(sample_deltas[:, t]).item()
            curr_inj.append(inj_val)
            
            # Drift (Adv vs Clean) - relative difference between predictions
            drift_val = torch.norm(y_pred_adv[:, t] - y_pred_clean[:, t]).item()
            clean_norm = torch.norm(y_pred_clean[:, t]).item() + 1e-8
            curr_drift.append(drift_val / clean_norm)
            
            # NEW: Per-step L2 error vs ground truth
            clean_err_t = torch.norm(y_pred_clean[:, t] - y_true[:, t]).item()
            adv_err_t = torch.norm(y_pred_adv[:, t] - y_true[:, t]).item()
            true_norm_t = torch.norm(y_true[:, t]).item() + 1e-8
            curr_clean_err.append(clean_err_t / true_norm_t)
            curr_adv_err.append(adv_err_t / true_norm_t)
            
            # Physics
            res_c = compute_physics_residual_tensor(problem, y_pred_clean[:, t], w_prev_c, problem.dt)
            res_a = compute_physics_residual_tensor(problem, y_pred_adv[:, t], w_prev_a, problem.dt)
            curr_c_phys.append(res_c.pow(2).mean().item())
            curr_a_phys.append(res_a.pow(2).mean().item())
            
            w_prev_c = y_pred_clean[:, t:t+1]
            w_prev_a = y_pred_adv[:, t:t+1]

        metrics['inj_curve'].append(curr_inj)
        metrics['drift_curve'].append(curr_drift)
        metrics['clean_phys_curve'].append(curr_c_phys)
        metrics['adv_phys_curve'].append(curr_a_phys)
        metrics['clean_error_curve'].append(curr_clean_err)
        metrics['adv_error_curve'].append(curr_adv_err)

        pert_str = ""
        if t_adv_src is not None:
            pert_str = f"IC Perturbation: {metrics['ic_pert_rel'][-1]:.2%}"
        
        # F. Plotting
        if i < args.num_samples_to_plot:
            # Use correct variable names - pass both adv error and drift
            plot_dynamics_sample(
                curr_clean_err,  # Clean error vs ground truth
                curr_adv_err if t_adv_src is not None else None,  # Adv error vs ground truth
                curr_c_phys,  # Clean physics residual
                curr_a_phys if t_adv_src is not None else None,  # Adv physics residual
                f"{args.output_dir}/sample_{i}_dynamics.png",
                drift_rel=curr_drift if t_adv_src is not None else None  # Drift (adv vs clean)
            )
            
            tru_np = y_true.squeeze().cpu().numpy()
            cln_np = y_pred_clean.squeeze().cpu().numpy()
            adv_np = y_pred_adv.squeeze().cpu().numpy() if t_adv_src is not None else None
            
            snap_indices = [t for t in [0, 5, 10, 20, future_steps-1] if t < future_steps]
            
            # Compute physics residual maps for snapshot indices
            phys_maps = []
            w_prev_plot = x_adv[:, -1:] if t_adv_src is not None else x_clean[:, -1:]
            y_pred_plot = y_pred_adv if t_adv_src is not None else y_pred_clean
            
            for snap_t in snap_indices:
                if snap_t == 0:
                    w_prev_snap = w_prev_plot
                else:
                    w_prev_snap = y_pred_plot[:, snap_t-1:snap_t]
                
                res = compute_physics_residual_tensor(problem, y_pred_plot[:, snap_t], w_prev_snap, problem.dt)
                phys_maps.append(res.squeeze().cpu().numpy())
            
            # Get IC data for spectral comparison
            x_clean_ic_np = x_clean.squeeze().cpu().numpy()
            x_adv_ic_np = x_adv.squeeze().cpu().numpy() if t_adv_src is not None else None
            
            plot_snapshots_comparison(
                tru_np, cln_np, adv_np, phys_maps, snap_indices, 
                f"{args.output_dir}/sample_{i}_snapshots.png", 
                pert_info=pert_str,
                x_clean_ic=x_clean_ic_np,
                x_adv_ic=x_adv_ic_np
            )

            # Only plot spectral evolution if we have adversarial data
            if t_adv_src is not None:
                plot_spectral_evolution(y_pred_clean.cpu().numpy()[0], 
                                        y_pred_adv.cpu().numpy()[0], 
                                        f"{args.output_dir}/sample_{i}_spectral_evolution.png")
                
                # NEW: Standalone spectral comparison plot (input pert vs output pert)
                # Use the IC perturbation and the final output perturbation
                input_pert = (x_adv - x_clean).squeeze().cpu().numpy()
                output_pert = (y_pred_adv - y_pred_clean).squeeze().cpu().numpy()
                
                spec_info = plot_spectral_comparison_ns(
                    input_pert, 
                    output_pert,
                    f"{args.output_dir}/sample_{i}_spectral_transfer.png",
                    time_idx=future_steps - 1
                )
                
                # Store spectral transfer info
                metrics['spectral_input_low_pct'].append(spec_info['input_low_pct'])
                metrics['spectral_input_high_pct'].append(spec_info['input_high_pct'])
                metrics['spectral_output_low_pct'].append(spec_info['output_low_pct'])
                metrics['spectral_output_high_pct'].append(spec_info['output_high_pct'])

    # ==========================================
    # 5. REPORT GENERATION
    # ==========================================
    
    avg_clean_l2 = np.mean(metrics['clean_st_l2'])
    avg_adv_l2 = np.mean(metrics['adv_st_l2'])
    avg_clean_mse = np.mean(metrics['clean_st_mse'])
    avg_adv_mse = np.mean(metrics['adv_st_mse'])
    
    # Temporal Averages
    avg_inj = np.mean(metrics['inj_curve'], axis=0)
    avg_drift = np.mean(metrics['drift_curve'], axis=0)
    avg_c_phys = np.mean(metrics['clean_phys_curve'], axis=0)
    avg_a_phys = np.mean(metrics['adv_phys_curve'], axis=0)
    avg_clean_err = np.mean(metrics['clean_error_curve'], axis=0)
    avg_adv_err = np.mean(metrics['adv_error_curve'], axis=0)

    lines = []
    lines.append("================================================================================")
    lines.append("                        NEURAL OPERATOR ROBUSTNESS REPORT                       ")
    lines.append("================================================================================")
    lines.append(f"Data Source:       {args.data_path}")
    lines.append(f"Transferability:   {src_res}x  -->  {tgt_res}x")
    lines.append(f"Number of Samples: {len(t_true)}")
    lines.append("")
    
    lines.append("[1] Initial Condition Perturbation (t=0, Source Res)")
    lines.append("-" * 96)
    if t_adv_src is not None:
        lines.append(f"  > Absolute L2 Norm: {np.mean(metrics['ic_pert_abs']):.6f}")
        lines.append(f"  > Relative L2 Norm: {np.mean(metrics['ic_pert_rel']):.2%} (of clean IC)")
        
        avg_ic_low = np.mean(metrics['ic_low_freq'])
        avg_ic_high = np.mean(metrics['ic_high_freq'])
        total_ic_energy = avg_ic_low + avg_ic_high
        if total_ic_energy > 1e-9:
             low_pct = avg_ic_low / total_ic_energy
             high_pct = avg_ic_high / total_ic_energy
             lines.append(f"  > Input Spectrum:   Low Freq (k<=6): {avg_ic_low:.4f} ({low_pct:.1%}) | High Freq (k>6): {avg_ic_high:.4f} ({high_pct:.1%})")
    else:
        lines.append("  > Clean Data (No Attack)")
    lines.append("")

    lines.append("[2] Global Performance (Space-Time Average)")
    lines.append("-" * 96)
    lines.append(f"{'Metric':<25} {'Baseline (Clean)':<25} {'Adversarial':<25} {'Delta (Ratio)':<20}")
    lines.append("-" * 96)

    def add_row_with_ratio(metric_name, c_val, a_val):
        if t_adv_src is not None:
            delta = a_val - c_val
            ratio = a_val / c_val if c_val > 1e-9 else 0.0
            change_str = f"{delta:+.2e} ({ratio:.2f}x)"
            lines.append(f"{metric_name:<25} {c_val:.6f}{' '*19} {a_val:.6f}{' '*19} {change_str:<20}")
        else:
            lines.append(f"{metric_name:<25} {c_val:.6f}{' '*19} {'N/A':<25} {'N/A':<20}")

    add_row_with_ratio("Relative L2 Error", avg_clean_l2, avg_adv_l2)
    add_row_with_ratio("Mean Squared Error", avg_clean_mse, avg_adv_mse)
    lines.append("")

    lines.append("[3] Temporal Evolution (Per-Step L2 Error vs Ground Truth)")
    lines.append("-" * 96)
    lines.append(f"{'Time':<6} | {'Clean Err %':<12} {'Adv Err %':<12} {'Ratio':<10} | {'Clean Phys':<12} {'Adv Phys':<12}")
    lines.append("-" * 96)
    
    T_len = len(avg_clean_err)
    indices = [0] + [int(T_len*p) for p in [0.25, 0.5, 0.75]] + [T_len-1]
    indices = sorted(list(set([i for i in indices if i < T_len and i >= 0])))
    
    for t in indices:
        c_err = avg_clean_err[t]
        a_err = avg_adv_err[t]
        ratio = a_err / c_err if c_err > 1e-9 else 0.0
        c_p = avg_c_phys[t]
        a_p = avg_a_phys[t]
        lines.append(f"T={t:<4} | {c_err:.2%}{' '*6} {a_err:.2%}{' '*6} {ratio:.2f}x{' '*4} | {c_p:.2e}{' '*6} {a_p:.2e}")

    lines.append("")
    lines.append("[4] Frequency Analysis (Spectral Vulnerability)")
    lines.append("-" * 96)
    lines.append(f"{'Band':<20} {'Clean Error':<20} {'Adv Error':<20} {'Ratio (Adv/Clean)':<20}")
    lines.append("-" * 96)
    
    avg_clean_low = np.mean(metrics['clean_low_freq'])
    avg_clean_high = np.mean(metrics['clean_high_freq'])
    avg_adv_low = np.mean(metrics['adv_low_freq'])
    avg_adv_high = np.mean(metrics['adv_high_freq'])
    
    ratio_low = avg_adv_low / avg_clean_low if avg_clean_low > 1e-9 else 0.0
    ratio_high = avg_adv_high / avg_clean_high if avg_clean_high > 1e-9 else 0.0
    
    lines.append(f"{'Low Freq (k<=6)':<20} {avg_clean_low:.4f}{' '*14} {avg_adv_low:.4f}{' '*14} {ratio_low:.2f}x")
    lines.append(f"{'High Freq (k>6)':<20} {avg_clean_high:.4f}{' '*14} {avg_adv_high:.4f}{' '*14} {ratio_high:.2f}x")

    # NEW: Spectral Transfer Analysis Section
    if len(metrics['spectral_input_low_pct']) > 0:
        lines.append("")
        lines.append("[5] Spectral Transfer Analysis (Input Pert → Output Pert)")
        lines.append("-" * 96)
        avg_in_low = np.mean(metrics['spectral_input_low_pct'])
        avg_in_high = np.mean(metrics['spectral_input_high_pct'])
        avg_out_low = np.mean(metrics['spectral_output_low_pct'])
        avg_out_high = np.mean(metrics['spectral_output_high_pct'])
        
        lines.append(f"  Input Perturbation:   {avg_in_low:.1f}% Low-Freq / {avg_in_high:.1f}% High-Freq")
        lines.append(f"  Output Perturbation:  {avg_out_low:.1f}% Low-Freq / {avg_out_high:.1f}% High-Freq")
        
        low_shift = avg_out_low - avg_in_low
        high_shift = avg_out_high - avg_in_high
        lines.append(f"  Spectral Shift:       Low-Freq {low_shift:+.1f}% | High-Freq {high_shift:+.1f}%")

    lines.append("================================================================================")

    # Print and Save
    report_text = "\n".join(lines)
    print("\n" + report_text)
    
    with open(f"{args.output_dir}/eval_summary.txt", "w") as f: 
        f.write(report_text)
        
    print(f"\nResults saved to: {args.output_dir}")

    # === SAVE ALL METRICS TO JSON ===
    full_report = {
        "config": {
            "source_res": src_res,
            "target_res": tgt_res,
            "samples": n_eval
        },
        "ic_perturbation": {
            "abs_l2": float(np.mean(metrics['ic_pert_abs'])),
            "rel_l2": float(np.mean(metrics['ic_pert_rel'])),
            "low_freq": float(np.mean(metrics['ic_low_freq'])),
            "high_freq": float(np.mean(metrics['ic_high_freq']))
        },
        "global_metrics": {
            "clean_st_l2": float(avg_clean_l2),
            "adv_st_l2": float(avg_adv_l2),
            "clean_st_mse": float(avg_clean_mse),
            "adv_st_mse": float(avg_adv_mse)
        },
        "frequency_analysis": {
            "clean_low_freq": float(np.mean(metrics['clean_low_freq'])),
            "clean_high_freq": float(np.mean(metrics['clean_high_freq'])),
            "adv_low_freq": float(np.mean(metrics['adv_low_freq'])),
            "adv_high_freq": float(np.mean(metrics['adv_high_freq']))
        },
        "temporal_evolution": {
            "injection_l2": [float(x) for x in avg_inj],
            "drift_l2_rel": [float(x) for x in avg_drift],
            "clean_physics": [float(x) for x in avg_c_phys],
            "adv_physics": [float(x) for x in avg_a_phys],
            # NEW: Per-step errors vs ground truth
            "clean_error_l2_rel": [float(x) for x in avg_clean_err],
            "adv_error_l2_rel": [float(x) for x in avg_adv_err],
        }
    }
    
    # Add spectral transfer info if available
    if len(metrics['spectral_input_low_pct']) > 0:
        full_report["spectral_transfer"] = {
            "input_low_freq_pct": float(np.mean(metrics['spectral_input_low_pct'])),
            "input_high_freq_pct": float(np.mean(metrics['spectral_input_high_pct'])),
            "output_low_freq_pct": float(np.mean(metrics['spectral_output_low_pct'])),
            "output_high_freq_pct": float(np.mean(metrics['spectral_output_high_pct'])),
        }
    
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, 'w') as f:
        json.dump(full_report, f, indent=4)
        
    print(f"\nFull metrics saved to: {json_path}")

if __name__ == "__main__":
    evaluate()