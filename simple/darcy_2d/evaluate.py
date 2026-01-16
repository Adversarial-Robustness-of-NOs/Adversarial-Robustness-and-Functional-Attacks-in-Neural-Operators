import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import sys
import os
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import tomllib

# Adjust imports to match your project structure
from models.model_factory import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. UTILS & SPECTRAL ANALYSIS
# ==========================================

def resize_tensor(x, target_res, mode='bilinear'):
    """Robustly resizes tensors to (target_res, target_res)."""
    if x.shape[-1] == target_res: return x
    if x.ndim == 3: x = x.unsqueeze(1)
    out = F.interpolate(x, size=(target_res, target_res), mode=mode, align_corners=True)
    return out

def calculate_l2_metrics(pred, true):
    """Calculates both Absolute and Relative L2 Error per sample."""
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    if true.ndim == 3: true = true.unsqueeze(1)
    diff = pred - true
    abs_l2 = torch.norm(diff.reshape(diff.shape[0], -1), dim=1)
    true_norm = torch.norm(true.reshape(true.shape[0], -1), dim=1)
    rel_l2 = abs_l2 / (true_norm + 1e-8)
    return abs_l2, rel_l2

def get_radial_spectrum(batch_img):
    """Computes Radial Average Power Spectrum (RAPS). Returns (B, max_r)."""
    if batch_img.ndim == 4: batch_img = batch_img.squeeze(1)
    
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

def analyze_bands(spectrum):
    """Splits spectrum into Low (0-25%) and High (25-100%) frequency magnitudes."""
    k_max = len(spectrum)
    split_idx = int(k_max * 0.25)
    low_mag = np.sum(spectrum[:split_idx])
    high_mag = np.sum(spectrum[split_idx:])
    return low_mag, high_mag

# ==========================================
# 2. PHYSICS LOSS (Darcy Equation)
# ==========================================

class DarcyPhysicsLoss(nn.Module):
    """
    Computes PDE residual for Darcy equation: -div(a * grad(u)) = f
    """
    def __init__(self, res, force=1.0):
        super().__init__()
        self.h = 1.0 / (res - 1)
        self.force = force
        
        # Central Difference Kernels
        self.dx_kernel = torch.tensor([[[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]]], 
                                      dtype=torch.float32) / self.h
        self.dy_kernel = torch.tensor([[[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]]], 
                                      dtype=torch.float32) / self.h

    def to(self, device):
        self.dx_kernel = self.dx_kernel.to(device)
        self.dy_kernel = self.dy_kernel.to(device)
        return self

    def compute_flux(self, u, a):
        u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate') 
        du_dx = F.conv2d(u_pad, self.dx_kernel)
        du_dy = F.conv2d(u_pad, self.dy_kernel)
        return a * du_dx, a * du_dy

    def forward(self, u, a):
        """
        Returns per-sample PDE residual and BC loss.
        u: predicted pressure (B, 1, H, W)
        a: permeability (B, 1, H, W)
        """
        # Calculate Residual: -div(a * grad(u)) - f = 0
        flux_x, flux_y = self.compute_flux(u, a)
        d_flux_x_dx = F.conv2d(F.pad(flux_x, (1, 1, 1, 1), mode='replicate'), self.dx_kernel)
        d_flux_y_dy = F.conv2d(F.pad(flux_y, (1, 1, 1, 1), mode='replicate'), self.dy_kernel)
        
        residual = -(d_flux_x_dx + d_flux_y_dy) - self.force
        
        # Per-sample PDE loss (MSE of residual)
        loss_pde = torch.mean(residual**2, dim=(1, 2, 3))
        
        # Dirichlet BC (u=0 at boundaries)
        loss_bc = (torch.mean(u[..., 0, :]**2, dim=(1, 2)) + 
                   torch.mean(u[..., -1, :]**2, dim=(1, 2)) + 
                   torch.mean(u[..., :, 0]**2, dim=(1, 2)) + 
                   torch.mean(u[..., :, -1]**2, dim=(1, 2)))
                   
        return loss_pde, loss_bc

def compute_physics_residual_map(u, a, force=1.0):
    """
    Computes the element-wise PDE residual: Res = -div(a * grad(u)) - f
    Returns a tensor of the same shape as u.
    """
    if u.ndim == 2: u = u.unsqueeze(0).unsqueeze(0)
    if u.ndim == 3: u = u.unsqueeze(1)
    if a.ndim == 2: a = a.unsqueeze(0).unsqueeze(0)
    if a.ndim == 3: a = a.unsqueeze(1)

    res = u.shape[-1]
    h = 1.0 / res # Assuming unit domain [0,1]
    
    # Central Difference Kernels
    dx_k = torch.tensor([[[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]]], device=u.device) / h
    dy_k = torch.tensor([[[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]]], device=u.device) / h
    
    # 1. Compute Flux: a * grad(u)
    u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate') 
    du_dx = F.conv2d(u_pad, dx_k)
    du_dy = F.conv2d(u_pad, dy_k)
    
    flux_x = a * du_dx
    flux_y = a * du_dy
    
    # 2. Compute Divergence of Flux
    flux_x_pad = F.pad(flux_x, (1, 1, 1, 1), mode='replicate')
    flux_y_pad = F.pad(flux_y, (1, 1, 1, 1), mode='replicate')
    
    d_flux_x_dx = F.conv2d(flux_x_pad, dx_k)
    d_flux_y_dy = F.conv2d(flux_y_pad, dy_k)
    
    # 3. Residual = -div(Flux) - f
    residual = -(d_flux_x_dx + d_flux_y_dy) - force
    
    return residual


def save_heatmap_standard(idx, x, y, out, metric_rmse, metric_rel, output_dir, x_clean=None, pde_diff=None, out_clean=None):
    """Visualizes Input, GT, Pred, Abs Error, Rel Error, Physics Residual, and Spectral Analysis."""
    # Compute Physics Residual Map for this sample
    x_tensor = torch.tensor(x, device=DEVICE).unsqueeze(0).unsqueeze(0)
    out_tensor = torch.tensor(out, device=DEVICE).unsqueeze(0).unsqueeze(0)
    phys_res_map = compute_physics_residual_map(out_tensor, x_tensor).squeeze().cpu().numpy()
    
    # Compute Relative Error Map & Scalar
    eps = 1e-6
    abs_err = np.abs(y - out)
    rel_err_map = abs_err / (np.abs(y) + eps)
    
    # Determine columns: base 6 + perturbation map (if x_clean) + spectral plot (if x_clean)
    cols = 6
    has_perturbation = x_clean is not None
    if has_perturbation: 
        cols += 2  # perturbation map + spectral comparison
    
    fig, axs = plt.subplots(1, cols, figsize=(5*cols, 5))
    
    def to_img(t): return t.squeeze() 
    cb_kwargs = {'fraction': 0.046, 'pad': 0.04}
    
    # 1. Input (Adversarial)
    im0 = axs[0].imshow(to_img(x), cmap='gray', origin='lower')
    axs[0].set_title("Adv Input (Permeability)")
    plt.colorbar(im0, ax=axs[0], **cb_kwargs)
    
    # 2. GT
    im1 = axs[1].imshow(to_img(y), cmap='inferno', origin='lower')
    axs[1].set_title("Ground Truth")
    plt.colorbar(im1, ax=axs[1], **cb_kwargs)
    
    # 3. Prediction
    im2 = axs[2].imshow(to_img(out), cmap='inferno', origin='lower')
    axs[2].set_title("Prediction")
    plt.colorbar(im2, ax=axs[2], **cb_kwargs)
    
    # 4. Absolute Error
    im3 = axs[3].imshow(abs_err, cmap='magma', origin='lower')
    axs[3].set_title(f"Abs Error (RMSE: {metric_rmse:.4f})")
    plt.colorbar(im3, ax=axs[3], **cb_kwargs)

    # 5. Relative Error
    im4 = axs[4].imshow(rel_err_map, cmap='magma', origin='lower')
    axs[4].set_title(f"Rel Error (L2: {metric_rel:.4f})")
    plt.colorbar(im4, ax=axs[4], **cb_kwargs)
    
    # 6. Physics Residual
    im5 = axs[5].imshow(np.abs(phys_res_map), cmap='twilight', origin='lower')
    title = "Physics Residual"
    if pde_diff is not None:
        title += f"\nLoss Diff: {pde_diff:.2e}"
    axs[5].set_title(title)
    plt.colorbar(im5, ax=axs[5], **cb_kwargs)
    
    # 7 & 8. Perturbation Map and Spectral Comparison (if x_clean available)
    if has_perturbation:
        # 7. Perturbation Map
        pert_map = x - x_clean
        pert_norm = np.linalg.norm(pert_map)
        im6 = axs[6].imshow(to_img(pert_map), cmap='seismic', origin='lower')
        axs[6].set_title(f"Input Perturbation\nL2: {pert_norm:.4f}")
        plt.colorbar(im6, ax=axs[6], **cb_kwargs)
        
        # 8. Spectral Frequency Line Plot
        ax_spec = axs[7]
        
        # Compute input perturbation spectrum
        input_pert = torch.tensor(pert_map, dtype=torch.float32).unsqueeze(0)
        if input_pert.ndim == 3:
            input_pert = input_pert.unsqueeze(1)
        input_pert_spectrum = get_radial_spectrum(input_pert)[0]
        
        # Compute output perturbation spectrum
        if out_clean is not None:
            output_pert = out - out_clean
        else:
            output_pert = out - y
        
        output_pert_tensor = torch.tensor(output_pert, dtype=torch.float32).unsqueeze(0)
        if output_pert_tensor.ndim == 3:
            output_pert_tensor = output_pert_tensor.unsqueeze(1)
        output_pert_spectrum = get_radial_spectrum(output_pert_tensor)[0]
        
        # Frequency axis
        k = np.arange(len(input_pert_spectrum))
        
        # Plot both spectra
        ax_spec.semilogy(k, input_pert_spectrum + 1e-10, 'b-', linewidth=2, label='Input Perturbation')
        ax_spec.semilogy(k, output_pert_spectrum + 1e-10, 'r-', linewidth=2, label='Output Perturbation')
        
        # Mark low/high frequency boundary (25%)
        boundary = int(len(k) * 0.25)
        ax_spec.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7, label='Low/High Boundary')
        
        # Shade regions
        ax_spec.axvspan(0, boundary, alpha=0.1, color='blue')
        ax_spec.axvspan(boundary, len(k), alpha=0.1, color='red')
        
        # Compute energy ratios for annotation
        in_low, in_high = analyze_bands(input_pert_spectrum)
        out_low, out_high = analyze_bands(output_pert_spectrum)
        in_total = in_low + in_high + 1e-10
        out_total = out_low + out_high + 1e-10
        
        in_low_pct = in_low / in_total * 100
        in_high_pct = in_high / in_total * 100
        out_low_pct = out_low / out_total * 100
        out_high_pct = out_high / out_total * 100
        
        ax_spec.set_xlabel('Frequency (k)', fontsize=10)
        ax_spec.set_ylabel('Magnitude (log)', fontsize=10)
        ax_spec.set_title(f'Spectral Comparison\nIn: {in_low_pct:.0f}%L/{in_high_pct:.0f}%H → Out: {out_low_pct:.0f}%L/{out_high_pct:.0f}%H', fontsize=10)
        ax_spec.legend(loc='upper right', fontsize=8)
        ax_spec.grid(True, alpha=0.3)
        ax_spec.set_xlim([0, len(k)-1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_{idx:03d}_eval.png", dpi=150)
    plt.close()


# ==========================================
# 3. PERTURBATION CHARACTERIZATION
# ==========================================

def characterize_perturbation(x_adv, x_clean):
    """
    Analyze the nature of input perturbations.
    Returns dict with various perturbation characteristics.
    """
    delta = x_adv - x_clean  # (B, 1, H, W)
    
    # Basic statistics
    abs_delta = torch.abs(delta)
    
    stats = {
        # Magnitude statistics
        'mean_abs': float(abs_delta.mean().item()),
        'max_abs': float(abs_delta.max().item()),
        'std': float(delta.std().item()),
        'sparsity': float((abs_delta < 1e-6).float().mean().item()),  # % of near-zero elements
        
        # L-inf and L2 norms (per sample, then averaged)
        'avg_linf': float(abs_delta.reshape(delta.shape[0], -1).max(dim=1)[0].mean().item()),
        'avg_l2': float(torch.norm(delta.reshape(delta.shape[0], -1), dim=1).mean().item()),
        'avg_l1': float(abs_delta.reshape(delta.shape[0], -1).sum(dim=1).mean().item()),
    }
    
    # Spectral analysis of perturbation
    delta_spectrum = get_radial_spectrum(delta)
    mean_spectrum = delta_spectrum.mean(axis=0)
    
    low_energy, high_energy = analyze_bands(mean_spectrum)
    total_energy = low_energy + high_energy + 1e-8
    
    stats['pert_low_freq_ratio'] = float(low_energy / total_energy)
    stats['pert_high_freq_ratio'] = float(high_energy / total_energy)
    
    # Smoothness (gradient magnitude of perturbation)
    if delta.shape[-1] > 2:
        grad_x = delta[..., :, 1:] - delta[..., :, :-1]
        grad_y = delta[..., 1:, :] - delta[..., :-1, :]
        grad_mag = torch.sqrt(grad_x[..., :-1, :].pow(2) + grad_y[..., :, :-1].pow(2) + 1e-8)
        stats['smoothness'] = float(1.0 / (grad_mag.mean().item() + 1e-8))  # Higher = smoother
        stats['avg_gradient_mag'] = float(grad_mag.mean().item())
    
    return stats, mean_spectrum


# ==========================================
# 4. DATA LOADING
# ==========================================

def load_data_custom(data_path, target_path=None):
    print(f"   Loading data from: {data_path}")
    with h5py.File(data_path, 'r') as f:
        keys, attrs = list(f.keys()), dict(f.attrs)
        x_key = 'nu' if 'nu' in keys else 'x'
        x = torch.tensor(f[x_key][:], dtype=torch.float32)
        x_clean = torch.tensor(f['original_nu'][:], dtype=torch.float32) if 'original_nu' in keys else None
        y_key = 'tensor' if 'tensor' in f else 'y'
        y = torch.tensor(f[y_key][:], dtype=torch.float32) if y_key in keys else None
        print(f"   Loaded x: {x.shape}, y: {y.shape if y is not None else None}")
        
    n_samples = x.shape[0]
        
    if target_path:
        print(f"   Loading target from: {target_path}")
        with h5py.File(target_path, 'r') as f:
            print(f"   Target file keys: {list(f.keys())}")
            y_key = 'tensor' if 'tensor' in f else 'y'
            print(f"   Using y_key: {y_key}")
            y_full = torch.tensor(f[y_key][:], dtype=torch.float32)
            print(f"   Target y_full shape: {y_full.shape}")
            # Match sample count - take first n_samples from target
            if y_full.shape[0] >= n_samples:
                y = y_full[:n_samples]
            else:
                # If target has fewer samples, limit x to match
                y = y_full
                x = x[:y.shape[0]]
                if x_clean is not None:
                    x_clean = x_clean[:y.shape[0]]
                print(f"   [WARNING] Target has fewer samples ({y.shape[0]}) than attack ({n_samples})")
            print(f"   Final y shape: {y.shape}")
            
    return x, y, x_clean, 'standard', attrs


# ==========================================
# 5. EVALUATION MAIN
# ==========================================

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--plot_samples", type=int, default=10)
    parser.add_argument("--force_term", type=float, default=1.0)
    parser.add_argument("--cross_res_mode", type=str, default=None, 
                       help="Cross-resolution mode: 'low', 'high', or None for standard")
    parser.add_argument("--base_res", type=int, default=None,
                       help="Base resolution the attack was generated at")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    with open(args.model_config, "rb") as f: 
        model_conf = tomllib.load(f)["config"]
    model = create_model(model_conf, DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE)['model_state_dict'], strict=False)
    model.eval()

    # Load data
    x, y, x_clean, mode, attrs = load_data_custom(args.data_path, args.target_path)
    
    # Handle cross-resolution mode
    cross_res_info = {}
    if args.cross_res_mode and args.target_path and args.base_res:
        cross_res_info['mode'] = args.cross_res_mode
        cross_res_info['base_res'] = args.base_res
        cross_res_info['target_res'] = y.shape[-1] if y is not None else x.shape[-1]
        
        # Resize adversarial inputs to target resolution
        original_res = x.shape[-1]
        target_res = cross_res_info['target_res']
        
        print(f"   Cross-Res Mode: {args.cross_res_mode}")
        print(f"   Input shape: {x.shape}, Target shape: {y.shape if y is not None else 'None'}")
        print(f"   Base Res: {args.base_res} -> Target Res: {target_res}")
        
        if original_res != target_res:
            x = resize_tensor(x, target_res)
            if x_clean is not None:
                x_clean = resize_tensor(x_clean, target_res)
            print(f"   Resized x from {original_res} to {target_res}")
    
    tensors = [x, y]
    if x_clean is not None: 
        tensors.append(x_clean)
    
    # Sanity check: all tensors must have same batch size
    batch_sizes = [t.shape[0] for t in tensors if t is not None]
    if len(set(batch_sizes)) > 1:
        min_size = min(batch_sizes)
        print(f"   [WARNING] Tensor size mismatch: {batch_sizes}. Truncating to {min_size}")
        tensors = [t[:min_size] if t is not None else t for t in tensors]
        x = tensors[0]
        y = tensors[1]
        if len(tensors) > 2:
            x_clean = tensors[2]
    
    loader = DataLoader(TensorDataset(*tensors), batch_size=model_conf.get('batch_size', 32))
    
    # Get resolution for physics loss
    res = x.shape[-1]
    physics_loss_fn = DarcyPhysicsLoss(res=res, force=args.force_term).to(DEVICE)
    
    # Results containers
    results = {
        'abs_l2': [], 'rel_l2': [], 
        'pert_rel_l2': [], 'pert_abs_l2': [], 
        'amp_abs_ratio': [], 'amp_rel_ratio': [],
        'pde_loss': [], 'bc_loss': []
    }
    spectra = {'in_adv': [], 'in_cln': [], 'out_pred': [], 'out_tgt': [], 'perturbation': []}
    
    # For perturbation characterization (aggregate across batches)
    all_x_adv = []
    all_x_cln = []

    plot_count = 0

    with torch.no_grad():
        for batch in loader:
            bx, by = batch[0].to(DEVICE), batch[1].to(DEVICE)
            bx = bx.unsqueeze(1) if bx.ndim == 3 else bx
            by = by.unsqueeze(1) if by.ndim == 3 else by
            
            out = model(bx)
            if out.shape[-1] != by.shape[-1]: 
                out = resize_tensor(out, by.shape[-1])
            
            # === Output Error Metrics ===
            abs_l2, rel_l2 = calculate_l2_metrics(out, by)
            results['abs_l2'].append(abs_l2.cpu().numpy())
            results['rel_l2'].append(rel_l2.cpu().numpy())
            
            # === Physics Loss ===
            pde_loss, bc_loss = physics_loss_fn(out, bx)
            results['pde_loss'].append(pde_loss.cpu().numpy())
            results['bc_loss'].append(bc_loss.cpu().numpy())
            
            # === Spectral Data ===
            spectra['in_adv'].append(get_radial_spectrum(bx))
            spectra['out_pred'].append(get_radial_spectrum(out))
            spectra['out_tgt'].append(get_radial_spectrum(by))
            
            # === Input Perturbation Analysis ===
            if x_clean is not None:
                bc = batch[2].to(DEVICE)
                bc = bc.unsqueeze(1) if bc.ndim == 3 else bc
                
                # Perturbation = adversarial - clean
                p_abs_l2, p_rel_l2 = calculate_l2_metrics(bx, bc)
                results['pert_abs_l2'].append(p_abs_l2.cpu().numpy())
                results['pert_rel_l2'].append(p_rel_l2.cpu().numpy())
                
                # Amplification Ratio: Output Error / Input Perturbation
                results['amp_abs_ratio'].append((abs_l2 / (p_abs_l2 + 1e-8)).cpu().numpy())
                results['amp_rel_ratio'].append((rel_l2 / (p_rel_l2 + 1e-8)).cpu().numpy())
                
                spectra['in_cln'].append(get_radial_spectrum(bc))
                spectra['perturbation'].append(get_radial_spectrum(bx - bc))
                
                # Store for characterization
                all_x_adv.append(bx.cpu())
                all_x_cln.append(bc.cpu())

            # === Save heatmaps for individual samples ===
            if plot_count < args.plot_samples:
                # Get clean input for comparison (if available)
                bc_np = None
                if x_clean is not None and len(batch) > 2:
                    bc_np = batch[2].cpu().numpy()
                
                for i in range(bx.shape[0]):
                    if plot_count >= args.plot_samples:
                        break
                    
                    # Get clean input for this sample
                    current_x_clean = bc_np[i].squeeze() if bc_np is not None else None
                    
                    # Get per-sample metrics from current batch
                    sample_rmse = torch.sqrt(torch.mean((out[i] - by[i])**2)).item()
                    sample_rel_l2 = rel_l2[i].item()
                    
                    # PDE loss difference (adv - clean baseline if available)
                    pde_diff = None
                    current_out_clean = None
                    if x_clean is not None and bc_np is not None:
                        # Compute clean prediction PDE loss for this sample
                        bc_tensor = torch.tensor(bc_np[i:i+1], device=DEVICE)
                        if bc_tensor.ndim == 3:
                            bc_tensor = bc_tensor.unsqueeze(1)
                        out_clean = model(bc_tensor)
                        if out_clean.shape[-1] != by.shape[-1]:
                            out_clean = resize_tensor(out_clean, by.shape[-1])
                        pde_clean, _ = physics_loss_fn(out_clean, bc_tensor)
                        pde_adv = pde_loss[i].item() if hasattr(pde_loss, '__getitem__') else pde_loss.mean().item()
                        pde_diff = pde_adv - pde_clean.mean().item()
                        # Store clean output for spectral comparison
                        current_out_clean = out_clean.squeeze().cpu().numpy()
                    
                    save_heatmap_standard(
                        plot_count, 
                        bx[i].squeeze().cpu().numpy(), 
                        by[i].squeeze().cpu().numpy(), 
                        out[i].squeeze().cpu().numpy(), 
                        sample_rmse,
                        sample_rel_l2,
                        args.output_dir,
                        x_clean=current_x_clean,
                        pde_diff=pde_diff,
                        out_clean=current_out_clean
                    )
                    print(f"   [SAVED] sample_{plot_count:03d}_eval.png")
                    plot_count += 1


    # ==========================================
    # AGGREGATION
    # ==========================================
    all_rel = np.concatenate(results['rel_l2'])
    all_abs = np.concatenate(results['abs_l2'])
    all_pde = np.concatenate(results['pde_loss'])
    all_bc = np.concatenate(results['bc_loss'])
    
    m_spec_in_adv = np.concatenate(spectra['in_adv']).mean(axis=0)
    m_spec_out_pred = np.concatenate(spectra['out_pred']).mean(axis=0)
    m_spec_out_tgt = np.concatenate(spectra['out_tgt']).mean(axis=0)
    
    in_adv_l, in_adv_h = analyze_bands(m_spec_in_adv)
    out_pred_l, out_pred_h = analyze_bands(m_spec_out_pred)
    out_tgt_l, out_tgt_h = analyze_bands(m_spec_out_tgt)
    
    # ==========================================
    # REPORTING
    # ==========================================
    summary_lines = ["ROBUSTNESS & STABILITY REPORT", "="*50]
    
    benchmark_data = {
        "metrics": {
            "avg_output_rel_l2": float(np.mean(all_rel)),
            "avg_output_abs_l2": float(np.mean(all_abs)),
            "avg_pde_loss": float(np.mean(all_pde)),
            "avg_bc_loss": float(np.mean(all_bc)),
            "avg_physics_loss": float(np.mean(all_pde) + np.mean(all_bc)),
        },
        "spectral_changes": {
            "out_low_diff": float((out_pred_l - out_tgt_l) / (out_tgt_l + 1e-8)),
            "out_high_diff": float((out_pred_h - out_tgt_h) / (out_tgt_h + 1e-8))
        },
        "perturbation": {},
        "cross_resolution": cross_res_info if cross_res_info else None
    }
    
    summary_lines.append(f"Avg Output Rel L2 Error:   {np.mean(all_rel):.2%}")
    summary_lines.append(f"Avg Output Abs L2 Error:   {np.mean(all_abs):.6f}")
    summary_lines.append(f"Avg PDE Residual Loss:     {np.mean(all_pde):.6f}")
    summary_lines.append(f"Avg BC Loss:               {np.mean(all_bc):.6f}")
    summary_lines.append("-" * 50)
    
    # === Perturbation Characterization ===
    if x_clean is not None and len(all_x_adv) > 0:
        # Concatenate all batches
        x_adv_full = torch.cat(all_x_adv, dim=0)
        x_cln_full = torch.cat(all_x_cln, dim=0)
        
        pert_stats, pert_spectrum = characterize_perturbation(x_adv_full, x_cln_full)
        
        m_spec_in_cln = np.concatenate(spectra['in_cln']).mean(axis=0)
        m_spec_pert = np.concatenate(spectra['perturbation']).mean(axis=0)
        cln_l, cln_h = analyze_bands(m_spec_in_cln)
        
        avg_abs_pert = np.mean(np.concatenate(results['pert_abs_l2']))
        avg_rel_pert = np.mean(np.concatenate(results['pert_rel_l2']))
        avg_abs_amp = np.mean(np.concatenate(results['amp_abs_ratio']))
        avg_rel_amp = np.mean(np.concatenate(results['amp_rel_ratio']))
        
        in_l_chg = (in_adv_l - cln_l) / (cln_l + 1e-8)
        in_h_chg = (in_adv_h - cln_h) / (cln_h + 1e-8)
        
        summary_lines.append("INPUT PERTURBATION ANALYSIS")
        summary_lines.append("-" * 50)
        summary_lines.append(f"Avg Perturbation Abs L2:   {avg_abs_pert:.6f}")
        summary_lines.append(f"Avg Perturbation Rel L2:   {avg_rel_pert:.2%}")
        summary_lines.append(f"Amplification Ratio (Abs): {avg_abs_amp:.2f}x")
        summary_lines.append(f"Amplification Ratio (Rel): {avg_rel_amp:.2f}x")
        summary_lines.append("-" * 50)
        summary_lines.append("PERTURBATION CHARACTERISTICS")
        summary_lines.append(f"  Max Abs Value:           {pert_stats['max_abs']:.6f}")
        summary_lines.append(f"  Avg L-inf Norm:          {pert_stats['avg_linf']:.6f}")
        summary_lines.append(f"  Sparsity:                {pert_stats['sparsity']:.2%}")
        summary_lines.append(f"  Smoothness:              {pert_stats['smoothness']:.2f}")
        summary_lines.append(f"  Low-Freq Energy Ratio:   {pert_stats['pert_low_freq_ratio']:.2%}")
        summary_lines.append(f"  High-Freq Energy Ratio:  {pert_stats['pert_high_freq_ratio']:.2%}")
        summary_lines.append("-" * 50)
        summary_lines.append("SPECTRAL CHANGES")
        summary_lines.append(f"  Input Low-Freq Change:   {in_l_chg:+.2%}")
        summary_lines.append(f"  Input High-Freq Change:  {in_h_chg:+.2%}")
        
        # Store in benchmark data
        benchmark_data["metrics"]["avg_abs_input_pert_l2"] = float(avg_abs_pert)
        benchmark_data["metrics"]["avg_rel_input_pert_l2"] = float(avg_rel_pert)
        benchmark_data["metrics"]["amplification_abs_ratio"] = float(avg_abs_amp)
        benchmark_data["metrics"]["amplification_rel_ratio"] = float(avg_rel_amp)
        
        benchmark_data["perturbation"] = {
            "max_abs": float(pert_stats['max_abs']),
            "avg_linf": float(pert_stats['avg_linf']),
            "avg_l2": float(pert_stats['avg_l2']),
            "sparsity": float(pert_stats['sparsity']),
            "smoothness": float(pert_stats['smoothness']),
            "avg_gradient_mag": float(pert_stats.get('avg_gradient_mag', 0)),
            "low_freq_ratio": float(pert_stats['pert_low_freq_ratio']),
            "high_freq_ratio": float(pert_stats['pert_high_freq_ratio']),
        }
        
        benchmark_data["spectral_changes"]["in_low_chg"] = float(in_l_chg)
        benchmark_data["spectral_changes"]["in_high_chg"] = float(in_h_chg)

    summary_lines.append(f"Output Low-Freq Diff:      {benchmark_data['spectral_changes']['out_low_diff']:+.2%}")
    summary_lines.append(f"Output High-Freq Diff:     {benchmark_data['spectral_changes']['out_high_diff']:+.2%}")
    
    # Save Results
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f: 
        f.write("\n".join(summary_lines))
    with open(os.path.join(args.output_dir, "benchmark_metrics.json"), "w") as f: 
        json.dump(benchmark_data, f, indent=4)
    
    print("\n" + "\n".join(summary_lines))
    print(f"\nBenchmark Metrics saved to: {args.output_dir}")


if __name__ == "__main__": 
    evaluate()