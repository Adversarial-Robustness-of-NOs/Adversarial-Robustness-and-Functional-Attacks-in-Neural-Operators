import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os
import tomllib
from scipy import stats

def get_radial_spectrum(img):
    """Computes the 1D radial average of the 2D Fourier power spectrum."""
    if img.ndim == 3: img = img[0] # Take first channel if needed
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r = np.sqrt(x**2 + y**2).astype(int)
    
    # Bin by radius
    tbin = np.bincount(r.ravel(), magnitude.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    return radial_profile[:h//2] # Return up to Nyquist

def analyze_spectral_nature(u_clean, u_adv, model_modes=None):
    """
    Decomposes the attack into Amplitude vs Phase changes and Frequency distribution.
    """
    perturbation = u_adv - u_clean
    
    # 1. FFT
    f_clean = np.fft.fft2(u_clean)
    f_adv = np.fft.fft2(u_adv)
    f_pert = np.fft.fft2(perturbation)
    
    # 2. Amplitude vs Phase Impact
    # How much of the L2 distance came from amp change vs phase shift?
    amp_clean = np.abs(f_clean)
    phase_clean = np.angle(f_clean)
    
    amp_adv = np.abs(f_adv)
    phase_adv = np.angle(f_adv)
    
    # Construct "Hybrid" signals
    # A: Clean Amp + Adv Phase
    # B: Adv Amp + Clean Phase
    x_phase_attack = np.fft.ifft2(amp_clean * np.exp(1j * phase_adv)).real
    x_amp_attack = np.fft.ifft2(amp_adv * np.exp(1j * phase_clean)).real
    
    l2_total = np.linalg.norm(u_adv - u_clean)
    l2_phase_contrib = np.linalg.norm(x_phase_attack - u_clean)
    l2_amp_contrib = np.linalg.norm(x_amp_attack - u_clean)
    
    # 3. Radial Profiles (1D Spectrum)
    prof_pert = get_radial_spectrum(perturbation)
    prof_clean = get_radial_spectrum(u_clean)
    
    return {
        'l2_total': l2_total,
        'l2_phase': l2_phase_contrib,
        'l2_amp': l2_amp_contrib,
        'spectrum_pert': prof_pert,
        'spectrum_clean': prof_clean
    }

def visualize_spectral_analysis(results, model_modes, output_dir, sample_idx):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Attack Contribution (Phase vs Amp)
    # Why did the attack change the input?
    labels = ['Total Perturbation', 'Due to Phase Shift', 'Due to Amp Change']
    values = [results['l2_total'], results['l2_phase'], results['l2_amp']]
    colors = ['gray', 'orange', 'purple']
    
    axs[0].bar(labels, values, color=colors, alpha=0.7)
    axs[0].set_title(f"Sample {sample_idx}: Nature of Perturbation")
    axs[0].set_ylabel("L2 Norm Impact")
    
    # Plot 2: Radial Power Spectrum (Log Scale)
    # Where is the attack energy located?
    freqs = np.arange(len(results['spectrum_pert']))
    axs[1].plot(freqs, np.log10(results['spectrum_clean'] + 1e-8), label='Clean Signal', color='blue', alpha=0.5)
    axs[1].plot(freqs, np.log10(results['spectrum_pert'] + 1e-8), label='Attack Noise', color='red', linewidth=2)
    
    if model_modes:
        # Show the FNO Cutoff
        axs[1].axvline(x=model_modes, color='green', linestyle='--', label=f'FNO Cutoff (k={model_modes})')
        axs[1].text(model_modes + 1, axs[1].get_ylim()[0], "Model Ignores -->", color='green')
        
    axs[1].set_title("Perturbation Power Spectrum")
    axs[1].set_xlabel("Frequency (k)")
    axs[1].set_ylabel("Log Power")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Ratio (Attack / Clean)
    # Is the attack proportionally stronger at high frequencies?
    ratio = results['spectrum_pert'] / (results['spectrum_clean'] + 1e-8)
    axs[2].plot(freqs, ratio, color='black')
    if model_modes:
        axs[2].axvline(x=model_modes, color='green', linestyle='--')
        
    axs[2].set_title("Relative Attack Strength (Noise/Signal Ratio)")
    axs[2].set_xlabel("Frequency (k)")
    axs[2].set_yscale('log')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spectral_analysis_{sample_idx}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="temporal/ns_2d/data/attack/fno/eps001/attack_pgd_spatial.h5")
    parser.add_argument("--model_config", type=str, default="temporal/ns_2d/fno_model.toml")
    parser.add_argument("--output_dir", type=str, default="spectral_analysis")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model Config to find Modes
    modes = None
    if os.path.exists(args.model_config):
        with open(args.model_config, "rb") as f:
            conf = tomllib.load(f)["config"]
            # FNO usually defines modes like [12, 12]
            if 'modes' in conf:
                m = conf['modes']
                modes = m[0] if isinstance(m, list) else m
                print(f"Detected Model Modes (Frequency Cutoff): k={modes}")

    # Load Data
    print(f"Loading {args.data_path}...")
    with h5py.File(args.data_path, 'r') as f:
        # Assuming Standard Attack (IC Perturbation)
        if 'adversarial/x_init' in f:
            x_adv = f['adversarial/x_init'][:]
            # We need clean IC. Usually 'train/u' contains full trajectory.
            # x_clean is u[:, 0:in_channels]
            u_full = f['train/u'][:]
            # Assume In-Channels logic or just take first frame for analysis
            x_clean = u_full[:, :x_adv.shape[1]] 
        else:
            print("Error: Could not find 'adversarial/x_init'")
            return

    # Analyze
    print("Running Spectral Forensics...")
    for i in range(args.num_samples):
        # Take just the first channel (u velocity) for spectral analysis
        # x is (Batch, Time/Channel, H, W)
        u_c = x_clean[i, 0] 
        u_a = x_adv[i, 0]
        
        metrics = analyze_spectral_nature(u_c, u_a, modes)
        visualize_spectral_analysis(metrics, modes, args.output_dir, i)
        
    print(f"Analysis saved to {args.output_dir}")

if __name__ == "__main__":
    main()