import numpy as np
import h5py
import os
from scipy.integrate import odeint
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

N_SAMPLES = 1000
NX = 1024 # Space resolution     
NT = 201 # Temporal resolution
X_L = 1.0 # X in (0..1.0)
T_L = 2.0 # T in (0..2.0)
K_TOT = 32 # Total number of modes the model support (sine waves to combine)
K_SEL = 24 # Number of modes chosen for this dataset

# Physics Params (PDEBench Scaling)
NU_CONFIG = 0.01
NU = NU_CONFIG / np.pi 

# Output Paths
OUT_DIR = 'data/eval_suite_k32/'
NUM_CORES = 1

# Noise Parameters
SPATIAL_SIGMA = 0.05   # Amplitude of pixel noise
SPECTRAL_SIGMA = 0.1   # Amplitude of Fourier mode noise

GENERATE_SPECTRAL_NOISE = True
GENERATE_SPATIAL_NOISE = True

# Keep CLI args around so forked workers can re-init state
INIT_ARGS = None

# Configuration
def init_globals(args):
    global N_SAMPLES
    global NX, NT, X_L, T_L, K_TOT, K_SEL
    global DX, X_GRID, T_EVAL, K_WAVENUMBERS
    global NU_CONFIG, NU
    global OUT_DIR, NUM_CORES
    global SPATIAL_SIGMA
    global SPECTRAL_SIGMA
    global GENERATE_SPATIAL_NOISE, GENERATE_SPECTRAL_NOISE
    global INIT_ARGS

    N_SAMPLES = args.num_samples
    NX = args.spatial_resolution
    NT = args.temporal_resolution
    X_L = args.max_x
    T_L = args.max_t
    K_TOT = args.k_tot
    K_SEL = args.k_sel

    NU_CONFIG = args.nu
    NU = NU_CONFIG / np.pi if args.nu_div_by_pi else NU_CONFIG
    
    OUT_DIR = args.out_dir
    NUM_CORES = args.parallelism

    GENERATE_SPECTRAL_NOISE = args.generate_spectral_noise
    GENERATE_SPATIAL_NOISE = args.generate_spatial_noise

    DX = X_L / NX
    X_GRID = np.linspace(0, X_L, NX, endpoint=False)
    T_EVAL = np.linspace(0, T_L, NT)
    K_WAVENUMBERS = 2 * np.pi * np.fft.rfftfreq(NX, d=DX)
    INIT_ARGS = args

# Solution to the burgeres equation via FFT ---
def burgers_rhs(u, t, k, nu):
    u_hat = np.fft.rfft(u)
    u_x = np.fft.irfft(1j * k * u_hat)
    u_xx = np.fft.irfft(-(k**2) * u_hat)
    return -u * u_x + nu * u_xx

# Input condition generator. Works as a sumation of sine waves, similar to PDEBench.
# This one, as opposed to PDEBench, creates clean input conditions (without nonliniearities).
def generate_base_ic(seed):
    """Generates the clean, smooth base signal (Simplified version)."""
    rng = np.random.default_rng(seed)
    
    k_tot = K_TOT 
    selected_indices = rng.integers(0, k_tot, size=K_SEL)
    active_modes = np.zeros(k_tot)
    for idx in selected_indices:
        active_modes[idx] += 1
        
    u = np.zeros(NX)
    L_domain = 1.0 
    modes = np.arange(1, k_tot + 1)
    
    for k_idx, count in enumerate(active_modes):
        if count == 0: continue
        k = modes[k_idx]
        amp = rng.uniform(0, 1)
        phase = 2 * np.pi * rng.uniform(0, 1)
        u += count * amp * np.sin(2 * np.pi * k * X_GRID / L_domain + phase)

    # Normalize clean base
    if np.max(np.abs(u)) > 1e-9:
        u = u / np.max(np.abs(u))
    return u

def add_spatial_noise(u, seed):
    """Adds Gaussian noise to the physical domain."""
    rng = np.random.default_rng(seed + 100000) # Different seed offset
    noise = rng.normal(0, SPATIAL_SIGMA, size=u.shape)
    return u + noise

def add_spectral_noise(u, seed):
    """Adds Gaussian noise to the frequency domain (preserves periodicity)."""
    rng = np.random.default_rng(seed + 200000)
    
    # Transform to freq
    u_hat = np.fft.rfft(u)
    
    # We add noise to real and imag parts separately
    noise_real = rng.normal(0, SPECTRAL_SIGMA, size=u_hat.shape)
    noise_imag = rng.normal(0, SPECTRAL_SIGMA, size=u_hat.shape)
    noise = noise_real + 1j * noise_imag
    
    # Apply noise (Skip DC component to keep mean stable-ish, optional)
    u_hat_noisy = u_hat + noise
    u_hat_noisy[0] = u_hat[0] 
    
    # Transform back
    return np.fft.irfft(u_hat_noisy, n=NX)

def process_sample(seed):
    """
    Generates Base, Spatial, and Spectral versions of one sample,
    then solves the PDE for ALL of them.
    """
    # A. Generate ICs
    u0_clean = generate_base_ic(seed)
    sol_clean = odeint(burgers_rhs, u0_clean, T_EVAL, args=(K_WAVENUMBERS, NU))
    result = {
        'clean': sol_clean.astype(np.float32)
    }

    if GENERATE_SPATIAL_NOISE:
        u0_spatial = add_spatial_noise(u0_clean, seed)
        sol_spatial = odeint(burgers_rhs, u0_spatial, T_EVAL, args=(K_WAVENUMBERS, NU))
        result['spatial'] = sol_spatial.astype(np.float32)
    
    if GENERATE_SPECTRAL_NOISE:
        u0_spectral = add_spectral_noise(u0_clean, seed)
        sol_spectral = odeint(burgers_rhs, u0_spectral, T_EVAL, args=(K_WAVENUMBERS, NU))    
        result['spectral'] = sol_spectral.astype(np.float32)
    
    return result

def generate_parallel():
    print(f"Samples: {N_SAMPLES}")
    print(f"Noise: Spatial sigma={SPATIAL_SIGMA}, Spectral sigma={SPECTRAL_SIGMA}")
    
    seeds = [np.random.randint(0, 10000000) + i for i in range(N_SAMPLES)]
    
    if NUM_CORES <= 1:
        results = [process_sample(s) for s in tqdm(seeds, total=N_SAMPLES)]
    else:
        with Pool(processes=NUM_CORES, initializer=init_globals, initargs=(INIT_ARGS,)) as pool:
            iterator = pool.imap(process_sample, seeds, chunksize=5)
            results = list(tqdm(iterator, total=N_SAMPLES))
        
    data_clean = np.stack([r['clean'] for r in results])
    data_spatial = np.stack([r['spatial'] for r in results])
    data_spectral = np.stack([r['spectral'] for r in results])
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Save Function
    def save_h5(name, data):
        path = os.path.join(OUT_DIR, name)
        with h5py.File(path, 'w') as f:
            f.create_dataset('tensor', data=data)
            f.create_dataset('x-coordinate', data=X_GRID)
            f.create_dataset('t-coordinate', data=T_EVAL)
            f.attrs['nu'] = NU
        print(f"Saved {name}")

    save_h5('eval_clean.h5', data_clean)
    save_h5('eval_spatial_noise.h5', data_spatial)
    save_h5('eval_spectral_noise.h5', data_spectral)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FNO training data generator")

    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to create")
    parser.add_argument("--spatial_resolution", type=int, default=1024, help="How many points on the x axis represent a funcion")
    parser.add_argument("--temporal_resolution", type=int, default=201, help="How many points on the time axis represent a funcion")
    parser.add_argument("--max_x", type=float, default=1.0, help="Default upper bound for X")
    parser.add_argument("--max_t", type=float, default=2.0, help="Default upper bound for T")
    parser.add_argument("--k_sel", type=int, default=32, help="Default number of modes for the function")
    parser.add_argument("--k_tot", type=int, default=24, help="Default total number of modes supported for the function")
    parser.add_argument("--nu", type=float, default=0.01, help="Default NU value")
    parser.add_argument("--nu_div_by_pi", action="store_true", help="Wether to scale it like PDEBench (divide by PI)")
    parser.add_argument("--spatial_sigma", type=float, default=0.05, help="Spatial noise coefficient (for spatial noise dataset)")
    parser.add_argument("--spectral_sigma", type=float, default=0.1, help="Spectral noise coefficient (for spectral noise dataset)")
    parser.add_argument("--out_dir", type=str, default="eval_suite", help="Output directory to output the dataset")
    parser.add_argument("--parallelism", type=int, default=1, help="Parallelism")

    parser.add_argument(
        "--generate_spatial_noise", 
        action=argparse.BooleanOptionalAction,
        default=True, 
        help="Generate a spatial noise dataset as well"
    )
    parser.add_argument(
        "--generate_spectral_noise", 
        action=argparse.BooleanOptionalAction,
        default=True, 
        help="Generate a spectral noise dataset as well"
    )

    args = parser.parse_args()

    init_globals(args)
    generate_parallel()