import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIG ---
DATA_PATH = "./ns_data.h5"
OUTPUT_DIR = "./vis_output"
N_SAMPLES_TO_PLOT = 10  # Number of different simulations to check
FRAMES_TO_PLOT = 6     # Number of time steps per simulation

def visualize_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File {DATA_PATH} not found. Run generate_ns_data.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading {DATA_PATH}...")
    with h5py.File(DATA_PATH, 'r') as f:
        # Shape is likely (N_Samples, T_Steps, X, Y)
        data = f['u'][:]
        print(f"Dataset Shape: {data.shape}")

        #if data.shape[-1] < data.shape[1]: # Heuristic: Time usually < Resolution for these files
        print("Transposing from (N, X, Y, T) to (N, T, X, Y)...")
        data = np.transpose(data, (0, 3, 1, 2))

        # Determine global min/max for consistent coloring
        # (This helps you see if the simulation is decaying or exploding)
        g_min, g_max = data.min(), data.max()
        limit = max(abs(g_min), abs(g_max))
        print(f"Vorticity Range: [{g_min:.4f}, {g_max:.4f}]")

        for i in range(N_SAMPLES_TO_PLOT):
            fig, axes = plt.subplots(1, FRAMES_TO_PLOT, figsize=(15, 3))
            
            # Select evenly spaced frames from the trajectory
            total_steps = data.shape[1]
            indices = np.linspace(0, total_steps-1, FRAMES_TO_PLOT, dtype=int)
            
            for ax_idx, t in enumerate(indices):
                field = data[i, t]
                
                # Plot
                #im = axes[ax_idx].imshow(field, cmap='seismic', vmin=-limit, vmax=limit, origin='lower')
                im = axes[ax_idx].imshow(field, cmap='seismic', vmin=-limit, vmax=limit, origin='lower', aspect='equal')
                axes[ax_idx].set_title(f"t={t}")
                axes[ax_idx].axis('off')
            
            # Add colorbar
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
            
            save_path = f"{OUTPUT_DIR}/sim_{i:03d}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved {save_path}")

if __name__ == "__main__":
    visualize_dataset()