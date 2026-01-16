import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import tomllib

# Import your project modules
# Adjust these paths if your folder structure is different
try:
    from simple.burgers_1d.burgers import Burgers1DSimple
    from simple.model_factory import create_model
except ImportError:
    sys.path.append(".")
    from simple.burgers_1d.burgers import Burgers1DSimple
    from simple.model_factory import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_mapping_plot(idx, u0_tensor, true_tensor, pred_tensor, dx, output_dir):
    """
    Plots the 1D mapping: u(x, 0) -> u(x, 2.0).
    Args:
        u0_tensor: (X,) Initial condition
        true_tensor: (X,) Ground truth at t=2.0
        pred_tensor: (X,) Prediction at t=2.0
    """
    u0 = u0_tensor.cpu().numpy().flatten()
    u_true = true_tensor.cpu().numpy().flatten()
    u_pred = pred_tensor.cpu().numpy().flatten()
    
    nx = len(u0)
    # Reconstruct x grid (assuming domain starts at 0)
    x_axis = np.linspace(0, nx * dx, nx)
    
    plt.figure(figsize=(10, 6))
    
    # 1. Plot Initial Condition (Input)
    plt.plot(x_axis, u0, 'k:', linewidth=1.5, label=r'Input $u(x, t=0)$', alpha=0.5)
    
    # 2. Plot Ground Truth (Target)
    plt.plot(x_axis, u_true, 'b-', linewidth=2.0, label=r'True $u(x, t=2.0)$', alpha=0.7)
    
    # 3. Plot Prediction
    plt.plot(x_axis, u_pred, 'r--', linewidth=2.0, label=r'Pred $u(x, t=2.0)$', alpha=0.9)
    
    # 4. Highlight Error
    plt.fill_between(x_axis, u_true, u_pred, color='red', alpha=0.1, label='Error')

    plt.title(f"Burgers 1D Mapping: Sample {idx}")
    plt.xlabel("x (Space)")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_path = Path(output_dir) / f"sample_{idx:03d}_mapping.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate 1D Burgers Mapping Model")
    
    parser.add_argument("--model_config", type=str, default="simple/burgers_1d/fno_model.toml")
    parser.add_argument("--data_config", type=str, default="simple/burgers_1d/data.toml")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to .h5 data file")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--plot_samples", type=int, default=10, help="Number of plots to generate")
    
    args = parser.parse_args()
    
    # --- 1. Load Configurations ---
    if not os.path.exists(args.model_config):
        print(f"Error: Model config '{args.model_config}' not found.")
        return

    with open(args.model_config, "rb") as f:
        model_conf = tomllib.load(f)["config"]
    
    with open(args.data_config, "rb") as f:
        data_conf = tomllib.load(f)["config"]

    # --- 2. Setup Output ---
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # --- 3. Load Data ---
    print(f"Loading data from {args.data_path}...")
    problem = Burgers1DSimple(data_conf)
    
    # Load data (Assuming it returns tensors for Initial Cond and Final Cond)
    # x_test shape might be (N, X, 2) or (N, 2, X) depending on previous steps
    x_test, y_test = problem.load_burgers_data_for_path(args.data_path, include_grid=True) 
    
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=model_conf['batch_size'])
    
    # --- 4. Load Model ---
    print("Initializing Model...")
    model = create_model(model_conf, DEVICE)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        weights = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(weights)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    l2_errors = []
    plotted_count = 0
    
    print("Starting Evaluation Loop...")
    
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
                            
            # Forward Pass
            pred = model(x_batch)
            
            # Ensure Output is (Batch, Grid)
            # FNO output might be (Batch, 1, Grid) or (Batch, Grid)
            if pred.dim() == 3:
                pred = pred.squeeze(1)
            if y_batch.dim() == 3:
                y_batch = y_batch.squeeze(1)
            
            # --- Metrics (Relative L2) ---
            diff_norms = torch.norm(pred - y_batch, p=2, dim=1)
            y_norms = torch.norm(y_batch, p=2, dim=1)
            rel_l2 = diff_norms / y_norms
            l2_errors.append(rel_l2.cpu())
            
            # --- Plotting ---
            if plotted_count < args.plot_samples:
                for b in range(x_batch.shape[0]):
                    if plotted_count >= args.plot_samples: break
                    
                    # Extract u0 (Initial Condition) for plotting
                    # We need to find the correct dimension based on the input shape
                    if x_batch.shape[-1] < x_batch.shape[1]: # (B, Grid, C)
                        # Channel 0 is u0
                        u0_sample = x_batch[b, :, 0]
                    else: # (B, C, Grid)
                        u0_sample = x_batch[b, 0, :]
                        
                    save_mapping_plot(
                        plotted_count, 
                        u0_sample, 
                        y_batch[b], 
                        pred[b], 
                        problem.dx, 
                        args.output_dir
                    )
                    plotted_count += 1

    # --- Summary ---
    all_l2 = torch.cat(l2_errors).numpy()
    mean_l2 = np.mean(all_l2)
    max_l2 = np.max(all_l2)
    min_l2 = np.min(all_l2)

    summary = (
        f"Evaluation Complete.\n"
        f"---------------------\n"
        f"Mean Relative L2: {mean_l2:.4f}\n"
        f"Min Relative L2:  {min_l2:.4f}\n"
        f"Max Relative L2:  {max_l2:.4f}\n"
        f"Plots saved to:   {args.output_dir}/\n"
    )
    
    print("\n" + summary)
    
    with open(f"{args.output_dir}/summary.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    evaluate()