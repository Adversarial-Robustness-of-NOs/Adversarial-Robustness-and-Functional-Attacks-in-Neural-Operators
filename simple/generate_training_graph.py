import matplotlib.pyplot as plt
import pandas as pd
import tomllib  # Built-in for Python 3.11+
from pathlib import Path

# List of dictionaries, each containing the path to the TOML and the CSV
EXPERIMENTS = [
    {
        "toml": "simple/darcy_2d/fno_model.toml", 
        "csv": "simple/darcy_2d/trained_models/best_fno_model.csv",
        "label": None  # Optional: Override the title from TOML
    },
    {
        "toml": "simple/darcy_2d/ffno_model.toml", 
        "csv": "simple/darcy_2d/trained_models/best_ffno_model.csv",
        "label": None  # Optional: Override the title from TOML
    },
    {
        "toml": "simple/darcy_2d/cno_model.toml", 
        "csv": "simple/darcy_2d/trained_models/best_cno_model.csv",
        "label": None  # Optional: Override the title from TOML
    },
]

def plot_training_comparison(experiments):
    plt.figure(figsize=(12, 7))
    
    # Iterate through each experiment definition
    for i, exp in enumerate(experiments):
        toml_path = Path(exp['toml'])
        csv_path = Path(exp['csv'])
        
        if not toml_path.exists() or not csv_path.exists():
            print(f"Skipping {toml_path} or {csv_path}: File not found.")
            continue

        # 1. Read Configuration (TOML)
        with open(toml_path, "rb") as f:
            config_data = tomllib.load(f)
        
        # Extract weights and metadata
        # Support both 'config' sub-dictionary or flat structure
        conf = config_data.get('config', config_data)
        
        model_name = exp.get('label') or config_data.get('title', f"Model {i+1}")
        w_physics = conf.get('physics_importance', 0.0)
        w_bc = conf.get('boundary_condition_weight', 0.0)

        # 2. Read Logs (CSV)
        df = pd.read_csv(csv_path)
        
        # 3. Plotting
        # Main Metric: Relative L2 Loss (Solid Line)
        # We try 'val_rel_l2' first, then 'val_mse', then 'data'
        if 'val_rel_l2' in df.columns:
            l2_col = 'val_rel_l2'
            l2_label = "Val L2"
        elif 'val_mse' in df.columns:
            l2_col = 'val_mse'
            l2_label = "Val MSE"
        else:
            l2_col = 'data'
            l2_label = "Data Loss"
            
        # Plot the main line
        p = plt.plot(df['epoch'], df[l2_col], linewidth=2.5, label=f"{model_name} ({l2_label})")
        base_color = p[0].get_color() # Get color to match sub-losses
        
        # Conditional: Physics Loss (Dashed Line)
        if w_physics > 0:
            # Prefer validation PDE loss if available, else training PDE loss
            pde_col = 'val_pde' if 'val_pde' in df.columns and df['val_pde'].mean() > 0 else 'pde'
            
            if pde_col in df.columns:
                plt.plot(df['epoch'], df[pde_col], 
                         linestyle='--', linewidth=1.5, alpha=0.8, color=base_color,
                         label=f"{model_name} (PDE Loss)")

        # Conditional: Boundary Condition Loss (Dotted Line)
        if w_bc > 0:
            if 'bc' in df.columns:
                plt.plot(df['epoch'], df['bc'], 
                         linestyle=':', linewidth=1.5, alpha=0.8, color=base_color,
                         label=f"{model_name} (BC Loss)")

    # Formatting
    plt.yscale('log') # Log scale is usually best for loss curves
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.title('Training Dynamics Comparison', fontsize=14)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Legend outside to prevent blocking
    plt.tight_layout()
    
    output_file = "training_comparison.png"
    plt.savefig(output_file, dpi=300)
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    plot_training_comparison(EXPERIMENTS)