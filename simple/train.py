from pathlib import Path
import sys
import tomllib
import torch
import torch.nn as nn
import time
import argparse
import pandas as pd
from simple.burgers_1d.burgers import Burgers1DSimple
from simple.darcy_2d.darcy import Darcy2DSimple
from models.model_factory import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_MODEL_PATH = "best_model.pth"
    
def init_globals(args):
    global OUTPUT_MODEL_PATH
    OUTPUT_MODEL_PATH = args.output_model

def get_config_by_name(name, data_config):
    if name == "simple_burgers_1d":
        return Burgers1DSimple(data_config)
    if name == "simple_darcy_2d":
        return Darcy2DSimple(data_config)
    return None

# --- 4. TRAINING FUNCTION ---
def train_model(problem, model_conf, model_save_path="best_model.pth"):
    print(f"\n=== Training Model: {model_conf['model_type']} ===")
    
    train_loader, test_loader = problem.load_training_data(model_conf['batch_size'], include_grid=True)

    # Initialize 2D FNO (Maps Space-Time Input -> Space-Time Output)
    model = create_model(model_conf, DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_conf['learning_rate'], weight_decay=model_conf['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=model_conf['step_size'], gamma=model_conf['gamma'])
    l1_loss_fn = nn.L1Loss()
    l2_loss_fn = nn.MSELoss()
    
    history = {'epoch': [], 'time': [], 'learning_rate': [], 'total': [], 'data': [], 'pde': [], 'bc': [], 'val_mse':[], 'val_rel_l2': [], 'val_pde': []}
    
    start_time = time.time()
    
    best_val_l2 = float('inf')

    w_data = 1.0
    w_pde = model_conf.get('physics_importance', 0.0)
    w_bc = model_conf.get('boundary_condition_weight', 0.0)

    for epoch in range(model_conf['epochs']):
        model.train()
        train_loss = 0
        train_pde = 0
        train_data = 0
        train_bc = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
    
            # Forward
            out = model(x)       
            # Data Loss
            loss_data = l1_loss_fn(out, y)
            loss_pde = torch.tensor(0.0, dtype=torch.float32)
            loss_bc = torch.tensor(0.0, dtype=torch.float32)
            if problem.has_pde_loss():
                loss_pde = w_pde * problem.pde_loss(out, y, x)            
            if problem.has_bc_loss():
                loss_bc = w_bc * problem.bc_loss(out, y, x)

            loss = (w_data * loss_data) + loss_pde + loss_bc

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            
            train_loss += loss.item()
            train_data += loss_data.item()
            train_pde += loss_pde.item()
            train_bc += loss_bc.item()
            
        scheduler.step()
        
        avg_loss = train_loss / len(train_loader)
        avg_data = train_data / len(train_loader)
        avg_pde = train_pde / len(train_loader)
        avg_bc = train_bc / len(train_loader)


        val_mse, val_rel_l2, val_pde = evaluate_model(model, test_loader, l2_loss_fn, problem.pde_loss if problem.has_pde_loss() else None)
        
        history['epoch'].append(epoch)
        history['time'].append(time.time()-start_time)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        history['total'].append(avg_loss)
        history['data'].append(avg_data)
        history['pde'].append(avg_pde)
        history['bc'].append(avg_bc)
        history['val_mse'].append(val_mse)
        history['val_rel_l2'].append(val_rel_l2)
        history['val_pde'].append(val_pde)

        if val_rel_l2 < best_val_l2:
            best_val_l2 = val_rel_l2
            print(f"New Best Model! (L2: {val_rel_l2:.4f}) Saving to {model_save_path}...")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_rel_l2,
                'config': model_conf,
            }, model_save_path)

        print(f"Ep {epoch+1:03d} | Learning rate: {scheduler.get_last_lr()[0]:.8f} | Train Loss: {avg_loss:.4f} | Data: {avg_data:.5f} | PDE: {avg_pde:.5f} | Val MSE: {val_mse:.4f} | Val Rel L2: {val_rel_l2:.4f} | Val PDE: {val_pde:.4f}")
                
    print(f"Training finished in {time.time()-start_time:.1f}s")
    return model, history

def evaluate_model(model, loader, loss_fn, pde_loss=None):
    """
    Computes validation metrics with OUTLIER REMOVAL:
    1. MSE Loss (Data fidelity)
    2. Relative L2 Error (Standard SciML metric)
    3. PDE loss (if provided)
    """
    model.eval()
    total_mse = 0
    total_rel_l2 = 0
    total_pde_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            
            # 1. Data Fidelity (MSE)
            total_mse += loss_fn(out, y).item()
            
            # 2. Relative L2 Error
            diff_norm = torch.norm(out.reshape(out.shape[0], -1) - y.reshape(y.shape[0], -1), p=2, dim=1)
            y_norm = torch.norm(y.reshape(y.shape[0], -1), p=2, dim=1)
            total_rel_l2 += torch.mean(diff_norm / y_norm).item()

            if pde_loss is not None:
                total_pde_loss += pde_loss(out, y, x)
                                                
    n_batches = len(loader)
    
    return total_mse/n_batches, total_rel_l2/n_batches, total_pde_loss/n_batches
    
if __name__ == "__main__":
    print(f"Running on: {DEVICE}")

    parser = argparse.ArgumentParser(description="Train Model")
    # -- The Config File Argument --
    parser.add_argument("--problem", type=str, default="simple_burgers_1d", help="Model string")
    parser.add_argument("--model_config", type=str, default="simple/burgers_1d/fno_model.toml", help="Path to TOML model config")
    parser.add_argument("--data_config", type=str, default="simple/burgers_1d/data.toml", help="Path to TOML data config")

    # -- Paths --
    parser.add_argument("--output_model", type=str, default="best_model.pth")
    
    # Parse CLI arguments
    args = parser.parse_args()

    # Load TOML file
    try:
        with open(args.model_config, "rb") as f:
            model_config = tomllib.load(f)["config"]
        with open(args.data_config, "rb") as f:
            data_config = tomllib.load(f)["config"]
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    # Initialize Globals
    init_globals(args)
    problem = get_config_by_name(args.problem, data_config)
    problem.print_problem(model_config)    
    
    path = Path(OUTPUT_MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    # A. Train Standard FNO
    hist_model, fno_hist = train_model(problem, model_config, path)

    df = pd.DataFrame(fno_hist)

    # Save to CSV
    log_path = path.with_suffix('.csv')
    df.to_csv(log_path, index_label='epoch')

    print(f"Log saved to {log_path}")