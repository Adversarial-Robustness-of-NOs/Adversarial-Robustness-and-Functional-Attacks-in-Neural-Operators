from pathlib import Path
import sys
import tomllib
import torch
import torch.nn as nn
import time
import argparse
import pandas as pd
from temporal.ns_2d.ns import NavierStokes2D
from models.model_factory import create_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_MODEL_PATH = "best_model.pth"
    
def init_globals(args):
    global OUTPUT_MODEL_PATH
    OUTPUT_MODEL_PATH = args.output_model

def get_config_by_name(name, data_config):
    if name == "temporal_ns_2d":
        return NavierStokes2D(data_config)
    return None

# --- TRAINING FUNCTION ---
def train_model(problem, model_conf, model_save_path="best_model.pth"):
    print(f"\n=== Training Model: {model_conf['model_type']} on {DEVICE} ===")
    
    train_loader, test_loader = problem.load_training_data(model_conf['batch_size'], include_grid=True)
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
        
        batch = 1
        for x, y in train_loader:
            ##print(f"Procecessing batch {batch}")
            batch += 1
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # --- Autoregressive Unrolling ---
            loss_data_batch = torch.tensor(0.0, device=DEVICE)
            loss_pde_batch = torch.tensor(0.0, device=DEVICE)   
            loss_bc_batch = torch.tensor(0.0, device=DEVICE)
            
            curr_x = x  # Initialize input (Batch, T_in, X, Y)
            
            # Loop through target steps (T_out)
            steps_to_predict = y.shape[1]
            
            for t in range(steps_to_predict):
                # 1. Predict next step
                pred = model(curr_x) # Output (Batch, 1, X, Y)
                
                # 2. Get corresponding ground truth
                target_t = y[:, t:t+1, ...] # (Batch, 1, X, Y)
                
                # 3. Accumulate Data Loss
                loss_data_batch += l1_loss_fn(pred, target_t)
                
                # 4. Accumulate PDE Loss
                if problem.has_pde_loss():
                    loss_pde_batch += w_pde * problem.pde_loss(pred, target_t, curr_x)

                if problem.has_bc_loss():
                    loss_bc_batch += w_bc * problem.bc_loss(pred, y, x)
                
                # 5. Update input for next iteration (Shift window)
                # Discard oldest frame, append new prediction
                # curr_x: [t_0, t_1, ..., t_last] -> [t_1, ..., t_last, pred]
                curr_x = torch.cat([curr_x[:, 1:, ...], pred], dim=1)
            
            # Normalize loss by number of steps
            loss_data = loss_data_batch / steps_to_predict
            loss_pde = loss_pde_batch / steps_to_predict
            loss_bc = loss_bc_batch / steps_to_predict
                
            # Combine and Step
            loss = (w_data * loss_data) + loss_pde + loss_bc

            optimizer.zero_grad()
            loss.backward()
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

        # Validation (re-uses evaluate_model, which is mostly static, 
        # but Rel L2 is still valid metric for sequence fidelity)
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

                
    return model, history

def evaluate_model(model, loader, loss_fn, pde_loss=None):
    model.eval()
    total_mse = 0
    total_rel_l2 = 0
    total_pde_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Unrolled Evaluation
            curr_x = x
            preds = []
            steps = y.shape[1]
            for t in range(steps):
                pred = model(curr_x)
                preds.append(pred)
                curr_x = torch.cat([curr_x[:, 1:, ...], pred], dim=1)
            
            out = torch.cat(preds, dim=1) # (Batch, T_out, X, Y)
            
            total_mse += loss_fn(out, y).item()
            
            # Relative L2 Error
            diff_norm = torch.norm(out.reshape(out.shape[0], -1) - y.reshape(y.shape[0], -1), p=2, dim=1)
            y_norm = torch.norm(y.reshape(y.shape[0], -1), p=2, dim=1)
            total_rel_l2 += torch.mean(diff_norm / y_norm).item()

            if pde_loss is not None:
                total_pde_loss += pde_loss(out, y, x)

    n = len(loader)
    return total_mse/n, total_rel_l2/n, total_pde_loss/n

if __name__ == "__main__":
    # Same arg parsing as before...
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="temporal_ns_2d") # Default changed to NS
    parser.add_argument("--model_config", type=str, default="temporal/ns_2d/fno_model.toml")
    parser.add_argument("--data_config", type=str, default="temporal/ns_2d/data.toml")
    parser.add_argument("--output_model", type=str, default="best_model.pth")
    args = parser.parse_args()

    # Load Configs
    try:
        with open(args.model_config, "rb") as f:
            model_config = tomllib.load(f)["config"]
        with open(args.data_config, "rb") as f:
            data_config = tomllib.load(f)["config"]
    except FileNotFoundError:
        print("Config not found.")
        sys.exit(1)

    init_globals(args)
    problem = get_config_by_name(args.problem, data_config)
    problem.print_problem(model_config)    
    
    path = Path(OUTPUT_MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    _, history = train_model(problem, model_config, str(path))

    df = pd.DataFrame(history)

    # Save to CSV
    log_path = path.with_suffix('.csv')
    df.to_csv(log_path, index_label='epoch')

    print(f"Log saved to {log_path}")