import os
import glob
from pathlib import Path
import subprocess
import json
import argparse
import tomllib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py

# === 1. CONFIGURATION ===
MODELS = {
    "fno":  "temporal/ns_2d/fno_attack_config.toml",
    "ffno": "temporal/ns_2d/ffno_attack_config.toml",
    "cno":  "temporal/ns_2d/cno_attack_config.toml",
}

CROSS_RES_TYPES = ["attack_spatial", "attack_spectral"]

# === 2. CONFIG PARSING HELPERS ===

def parse_configs(attack_config_path):
    if not os.path.exists(attack_config_path):
        raise FileNotFoundError(f"Attack config not found: {attack_config_path}")

    with open(attack_config_path, "rb") as f:
        atk_cfg = tomllib.load(f)
    
    info = {
        "output_dir": atk_cfg['general']['output_dir'],
        "model_conf": atk_cfg['model']['model_config_path'],
        "model_path": atk_cfg['model']['model_path'],
        "data_conf": atk_cfg['model']['data_config_path']
    }
    
    if os.path.exists(info["data_conf"]):
        with open(info["data_conf"], "rb") as f:
            data_cfg = tomllib.load(f)
        
        info["low_res"] = data_cfg.get('config', {}).get("low_res_data_path")
        info["high_res"] = data_cfg.get('config', {}).get("high_res_data_path")
        info["base_data_path"] = data_cfg.get('config', {}).get("data_path")
        info["base_res"] = data_cfg.get('config', {}).get("resolution", 64)
    else:
        info["low_res"] = None
        info["high_res"] = None
        info["base_data_path"] = None
        info["base_res"] = 64
        
    return info

# === 3. EXECUTION FUNCTIONS ===

def run_attacks(attack_config_path):
    print(f"   -> Generating Attacks...")
    subprocess.run([
        "python", "-m", "temporal.ns_2d.attack",
        "--attack_config", attack_config_path
    ], check=True)


def evaluate_folder(model_name, info):
    attack_dir = info["output_dir"]
    print(f"   -> Evaluating in: {attack_dir}")
    
    h5_files = glob.glob(os.path.join(attack_dir, "*.h5"))
    if not h5_files:
        print(f"      [WARNING] No .h5 files found.")
        return

    for h5 in h5_files:
        basename = os.path.basename(h5).replace(".h5", "")
        
        # Standard Evaluation
        std_out = os.path.join(attack_dir, "eval_results", basename, "standard")
        subprocess.run([
            "python", "-m", "temporal.ns_2d.evaluate",
            "--model_config", info["model_conf"],
            "--model_path", info["model_path"],
            "--data_path", h5,
            "--output_dir", std_out,
            "--num_samples_to_plot", "5"
        ], check=True)
        
        # Cross-Resolution Evaluation for PGD pure attacks
        is_cross_resolution = any(t in basename for t in CROSS_RES_TYPES) and "stealth" not in basename.lower()
        
        if is_cross_resolution:
            # Downsample (Low Res)
            if info.get("low_res") and os.path.exists(info["low_res"]):
                ds_out = os.path.join(attack_dir, "eval_results", basename, "cross_res_low")
                try:
                    subprocess.run([
                        "python", "-m", "temporal.ns_2d.evaluate",
                        "--model_config", info["model_conf"],
                        "--model_path", info["model_path"],
                        "--data_path", h5,
                        "--ref_data_path", info["low_res"],
                        "--output_dir", ds_out,
                        "--num_samples_to_plot", "0"
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"      [WARNING] Low-res evaluation failed: {e}")
            
            # Upsample (High Res)
            if info.get("high_res") and os.path.exists(info["high_res"]):
                us_out = os.path.join(attack_dir, "eval_results", basename, "cross_res_high")
                try:
                    subprocess.run([
                        "python", "-m", "temporal.ns_2d.evaluate",
                        "--model_config", info["model_conf"],
                        "--model_path", info["model_path"],
                        "--data_path", h5,
                        "--ref_data_path", info["high_res"],
                        "--output_dir", us_out,
                        "--num_samples_to_plot", "0"
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"      [WARNING] High-res evaluation failed: {e}")


def evaluate_clean(model_name, info):
    """Evaluate model on clean (non-adversarial) data as baseline."""
    attack_dir = info["output_dir"]
    clean_data_path = info.get("base_data_path")
    
    if not clean_data_path or not os.path.exists(clean_data_path):
        print(f"   [SKIP] No clean data path for {model_name}")
        return
    
    print(f"   -> Evaluating clean baseline: {clean_data_path}")
    
    clean_out = os.path.join(attack_dir, "eval_results", "clean_baseline", "standard")
    
    try:
        subprocess.run([
            "python", "-m", "temporal.ns_2d.evaluate",
            "--model_config", info["model_conf"],
            "--model_path", info["model_path"],
            "--data_path", clean_data_path,
            "--output_dir", clean_out,
            "--num_samples_to_plot", "0"
        ], check=True)
        print(f"      [OK] Clean baseline evaluated")
    except subprocess.CalledProcessError as e:
        print(f"      [WARNING] Clean evaluation failed: {e}")


# === 4. AGGREGATION ===

def classify_attack(attack_name):
    name = attack_name.lower()
    if "clean" in name or "baseline" in name:
        return "Clean"
    elif "noise" in name:
        return "Noise"
    elif "sequential" in name:
        return "Sequential"
    elif "stealth" in name:
        return "PGD-Stealth"
    elif "pgd" in name or "spatial" in name or "spectral" in name:
        return "PGD-Pure"
    else:
        return "Other"


def get_domain(attack_name):
    name = attack_name.lower()
    if "spectral" in name:
        return "Spectral"
    elif "spatial" in name:
        return "Spatial"
    else:
        return "Unknown"


def parse_attack_params(attack_name):
    """Extract epsilon, std, pde weight from attack name."""
    name = attack_name.lower().replace("attack_", "").replace(".h5", "")
    
    epsilon = None
    target_l2 = None
    pde_weight = 0.0
    
    if "_eps" in name:
        try:
            epsilon = float(name.split("_eps")[1].split("_")[0])
        except:
            pass
    
    if "_std" in name:
        try:
            target_l2 = float(name.split("_std")[1].split("_")[0])
        except:
            pass
    
    if "target_l2_" in name:
        try:
            target_l2 = float(name.split("target_l2_")[1].split("_")[0])
        except:
            pass
            
    if "_pde" in name:
        try:
            pde_weight = float(name.split("_pde")[1].split("_")[0])
        except:
            pass

    if "_l2_" in name and target_l2 is None:
        try:
            l2_part = name.split("_l2_")[1]
            l2_value = l2_part.split("_")[0] if "_" in l2_part else l2_part
            target_l2 = float(l2_value)
        except:
            pass
    
    return epsilon, target_l2, pde_weight


def aggregate_stats():
    print("\n[BENCHMARK] Aggregating Metrics...")
    results = []
    temporal_data = []
    
    search_dirs = []
    for m_name, cfg_path in MODELS.items():
        try:
            info = parse_configs(cfg_path)
            search_dirs.append((m_name, info["output_dir"]))
        except Exception as e:
            print(f"Skipping {m_name}: {e}")

    for model_name, base_dir in search_dirs:
        json_files = glob.glob(f"{base_dir}/**/metrics.json", recursive=True)
        
        for jf in json_files:
            try:
                parts = jf.split(os.sep)
                attack_name = parts[-3]
                mode = parts[-2]
                
                with open(jf, 'r') as f:
                    m = json.load(f)
                
                # Try to read attack parameters from h5 file first
                h5_path = os.path.join(base_dir, attack_name + ".h5")
                h5_params = {}
                if os.path.exists(h5_path):
                    try:
                        with h5py.File(h5_path, 'r') as hf:
                            for key in ['epsilon', 'target_l2', 'lambda_pde', 'lambda_bc', 'attack_type', 'attack_domain']:
                                if key in hf.attrs:
                                    h5_params[key] = hf.attrs[key]
                    except Exception:
                        pass
                
                # Fall back to parsing from filename
                epsilon, target_l2, pde_weight = parse_attack_params(attack_name)
                
                # Override with h5 params if available
                if 'epsilon' in h5_params and h5_params['epsilon'] is not None:
                    epsilon = float(h5_params['epsilon'])
                if 'target_l2' in h5_params and h5_params['target_l2'] is not None:
                    target_l2 = float(h5_params['target_l2'])
                if 'lambda_pde' in h5_params:
                    pde_weight = float(h5_params['lambda_pde'])
                
                ic_pert = m.get('ic_perturbation', {})
                global_m = m.get('global_metrics', {})
                freq = m.get('frequency_analysis', {})
                temp = m.get('temporal_evolution', {})
                
                category = classify_attack(attack_name)
                domain = get_domain(attack_name)
                
                # Debug noise and PGD data
                ic_rel = ic_pert.get('rel_l2')
                adv_l2 = global_m.get('adv_st_l2')
                ic_rel_str = f"{ic_rel:.4f}" if ic_rel is not None else "N/A"
                adv_l2_str = f"{adv_l2:.4f}" if adv_l2 is not None else "N/A"
                
                if category == 'Noise':
                    print(f"   DEBUG Noise: {attack_name} | Domain={domain} | IC_Pert_Rel={ic_rel_str} | Adv_L2={adv_l2_str}")
                elif category in ['PGD-Pure', 'PGD-Stealth'] and mode == 'standard':
                    print(f"   DEBUG PGD: {attack_name} | Domain={domain} | IC_Pert_Rel={ic_rel_str} | Eps={epsilon}")
                
                entry = {
                    'Model': model_name.upper(),
                    'Mode': mode,
                    'Attack_Name': attack_name,
                    'Category': category,
                    'Domain': domain,
                    'Epsilon': epsilon,
                    'Target_L2': target_l2,
                    'Lambda_PDE': pde_weight,
                    # IC Perturbation
                    'IC_Pert_Abs': ic_pert.get('abs_l2', np.nan),
                    'IC_Pert_Rel': ic_pert.get('rel_l2', np.nan),
                    'IC_Low_Freq': ic_pert.get('low_freq', np.nan),
                    'IC_High_Freq': ic_pert.get('high_freq', np.nan),
                    # Global metrics
                    'Clean_L2': global_m.get('clean_st_l2', np.nan),
                    'Adv_L2': global_m.get('adv_st_l2', np.nan),
                    'Clean_MSE': global_m.get('clean_st_mse', np.nan),
                    'Adv_MSE': global_m.get('adv_st_mse', np.nan),
                    # Frequency analysis
                    'Clean_Low_Freq': freq.get('clean_low_freq', np.nan),
                    'Clean_High_Freq': freq.get('clean_high_freq', np.nan),
                    'Adv_Low_Freq': freq.get('adv_low_freq', np.nan),
                    'Adv_High_Freq': freq.get('adv_high_freq', np.nan),
                }
                
                # Amplification ratio
                if pd.notna(entry['IC_Pert_Rel']) and entry['IC_Pert_Rel'] > 1e-9:
                    entry['Amp_Ratio'] = entry['Adv_L2'] / entry['IC_Pert_Rel']
                else:
                    entry['Amp_Ratio'] = np.nan
                
                results.append(entry)
                
                # Temporal data
                if mode == "standard" and 'drift_l2_rel' in temp:
                    drift = temp['drift_l2_rel']
                    injection = temp.get('injection_l2', [np.nan] * len(drift))
                    clean_phys = temp.get('clean_physics', [np.nan] * len(drift))
                    adv_phys = temp.get('adv_physics', [np.nan] * len(drift))
                    # NEW: Per-step L2 errors vs ground truth
                    clean_err = temp.get('clean_error_l2_rel', [np.nan] * len(drift))
                    adv_err = temp.get('adv_error_l2_rel', [np.nan] * len(drift))
                    
                    for t, (d, inj, cp, ap, ce, ae) in enumerate(zip(drift, injection, clean_phys, adv_phys, clean_err, adv_err)):
                        temporal_data.append({
                            'Model': model_name.upper(),
                            'Attack_Name': attack_name,
                            'Category': classify_attack(attack_name),
                            'Domain': get_domain(attack_name),
                            'Epsilon': epsilon,
                            'Time_Step': t,
                            'Drift': d,
                            'Injection': inj,
                            'Clean_Physics': cp,
                            'Adv_Physics': ap,
                            'Clean_Error': ce,
                            'Adv_Error': ae,
                        })
                        
            except Exception as e:
                print(f"   [WARNING] Failed to parse {jf}: {e}")

    return pd.DataFrame(results), pd.DataFrame(temporal_data)


# === 5. PLOTTING FUNCTIONS ===

def plot_clean_baseline(df, plot_dir):
    """Plot 1: Clean Baseline Comparison"""
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty:
        print("   [SKIP] No data for clean baseline plot.")
        return
    
    clean_df = std_df[std_df['Category'] == 'Clean']
    if clean_df.empty:
        print("   [SKIP] No clean baseline data found.")
        return
    
    models = sorted(std_df['Model'].unique())
    clean_baseline = clean_df.groupby('Model')['Clean_L2'].mean()
    
    attack_df = std_df[std_df['Category'] != 'Clean']
    categories = sorted(attack_df['Category'].unique())
    
    category_colors = {
        'Clean': '#2E7D32',
        'Noise': '#808080',
        'PGD-Pure': '#E07A5F',
        'PGD-Stealth': '#3D7C98',
        'Sequential': '#81B29A',
        'Other': '#F4A261'
    }
    
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 8))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Output Error: Clean Baseline vs Adversarial Attacks', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_attack = attack_df[attack_df['Model'] == model]
        model_clean = clean_baseline.get(model, np.nan)
        
        cat_errors = model_attack.groupby('Category')['Adv_L2'].agg(['mean', 'std'])
        
        all_cats = ['Clean'] + list(cat_errors.index)
        n_cats = len(all_cats)
        x = np.arange(n_cats)
        
        means = [model_clean] + list(cat_errors['mean'])
        stds = [0] + list(cat_errors['std'].fillna(0))
        colors = [category_colors.get(c, '#808080') for c in all_cats]
        
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5, edgecolor='white')
        
        if pd.notna(model_clean):
            ax.axhline(y=model_clean, color='#2E7D32', linestyle='--', linewidth=2, alpha=0.7)
        
        for bar, val in zip(bars, means):
            if pd.notna(val) and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + max([m for m in means if pd.notna(m)])*0.02,
                       f'{val:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_cats, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Output Rel L2 Error', fontsize=11)
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "1_clean_baseline.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 1_clean_baseline.png")
    
    # Save CSV
    csv_data = []
    for model in models:
        model_clean = clean_baseline.get(model, np.nan)
        csv_data.append({'Model': model, 'Category': 'Clean', 'Mean_Error': model_clean, 'Std_Error': 0})
        
        model_attack = attack_df[attack_df['Model'] == model]
        for cat in categories:
            cat_data = model_attack[model_attack['Category'] == cat]
            if not cat_data.empty:
                csv_data.append({
                    'Model': model,
                    'Category': cat,
                    'Mean_Error': cat_data['Adv_L2'].mean(),
                    'Std_Error': cat_data['Adv_L2'].std()
                })
    
    pd.DataFrame(csv_data).to_csv(os.path.join(plot_dir, "1_clean_baseline.csv"), index=False)
    print(f"   [SAVED] 1_clean_baseline.csv")


def plot_pgd_robustness(df, plot_dir):
    """Plot 2: PGD Robustness - Two rows per domain:
      Row 1: By Epsilon
      Row 2: By Input Perturbation Magnitude
    """
    std_df = df[(df['Mode'] == 'standard') & (df['Category'].isin(['PGD-Pure', 'PGD-Stealth']))].copy()
    clean_df = df[(df['Mode'] == 'standard') & (df['Category'] == 'Clean')]
    noise_df = df[(df['Mode'] == 'standard') & (df['Category'] == 'Noise')]
    
    if std_df.empty:
        print("   [SKIP] No PGD data.")
        return
    
    # Get clean baseline
    clean_baseline = {}
    for model in std_df['Model'].unique():
        model_clean = clean_df[clean_df['Model'] == model]
        if not model_clean.empty:
            clean_baseline[model] = model_clean['Clean_L2'].mean()
    
    colors = {'PGD-Pure': '#E07A5F', 'PGD-Stealth': '#3D7C98'}
    
    # Separate by domain
    for domain in ['Spatial', 'Spectral']:
        domain_df = std_df[std_df['Domain'] == domain]
        domain_noise = noise_df[noise_df['Domain'] == domain]
        
        # Debug domain filtering
        print(f"\n   DEBUG Domain={domain}:")
        print(f"     PGD data: {len(domain_df)} rows")
        print(f"     Noise data: {len(domain_noise)} rows")
        if not noise_df.empty:
            print(f"     All noise domains: {noise_df['Domain'].value_counts().to_dict()}")
            print(f"     Noise Target_L2 values: {noise_df['Target_L2'].unique().tolist()}")
        
        if domain_df.empty:
            print(f"     [SKIP] No PGD data for {domain}")
            continue
        
        models = sorted(domain_df['Model'].unique())
        fig, axes = plt.subplots(2, len(models), figsize=(7 * len(models), 12))
        if len(models) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'PGD {domain} Attack Analysis', fontsize=16, fontweight='bold')
        
        # --- Row 1: By Epsilon (Grouped Bar Chart) ---
        for idx, model in enumerate(models):
            ax = axes[0, idx]
            model_df = domain_df[domain_df['Model'] == model]
            model_noise = domain_noise[domain_noise['Model'] == model]
            
            # Debug
            print(f"   DEBUG: {model} {domain} - PGD rows: {len(model_df)}, Noise rows: {len(model_noise)}")
            
            if model_df['Epsilon'].notna().any():
                # Aggregate by epsilon and category
                summary = model_df.groupby(['Epsilon', 'Category'])['Adv_L2'].agg(['mean', 'std']).reset_index()
                summary.columns = ['Epsilon', 'Category', 'Mean', 'Std']
                
                epsilons = sorted(summary['Epsilon'].unique())
                x = np.arange(len(epsilons))
                width = 0.35
                
                for i, cat in enumerate(['PGD-Pure', 'PGD-Stealth']):
                    cat_data = summary[summary['Category'] == cat]
                    if not cat_data.empty:
                        vals = [cat_data[cat_data['Epsilon'] == e]['Mean'].values[0] if e in cat_data['Epsilon'].values else 0 for e in epsilons]
                        stds = [cat_data[cat_data['Epsilon'] == e]['Std'].values[0] if e in cat_data['Epsilon'].values else 0 for e in epsilons]
                        
                        offset = -width/2 if i == 0 else width/2
                        bars = ax.bar(x + offset, vals, width, 
                                     label=cat, color=colors[cat], alpha=0.8,
                                     yerr=stds, capsize=3)
                        
                        # Value labels
                        for bar, val in zip(bars, vals):
                            height = bar.get_height()
                            ax.annotate(f'{val:.2%}',
                                       xy=(bar.get_x() + bar.get_width()/2, height),
                                       xytext=(0, 3), textcoords='offset points',
                                       ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Add noise reference line(s)
                if not model_noise.empty:
                    # Group noise by Target_L2 and show as horizontal lines
                    noise_by_target = model_noise.groupby('Target_L2')['Adv_L2'].mean()
                    for target_l2, noise_val in noise_by_target.items():
                        ax.axhline(y=noise_val, color='#808080', linestyle=':', 
                                  linewidth=2, alpha=0.8, label=f'Noise (target={target_l2})')
                        # Annotate on right side
                        ax.annotate(f'Noise: {noise_val:.2%}', xy=(len(epsilons)-0.5, noise_val),
                                   fontsize=8, color='#808080', va='bottom')
                
                # Clean baseline
                if model in clean_baseline:
                    ax.axhline(y=clean_baseline[model], color='#2E7D32', linestyle='--', 
                              linewidth=2, alpha=0.7, label=f'Clean: {clean_baseline[model]:.2%}')
                
                ax.set_xticks(x)
                ax.set_xticklabels([f'ε={e}' for e in epsilons], fontsize=10)
                ax.set_ylabel('Output Rel L2 Error', fontsize=11)
                ax.set_title(f'{model}: By Epsilon', fontsize=13, fontweight='bold')
                
                # Clean up legend - remove duplicates
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=9)
                
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
            else:
                ax.text(0.5, 0.5, 'No epsilon data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model}: By Epsilon')
        
        # --- Row 2: By Input Perturbation (Bucketed) ---
        for idx, model in enumerate(models):
            ax = axes[1, idx]
            model_df = domain_df[domain_df['Model'] == model].copy()
            model_noise = domain_noise[domain_noise['Model'] == model].copy()
            
            # Debug noise data
            if not model_noise.empty:
                print(f"   DEBUG: {model} {domain} - Noise IC_Pert_Rel range: [{model_noise['IC_Pert_Rel'].min():.4f}, {model_noise['IC_Pert_Rel'].max():.4f}]")
            else:
                print(f"   DEBUG: {model} {domain} - No noise data found")
            
            if model_df['IC_Pert_Rel'].notna().any():
                # Create buckets from PGD data
                try:
                    model_df['Bucket'], bins = pd.qcut(model_df['IC_Pert_Rel'], q=5, labels=False, retbins=True, duplicates='drop')
                except ValueError:
                    model_df['Bucket'], bins = pd.cut(model_df['IC_Pert_Rel'], bins=5, labels=False, retbins=True)
                
                n_buckets = len(bins) - 1
                print(f"   DEBUG: {model} {domain} - PGD bins: {bins}")
                
                # Aggregate by bucket and category
                summary = model_df.groupby(['Bucket', 'Category'])['Adv_L2'].agg(['mean', 'std']).reset_index()
                
                # Width depends on whether we have noise data
                has_noise_in_range = False
                if not model_noise.empty and model_noise['IC_Pert_Rel'].notna().any():
                    noise_pert_range = (model_noise['IC_Pert_Rel'].min(), model_noise['IC_Pert_Rel'].max())
                    pgd_range = (bins[0], bins[-1])
                    # Check if there's overlap
                    has_noise_in_range = not (noise_pert_range[1] < pgd_range[0] or noise_pert_range[0] > pgd_range[1])
                    print(f"   DEBUG: {model} {domain} - Noise range {noise_pert_range} vs PGD range {pgd_range}, overlap={has_noise_in_range}")
                
                x = np.arange(n_buckets)
                n_categories = 3 if has_noise_in_range else 2  # PGD-Pure, PGD-Stealth, [Noise]
                width = 0.8 / n_categories
                
                # Plot bars for PGD categories
                for i, cat in enumerate(['PGD-Pure', 'PGD-Stealth']):
                    cat_data = summary[summary['Category'] == cat]
                    if not cat_data.empty:
                        vals = [cat_data[cat_data['Bucket'] == b]['mean'].values[0] if b in cat_data['Bucket'].values else 0 for b in range(n_buckets)]
                        stds = [cat_data[cat_data['Bucket'] == b]['std'].values[0] if b in cat_data['Bucket'].values else 0 for b in range(n_buckets)]
                        
                        offset = (i - n_categories/2 + 0.5) * width
                        bars = ax.bar(x + offset, vals, width,
                                     label=cat, color=colors[cat], alpha=0.8,
                                     yerr=stds, capsize=3)
                        
                        # Value labels
                        for bar, val in zip(bars, vals):
                            if val > 0:
                                height = bar.get_height()
                                ax.annotate(f'{val:.2%}',
                                           xy=(bar.get_x() + bar.get_width()/2, height),
                                           xytext=(0, 3), textcoords='offset points',
                                           ha='center', va='bottom', fontsize=8, rotation=45)
                
                # Add noise bars for matching buckets
                if has_noise_in_range:
                    noise_vals = []
                    noise_stds = []
                    for b_idx in range(n_buckets):
                        bin_low, bin_high = bins[b_idx], bins[b_idx + 1]
                        # More generous matching
                        matching_noise = model_noise[
                            (model_noise['IC_Pert_Rel'] >= bin_low * 0.8) & 
                            (model_noise['IC_Pert_Rel'] <= bin_high * 1.2)
                        ]
                        if not matching_noise.empty:
                            noise_vals.append(matching_noise['Adv_L2'].mean())
                            noise_stds.append(matching_noise['Adv_L2'].std())
                        else:
                            noise_vals.append(0)
                            noise_stds.append(0)
                    
                    # Plot noise as third set of bars
                    offset = (2 - n_categories/2 + 0.5) * width
                    bars = ax.bar(x + offset, noise_vals, width,
                                 label='Noise', color='#808080', alpha=0.8,
                                 yerr=noise_stds, capsize=3)
                    
                    for bar, val in zip(bars, noise_vals):
                        if val > 0:
                            height = bar.get_height()
                            ax.annotate(f'{val:.2%}',
                                       xy=(bar.get_x() + bar.get_width()/2, height),
                                       xytext=(0, 3), textcoords='offset points',
                                       ha='center', va='bottom', fontsize=8, rotation=45)
                
                # If noise doesn't fit in PGD range, show as horizontal line with note
                elif not model_noise.empty and model_noise['IC_Pert_Rel'].notna().any():
                    noise_avg = model_noise['Adv_L2'].mean()
                    noise_pert_avg = model_noise['IC_Pert_Rel'].mean() * 100
                    ax.axhline(y=noise_avg, color='#808080', linestyle='--', linewidth=2, 
                              label=f'Noise (IC={noise_pert_avg:.1f}%)')
                
                # Clean baseline
                if model in clean_baseline:
                    ax.axhline(y=clean_baseline[model], color='#2E7D32', linestyle=':', 
                              linewidth=2, alpha=0.7, label=f'Clean: {clean_baseline[model]:.2%}')
                
                # X-axis labels: show input perturbation ranges
                x_labels = []
                for i in range(n_buckets):
                    low, high = bins[i] * 100, bins[i+1] * 100
                    x_labels.append(f'{low:.1f}-{high:.1f}%')
                
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, fontsize=9, rotation=20, ha='right')
                ax.set_xlabel('IC Perturbation (Rel L2 %)', fontsize=11)
                ax.set_ylabel('Output Rel L2 Error', fontsize=11)
                ax.set_title(f'{model}: By Input Perturbation', fontsize=13, fontweight='bold')
                
                # Clean up legend
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=9)
                
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
            else:
                ax.text(0.5, 0.5, 'No perturbation data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model}: By Input Perturbation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"2_pgd_{domain.lower()}_analysis.png"), dpi=150)
        plt.close()
        print(f"   [SAVED] 2_pgd_{domain.lower()}_analysis.png")


def plot_noise_baseline(df, plot_dir):
    """Plot 3: Noise Baseline"""
    noise_df = df[(df['Category'] == 'Noise') & (df['Mode'] == 'standard')].copy()
    clean_df = df[(df['Category'] == 'Clean') & (df['Mode'] == 'standard')]
    
    if noise_df.empty:
        print("   [SKIP] No noise data.")
        return
    
    clean_baseline = clean_df.groupby('Model')['Clean_L2'].mean().to_dict()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Noise Baseline: Output Error by Domain', fontsize=14, fontweight='bold')
    
    for idx, domain in enumerate(['Spatial', 'Spectral']):
        ax = axes[idx]
        domain_df = noise_df[noise_df['Domain'] == domain]
        
        if domain_df.empty:
            ax.text(0.5, 0.5, f'No {domain} noise data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{domain} Noise')
            continue
        
        # Group by model and target_l2
        if domain_df['Target_L2'].notna().any():
            pivot = domain_df.pivot_table(index='Target_L2', columns='Model', values='Adv_L2')
        else:
            pivot = domain_df.groupby('Model')['Adv_L2'].mean().to_frame().T
        
        if not pivot.empty:
            pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_xlabel('Target L2 / Noise Std')
            ax.set_ylabel('Output Rel L2 Error')
            ax.set_title(f'{domain} Noise')
            ax.legend(title='Model')
            
            # Add clean baseline lines
            colors = plt.cm.tab10.colors
            for i, model in enumerate(pivot.columns):
                if model in clean_baseline:
                    ax.axhline(y=clean_baseline[model], color=colors[i % len(colors)], 
                              linestyle='--', linewidth=1.5, alpha=0.6)
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "3_noise_baseline.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 3_noise_baseline.png")


def plot_temporal_drift(df_time, plot_dir):
    """Plot 4: Per-Step L2 Error vs Ground Truth Over Time"""
    if df_time.empty:
        print("   [SKIP] No temporal data.")
        return
    
    # Debug output
    print(f"   DEBUG: Temporal data shape: {df_time.shape}")
    print(f"   DEBUG: Categories: {df_time['Category'].unique().tolist()}")
    print(f"   DEBUG: Domains: {df_time['Domain'].unique().tolist()}")
    
    # Check for the new error columns
    has_error_cols = 'Clean_Error' in df_time.columns and 'Adv_Error' in df_time.columns
    
    if not has_error_cols:
        print("   [SKIP] No per-step error data.")
        return
    
    # Get all attack categories (exclude Clean baseline)
    attack_df = df_time[df_time['Category'] != 'Clean'].copy()
    
    if attack_df.empty:
        print("   [SKIP] No attack temporal data.")
        return
    
    # Plot per category
    categories = sorted(attack_df['Category'].unique())
    print(f"   DEBUG: Attack categories found: {categories}")
    
    for category in categories:
        cat_df = attack_df[attack_df['Category'] == category]
        
        if cat_df.empty:
            continue
        
        # Get unique models
        models = sorted(cat_df['Model'].unique())
        
        # For PGD attacks, also split by domain
        if 'PGD' in category:
            domains = sorted(cat_df['Domain'].dropna().unique())
            if len(domains) == 0:
                domains = ['All']
        else:
            domains = ['All']
        
        for domain in domains:
            if domain == 'All':
                domain_df = cat_df
                domain_label = ""
            else:
                domain_df = cat_df[cat_df['Domain'] == domain]
                domain_label = f" ({domain})"
            
            if domain_df.empty:
                continue
            
            # Get max epsilon for title
            max_eps = domain_df['Epsilon'].max() if domain_df['Epsilon'].notna().any() else None
            eps_label = f" ε={max_eps}" if max_eps is not None else ""
            
            fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
            if len(models) == 1:
                axes = [axes]
            
            fig.suptitle(f'Per-Step L2 Error: {category}{domain_label}{eps_label}', fontsize=14, fontweight='bold')
            
            for idx, model in enumerate(models):
                ax = axes[idx]
                model_df = domain_df[domain_df['Model'] == model]
                
                # If multiple epsilons, use max
                if model_df['Epsilon'].notna().any():
                    max_eps_model = model_df['Epsilon'].max()
                    model_df = model_df[model_df['Epsilon'] == max_eps_model]
                
                model_df = model_df.sort_values('Time_Step')
                
                # Average over samples
                time_avg = model_df.groupby('Time_Step').agg({
                    'Clean_Error': 'mean',
                    'Adv_Error': 'mean'
                }).reset_index()
                
                if time_avg.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Debug
                clean_range = (time_avg['Clean_Error'].min(), time_avg['Clean_Error'].max())
                adv_range = (time_avg['Adv_Error'].min(), time_avg['Adv_Error'].max())
                print(f"   DEBUG: {model} {category}{domain_label} - Clean: [{clean_range[0]:.4f}, {clean_range[1]:.4f}], Adv: [{adv_range[0]:.4f}, {adv_range[1]:.4f}]")
                
                # Plot
                ax.plot(time_avg['Time_Step'], time_avg['Clean_Error'] * 100, 
                       'b-', marker='o', label='Clean', linewidth=2, markersize=4)
                ax.plot(time_avg['Time_Step'], time_avg['Adv_Error'] * 100, 
                       'r--', marker='s', label='Adversarial', linewidth=2, markersize=4)
                
                ax.set_xlabel('Time Step', fontsize=11)
                ax.set_ylabel('Rel L2 Error (%)', fontsize=11)
                ax.set_title(f'{model}', fontsize=13, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Fill between if there's a difference
                if (time_avg['Adv_Error'] - time_avg['Clean_Error']).abs().mean() > 1e-6:
                    ax.fill_between(time_avg['Time_Step'], 
                                   time_avg['Clean_Error'] * 100, 
                                   time_avg['Adv_Error'] * 100,
                                   alpha=0.2, color='red')
            
            # Create filename
            safe_category = category.replace('-', '_').lower()
            safe_domain = domain.lower() if domain != 'All' else ''
            filename = f"4_temporal_{safe_category}"
            if safe_domain:
                filename += f"_{safe_domain}"
            filename += ".png"
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, filename), dpi=150)
            plt.close()
            print(f"   [SAVED] {filename}")
    
    # Also create epsilon comparison plot
    _plot_temporal_error_by_epsilon(attack_df, plot_dir)


def _plot_temporal_error_by_epsilon(df_time, plot_dir):
    """Plot per-step error curves for different epsilons and categories."""
    if df_time.empty or 'Adv_Error' not in df_time.columns:
        return
    
    models = sorted(df_time['Model'].unique())
    categories = sorted(df_time['Category'].unique())
    
    for model in models:
        model_df = df_time[df_time['Model'] == model]
        
        if model_df.empty:
            continue
        
        # Create one plot per category
        for category in categories:
            cat_df = model_df[model_df['Category'] == category]
            
            if cat_df.empty:
                continue
            
            epsilons = sorted(cat_df['Epsilon'].dropna().unique())
            if len(epsilons) == 0:
                # No epsilon data, just plot the category
                epsilons = [None]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = plt.cm.plasma(np.linspace(0.2, 0.9, max(len(epsilons), 1)))
            
            # Plot clean baseline
            clean_avg = cat_df.groupby('Time_Step')['Clean_Error'].mean().reset_index()
            if not clean_avg.empty:
                ax.plot(clean_avg['Time_Step'], clean_avg['Clean_Error'] * 100, 
                       'k--', linewidth=2, label='Clean', alpha=0.7)
            
            for idx, eps in enumerate(epsilons):
                if eps is not None:
                    eps_df = cat_df[cat_df['Epsilon'] == eps]
                    label = f'ε={eps:.3f}'
                else:
                    eps_df = cat_df
                    label = 'Adversarial'
                
                time_avg = eps_df.groupby('Time_Step')['Adv_Error'].mean().reset_index()
                if not time_avg.empty:
                    ax.plot(time_avg['Time_Step'], time_avg['Adv_Error'] * 100, 
                           marker='o', color=colors[idx], label=label, linewidth=2, markersize=3)
            
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel('Rel L2 Error (%)', fontsize=11)
            ax.set_title(f'{model}: {category} Per-Step Error', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            safe_category = category.replace('-', '_').lower()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"4b_temporal_eps_{model.lower()}_{safe_category}.png"), dpi=150)
            plt.close()
            print(f"   [SAVED] 4b_temporal_eps_{model.lower()}_{safe_category}.png")


def plot_spectral_cascade(df, plot_dir):
    """Plot 5: Spectral Cascade (High Freq Input -> Low Freq Output)"""
    stealth_df = df[(df['Category'] == 'PGD-Stealth') & (df['Mode'] == 'standard')].copy()
    
    if stealth_df.empty:
        print("   [SKIP] No stealth data for spectral cascade.")
        return
    
    # Get max pde_weight and epsilon
    max_pde = stealth_df['Lambda_PDE'].max()
    max_eps = stealth_df['Epsilon'].max()
    
    subset = stealth_df[(stealth_df['Lambda_PDE'] == max_pde) & (stealth_df['Epsilon'] == max_eps)]
    
    if subset.empty:
        print("   [SKIP] No matching stealth data.")
        return
    
    models = sorted(subset['Model'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    ic_high = [subset[subset['Model'] == m]['IC_High_Freq'].mean() for m in models]
    out_low = [subset[subset['Model'] == m]['Adv_Low_Freq'].mean() for m in models]
    
    bars1 = ax.bar(x - width/2, ic_high, width, label='Input High Freq', color='#E07A5F', alpha=0.8)
    bars2 = ax.bar(x + width/2, out_low, width, label='Output Low Freq Error', color='#3D7C98', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('L2 Norm (Frequency Domain)', fontsize=11)
    ax.set_title(f'Spectral Error Cascade (Stealth Attack)\nε={max_eps}, λ_PDE={max_pde}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "5_spectral_cascade.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 5_spectral_cascade.png")


def plot_cross_resolution(df, plot_dir):
    """Plot 6: Cross-Resolution Analysis"""
    cross_res_df = df[df['Mode'].str.contains('cross_res', na=False)].copy()
    
    if cross_res_df.empty:
        print("   [SKIP] No cross-resolution data.")
        return
    
    models = sorted(cross_res_df['Model'].unique())
    
    plot_data = []
    for model in models:
        model_df = df[df['Model'] == model]
        model_cross = cross_res_df[cross_res_df['Model'] == model]
        
        for attack in model_cross['Attack_Name'].unique():
            attack_df = model_df[model_df['Attack_Name'] == attack]
            
            std_err = attack_df[attack_df['Mode'] == 'standard']['Adv_L2'].mean()
            low_err = attack_df[attack_df['Mode'] == 'cross_res_low']['Adv_L2'].mean()
            high_err = attack_df[attack_df['Mode'] == 'cross_res_high']['Adv_L2'].mean()
            
            epsilon = attack_df['Epsilon'].iloc[0] if not attack_df.empty else np.nan
            domain = 'Spectral' if 'spectral' in attack.lower() else 'Spatial'
            
            plot_data.append({
                'Model': model,
                'Attack': attack,
                'Domain': domain,
                'Epsilon': epsilon,
                'Err_Standard': std_err,
                'Err_Low_Res': low_err,
                'Err_High_Res': high_err,
            })
    
    cross_df = pd.DataFrame(plot_data)
    
    if cross_df.empty:
        print("   [SKIP] No cross-resolution comparison data.")
        return
    
    # Save CSV
    cross_df.to_csv(os.path.join(plot_dir, "6_cross_resolution.csv"), index=False)
    print(f"   [SAVED] 6_cross_resolution.csv")
    
    # Plot
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 8))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Cross-Resolution Analysis: PGD Attack Transferability', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = cross_df[cross_df['Model'] == model].sort_values(['Domain', 'Epsilon'])
        
        if model_data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model}')
            continue
        
        n_attacks = len(model_data)
        x = np.arange(n_attacks)
        width = 0.25
        
        low_vals = model_data['Err_Low_Res'].fillna(0)
        std_vals = model_data['Err_Standard'].fillna(0)
        high_vals = model_data['Err_High_Res'].fillna(0)
        
        ax.bar(x - width, low_vals, width, label='Low Res', color='#90BE6D', alpha=0.8)
        ax.bar(x, std_vals, width, label='Standard', color='#577590', alpha=0.8)
        ax.bar(x + width, high_vals, width, label='High Res', color='#F94144', alpha=0.8)
        
        x_labels = [f"{row['Domain']}\nε={row['Epsilon']:.3f}" if pd.notna(row['Epsilon']) 
                   else row['Domain'] for _, row in model_data.iterrows()]
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel('Output Rel L2 Error', fontsize=11)
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "6_cross_resolution.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 6_cross_resolution.png")


def plot_amplification_ranking(df, plot_dir):
    """Plot 7: Amplification Ranking (per model)"""
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty or not std_df['Amp_Ratio'].notna().any():
        print("   [SKIP] No amplification data.")
        return
    
    plot_df = std_df[std_df['Amp_Ratio'].notna() & std_df['IC_Pert_Rel'].notna()].copy()
    
    if plot_df.empty:
        print("   [SKIP] No complete amplification data.")
        return
    
    category_colors = {
        'Clean': '#2E7D32',
        'Noise': '#808080',
        'PGD-Pure': '#E07A5F',
        'PGD-Stealth': '#3D7C98',
        'Sequential': '#81B29A',
        'Other': '#F4A261'
    }
    
    models = sorted(plot_df['Model'].unique())
    
    # Text report
    report_lines = ["=" * 100, "AMPLIFICATION RANKING REPORT", "=" * 100, ""]
    
    for model in models:
        model_df = plot_df[plot_df['Model'] == model].copy()
        
        if model_df.empty:
            continue
        
        # Create label with domain info
        def make_label(row):
            parts = [row['Category']]
            # Add domain (Spatial/Spectral)
            if pd.notna(row.get('Domain')) and row.get('Domain') != 'Unknown':
                parts.append(row['Domain'][:4])  # "Spat" or "Spec"
            if pd.notna(row.get('Epsilon')):
                parts.append(f"ε={row['Epsilon']:.3f}")
            if pd.notna(row.get('Lambda_PDE')) and row.get('Lambda_PDE', 0) > 0:
                parts.append(f"λ={row['Lambda_PDE']:.0f}")
            if pd.notna(row.get('Target_L2')) and row['Category'] == 'Noise':
                parts.append(f"L2={row['Target_L2']:.1%}")
            return ' | '.join(parts)
        
        model_df['Label'] = model_df.apply(make_label, axis=1)
        model_df = model_df.sort_values('Amp_Ratio', ascending=True)
        
        # Report
        report_lines.extend([f"MODEL: {model}", "=" * 100, "",
            f"{'Rank':<6} {'Experiment':<50} {'IC Pert':<12} {'Output Err':<12} {'Amplification':<12}",
            f"{'-'*6} {'-'*50} {'-'*12} {'-'*12} {'-'*12}"])
        
        for rank, (_, row) in enumerate(model_df.sort_values('Amp_Ratio', ascending=False).iterrows(), 1):
            report_lines.append(f"{rank:<6} {row['Label'][:48]:<50} {row['IC_Pert_Rel']:.2%}{' '*5} "
                              f"{row['Adv_L2']:.2%}{' '*5} {row['Amp_Ratio']:.2f}x")
        
        report_lines.extend(["", f"Avg Amplification: {model_df['Amp_Ratio'].mean():.2f}x", ""])
        
        # Plot
        n_experiments = len(model_df)
        fig_height = max(6, n_experiments * 0.5)
        
        fig, axes = plt.subplots(1, 3, figsize=(14, fig_height), gridspec_kw={'width_ratios': [2, 2, 1.5]})
        fig.suptitle(f'{model}: Experiments Ranked by Amplification Ratio', fontsize=14, fontweight='bold')
        
        y_pos = np.arange(n_experiments)
        bar_height = 0.7
        colors = [category_colors.get(cat, '#808080') for cat in model_df['Category']]
        
        # Panel 1: IC Perturbation
        ax1 = axes[0]
        input_vals = model_df['IC_Pert_Rel'].values * 100
        ax1.barh(y_pos, input_vals, height=bar_height, color=colors, alpha=0.8, edgecolor='white')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(model_df['Label'], fontsize=10)
        ax1.set_xlabel('IC Perturbation (Rel L2 %)', fontsize=11)
        ax1.set_title('Input Perturbation', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Panel 2: Output Error
        ax2 = axes[1]
        output_vals = model_df['Adv_L2'].values * 100
        ax2.barh(y_pos, output_vals, height=bar_height, color=colors, alpha=0.8, edgecolor='white')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])
        ax2.set_xlabel('Output Error (Rel L2 %)', fontsize=11)
        ax2.set_title('Output Error', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Panel 3: Amplification
        ax3 = axes[2]
        amp_vals = model_df['Amp_Ratio'].values
        amp_colors = ['#2E7D32' if v < 1 else '#C62828' for v in amp_vals]
        ax3.barh(y_pos, amp_vals, height=bar_height, color=amp_colors, alpha=0.8, edgecolor='white')
        ax3.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([])
        ax3.set_xlabel('Amplification Ratio', fontsize=11)
        ax3.set_title('Amplification', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(os.path.join(plot_dir, f"7_amplification_{model.lower()}.png"), dpi=150)
        plt.close()
        print(f"   [SAVED] 7_amplification_{model.lower()}.png")
        
        # Save CSV
        model_df[['Label', 'Category', 'Domain', 'Epsilon', 'Lambda_PDE', 'IC_Pert_Rel', 'Adv_L2', 'Amp_Ratio']].to_csv(
            os.path.join(plot_dir, f"7_amplification_{model.lower()}.csv"), index=False)
        print(f"   [SAVED] 7_amplification_{model.lower()}.csv")
    
    # Save report
    with open(os.path.join(plot_dir, "7_amplification_report.txt"), 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"   [SAVED] 7_amplification_report.txt")


def plot_summary_heatmap(df, plot_dir):
    """Plot 8: Summary Heatmap"""
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty:
        print("   [SKIP] No data for heatmap.")
        return
    
    # Pivot: Models vs Categories
    pivot = std_df.pivot_table(index='Category', columns='Model', values='Adv_L2', aggfunc='mean')
    
    if pivot.empty:
        print("   [SKIP] Cannot create pivot table.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
                linewidths=0.5, cbar_kws={'label': 'Output Rel L2 Error'})
    
    ax.set_title('Adversarial Robustness Summary', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Attack Category', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "8_summary_heatmap.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 8_summary_heatmap.png")
    
    # Save CSV
    pivot.to_csv(os.path.join(plot_dir, "8_summary_heatmap.csv"))
    print(f"   [SAVED] 8_summary_heatmap.csv")


def plot_physics_residual(df_time, plot_dir):
    """Plot 9: Physics Residual Comparison - PGD-Pure vs PGD-Stealth
    
    This shows how the physics constraint (lambda_pde) in stealth attacks
    keeps the physics residual lower while still causing prediction errors.
    """
    if df_time.empty:
        print("   [SKIP] No temporal data for physics residual.")
        return
    
    # Check if we have physics data
    if 'Adv_Physics' not in df_time.columns:
        print("   [SKIP] No physics residual data.")
        return
    
    # Get both PGD-Pure and PGD-Stealth data
    pure_df = df_time[df_time['Category'] == 'PGD-Pure'].copy()
    stealth_df = df_time[df_time['Category'] == 'PGD-Stealth'].copy()
    
    if pure_df.empty and stealth_df.empty:
        print("   [SKIP] No PGD attack data for physics comparison.")
        return
    
    models = sorted(set(pure_df['Model'].unique()) | set(stealth_df['Model'].unique()))
    
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Physics Residual: PGD-Pure vs PGD-Stealth', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        model_pure = pure_df[pure_df['Model'] == model]
        model_stealth = stealth_df[stealth_df['Model'] == model]
        
        has_data = False
        
        # Plot Clean baseline (from either)
        if not model_pure.empty and 'Clean_Physics' in model_pure.columns:
            clean_avg = model_pure.groupby('Time_Step')['Clean_Physics'].mean().reset_index()
            if not clean_avg.empty:
                ax.semilogy(clean_avg['Time_Step'], clean_avg['Clean_Physics'], 
                           'k--', marker='.', label='Clean', linewidth=2, markersize=3, alpha=0.7)
                has_data = True
        
        # Plot PGD-Pure (at max epsilon)
        if not model_pure.empty:
            if model_pure['Epsilon'].notna().any():
                max_eps = model_pure['Epsilon'].max()
                pure_subset = model_pure[model_pure['Epsilon'] == max_eps]
            else:
                pure_subset = model_pure
            
            if not pure_subset.empty:
                pure_avg = pure_subset.groupby('Time_Step')['Adv_Physics'].mean().reset_index()
                if not pure_avg.empty:
                    ax.semilogy(pure_avg['Time_Step'], pure_avg['Adv_Physics'], 
                               'r-', marker='o', label=f'PGD-Pure (ε={max_eps:.3f})', 
                               linewidth=2, markersize=4)
                    has_data = True
        
        # Plot PGD-Stealth (at max epsilon, max lambda_pde)
        if not model_stealth.empty:
            if model_stealth['Epsilon'].notna().any():
                max_eps_s = model_stealth['Epsilon'].max()
                stealth_subset = model_stealth[model_stealth['Epsilon'] == max_eps_s]
            else:
                stealth_subset = model_stealth
            
            if not stealth_subset.empty:
                stealth_avg = stealth_subset.groupby('Time_Step')['Adv_Physics'].mean().reset_index()
                if not stealth_avg.empty:
                    ax.semilogy(stealth_avg['Time_Step'], stealth_avg['Adv_Physics'], 
                               'b-', marker='s', label=f'PGD-Stealth (ε={max_eps_s:.3f})', 
                               linewidth=2, markersize=4)
                    has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Physics Residual (log)', fontsize=11)
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "9_physics_residual.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 9_physics_residual.png")
    
    # Also create a bar chart comparing average physics residual across attack types
    _plot_physics_residual_bar(df_time, plot_dir)


def _plot_physics_residual_bar(df_time, plot_dir):
    """Bar chart comparing average physics residual across attack types."""
    if df_time.empty or 'Adv_Physics' not in df_time.columns:
        return
    
    # Average over all time steps
    summary = df_time.groupby(['Model', 'Category']).agg({
        'Adv_Physics': 'mean',
        'Clean_Physics': 'mean'
    }).reset_index()
    
    if summary.empty:
        return
    
    models = sorted(summary['Model'].unique())
    categories = ['PGD-Pure', 'PGD-Stealth', 'Noise', 'Sequential']
    categories = [c for c in categories if c in summary['Category'].values]
    
    if not categories:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.8 / (len(categories) + 1)  # +1 for Clean
    
    colors = {
        'Clean': '#2E7D32',
        'PGD-Pure': '#E07A5F', 
        'PGD-Stealth': '#3D7C98',
        'Noise': '#808080',
        'Sequential': '#81B29A'
    }
    
    # Plot Clean baseline
    clean_vals = [summary[(summary['Model'] == m)]['Clean_Physics'].mean() for m in models]
    ax.bar(x - width * len(categories) / 2, clean_vals, width, label='Clean', color=colors['Clean'], alpha=0.8)
    
    # Plot each category
    for i, cat in enumerate(categories):
        cat_vals = []
        for m in models:
            cat_data = summary[(summary['Model'] == m) & (summary['Category'] == cat)]
            cat_vals.append(cat_data['Adv_Physics'].mean() if not cat_data.empty else 0)
        
        offset = (i + 1 - len(categories) / 2) * width
        ax.bar(x + offset, cat_vals, width, label=cat, color=colors.get(cat, '#808080'), alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('Average Physics Residual', fontsize=11)
    ax.set_title('Physics Residual by Attack Type', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "9b_physics_residual_bar.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 9b_physics_residual_bar.png")


def plot_injection_vs_drift(df_time, plot_dir):
    """Plot 10: Injection vs Drift Analysis - Shows how initial perturbation evolves"""
    if df_time.empty:
        print("   [SKIP] No temporal data for injection/drift.")
        return
    
    if 'Injection' not in df_time.columns or 'Drift' not in df_time.columns:
        print("   [SKIP] No injection/drift data.")
        return
    
    # Get PGD attacks
    pgd_df = df_time[df_time['Category'].isin(['PGD-Pure', 'PGD-Stealth'])].copy()
    if pgd_df.empty:
        print("   [SKIP] No PGD temporal data.")
        return
    
    models = sorted(pgd_df['Model'].unique())
    
    # Create one plot per model showing injection vs drift for different epsilons
    for model in models:
        model_df = pgd_df[pgd_df['Model'] == model]
        
        if model_df.empty:
            continue
        
        # Get unique epsilons
        epsilons = sorted(model_df['Epsilon'].dropna().unique())
        if len(epsilons) == 0:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{model}: Injection vs Drift Over Time', fontsize=14, fontweight='bold')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons)))
        
        # Left: Injection
        ax1 = axes[0]
        for eps, color in zip(epsilons, colors):
            eps_df = model_df[model_df['Epsilon'] == eps]
            time_avg = eps_df.groupby('Time_Step')['Injection'].mean().reset_index()
            ax1.plot(time_avg['Time_Step'], time_avg['Injection'], 
                    marker='o', color=color, label=f'ε={eps:.3f}', linewidth=2, markersize=3)
        
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Injection L2', fontsize=11)
        ax1.set_title('Initial Perturbation Injection', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Right: Drift
        ax2 = axes[1]
        for eps, color in zip(epsilons, colors):
            eps_df = model_df[model_df['Epsilon'] == eps]
            time_avg = eps_df.groupby('Time_Step')['Drift'].mean().reset_index()
            ax2.plot(time_avg['Time_Step'], time_avg['Drift'], 
                    marker='s', color=color, label=f'ε={eps:.3f}', linewidth=2, markersize=3)
        
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Drift (Rel L2)', fontsize=11)
        ax2.set_title('Error Drift Over Time', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"10_injection_drift_{model.lower()}.png"), dpi=150)
        plt.close()
        print(f"   [SAVED] 10_injection_drift_{model.lower()}.png")


def plot_sequential_attacks(df, df_time, plot_dir):
    """Plot 11: Sequential Attack Analysis"""
    seq_df = df[(df['Category'] == 'Sequential') & (df['Mode'] == 'standard')].copy()
    
    if seq_df.empty:
        print("   [SKIP] No sequential attack data.")
        return
    
    models = sorted(seq_df['Model'].unique())
    
    # Compare sequential vs single-step PGD
    pgd_df = df[(df['Category'] == 'PGD-Pure') & (df['Mode'] == 'standard')].copy()
    
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Sequential vs Single-Step Attack Comparison', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        model_seq = seq_df[seq_df['Model'] == model]
        model_pgd = pgd_df[pgd_df['Model'] == model]
        
        x_labels = []
        seq_vals = []
        pgd_vals = []
        
        # Group by epsilon
        for eps in sorted(model_seq['Epsilon'].dropna().unique()):
            eps_seq = model_seq[model_seq['Epsilon'] == eps]['Adv_L2'].mean()
            eps_pgd = model_pgd[model_pgd['Epsilon'] == eps]['Adv_L2'].mean() if not model_pgd.empty else np.nan
            
            x_labels.append(f'ε={eps:.3f}')
            seq_vals.append(eps_seq)
            pgd_vals.append(eps_pgd if pd.notna(eps_pgd) else 0)
        
        if not x_labels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        x = np.arange(len(x_labels))
        width = 0.35
        
        ax.bar(x - width/2, seq_vals, width, label='Sequential', color='#81B29A', alpha=0.8)
        ax.bar(x + width/2, pgd_vals, width, label='PGD Single-Step', color='#E07A5F', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel('Output Rel L2 Error', fontsize=11)
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "11_sequential_attacks.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 11_sequential_attacks.png")


def plot_stealth_analysis(df, plot_dir):
    """Plot 12: Stealth Attack Analysis - Lambda_PDE vs Output Error"""
    stealth_df = df[(df['Category'] == 'PGD-Stealth') & (df['Mode'] == 'standard')].copy()
    
    if stealth_df.empty:
        print("   [SKIP] No stealth attack data.")
        return
    
    if not stealth_df['Lambda_PDE'].notna().any():
        print("   [SKIP] No Lambda_PDE data for stealth attacks.")
        return
    
    models = sorted(stealth_df['Model'].unique())
    
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Stealth Attack: Physics Constraint vs Output Error', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = stealth_df[stealth_df['Model'] == model]
        
        # Group by Lambda_PDE and Epsilon
        for eps in sorted(model_df['Epsilon'].dropna().unique()):
            eps_df = model_df[model_df['Epsilon'] == eps]
            grouped = eps_df.groupby('Lambda_PDE')['Adv_L2'].mean().reset_index()
            
            if not grouped.empty:
                ax.plot(grouped['Lambda_PDE'], grouped['Adv_L2'], 
                       marker='o', label=f'ε={eps:.3f}', linewidth=2)
        
        ax.set_xlabel('Lambda PDE (Physics Weight)', fontsize=11)
        ax.set_ylabel('Output Rel L2 Error', fontsize=11)
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Log scale if values span orders of magnitude
        if ax.get_xlim()[1] / max(ax.get_xlim()[0], 1e-10) > 100:
            ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "12_stealth_analysis.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 12_stealth_analysis.png")


def plot_frequency_analysis(df, plot_dir):
    """Plot 13: Frequency Analysis - Input vs Output frequency distribution"""
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty:
        print("   [SKIP] No data for frequency analysis.")
        return
    
    # Check for frequency columns
    freq_cols = ['IC_Low_Freq', 'IC_High_Freq', 'Adv_Low_Freq', 'Adv_High_Freq']
    if not all(col in std_df.columns for col in freq_cols):
        print("   [SKIP] Missing frequency columns.")
        return
    
    models = sorted(std_df['Model'].unique())
    categories = ['PGD-Pure', 'PGD-Stealth', 'Noise']
    
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models)))
    if len(models) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Frequency Analysis: Input Perturbation vs Output Error', fontsize=14, fontweight='bold')
    
    category_colors = {'PGD-Pure': '#E07A5F', 'PGD-Stealth': '#3D7C98', 'Noise': '#808080', 'Sequential': '#81B29A'}
    
    for idx, model in enumerate(models):
        model_df = std_df[std_df['Model'] == model]
        
        # Left: Input Frequency Distribution (stacked bar)
        ax1 = axes[idx, 0]
        
        valid_cats = []
        low_fracs = []
        high_fracs = []
        colors = []
        
        for cat in categories:
            cat_df = model_df[model_df['Category'] == cat]
            if not cat_df.empty:
                low = cat_df['IC_Low_Freq'].mean()
                high = cat_df['IC_High_Freq'].mean()
                total = low + high
                if total > 1e-9:
                    valid_cats.append(cat)
                    low_fracs.append(low / total)
                    high_fracs.append(high / total)
                    colors.append(category_colors.get(cat, '#808080'))
        
        if valid_cats:
            x = np.arange(len(valid_cats))
            ax1.bar(x, low_fracs, label='Low Freq (k≤6)', color=colors, alpha=0.5)
            ax1.bar(x, high_fracs, bottom=low_fracs, label='High Freq (k>6)', color=colors, alpha=1.0)
            ax1.set_xticks(x)
            ax1.set_xticklabels(valid_cats, fontsize=10)
        
        ax1.set_ylabel('Frequency Distribution', fontsize=11)
        ax1.set_title(f'{model}: Input Perturbation Spectrum', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.legend(fontsize=8)
        
        # Right: Output Frequency Error
        ax2 = axes[idx, 1]
        
        valid_cats = []
        low_vals = []
        high_vals = []
        
        for cat in categories:
            cat_df = model_df[model_df['Category'] == cat]
            if not cat_df.empty:
                valid_cats.append(cat)
                low_vals.append(cat_df['Adv_Low_Freq'].mean())
                high_vals.append(cat_df['Adv_High_Freq'].mean())
        
        if valid_cats:
            x = np.arange(len(valid_cats))
            width = 0.35
            
            ax2.bar(x - width/2, low_vals, width, label='Low Freq Error', color='#577590', alpha=0.8)
            ax2.bar(x + width/2, high_vals, width, label='High Freq Error', color='#F94144', alpha=0.8)
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(valid_cats, fontsize=10)
        
        ax2.set_ylabel('Frequency Error (L2)', fontsize=11)
        ax2.set_title(f'{model}: Output Error Spectrum', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "13_frequency_analysis.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 13_frequency_analysis.png")


def plot_absolute_perturbation_comparison(df, plot_dir):
    """
    Plot 14: Absolute Perturbation Comparison
    Compare all attack types on the same absolute input perturbation scale.
    Creates scatter plots and efficiency curves showing output error vs absolute input L2.
    """
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty:
        print("   [SKIP] No data for absolute comparison.")
        return
    
    # Filter to rows with valid perturbation data
    plot_df = std_df[std_df['IC_Pert_Abs'].notna() & 
                     std_df['Adv_L2'].notna()].copy()
    
    if plot_df.empty:
        print("   [SKIP] No perturbation data for absolute comparison.")
        return
    
    # Color scheme by category
    category_colors = {
        'Noise': '#808080',
        'PGD-Pure': '#E07A5F',
        'PGD-Stealth': '#3D7C98',
        'Sequential': '#81B29A',
        'Clean': '#2E7D32',
        'Other': '#F4A261'
    }
    
    # Marker scheme by domain
    domain_markers = {
        'Spatial': 'o',
        'Spectral': 's',
        'Unknown': '^'
    }
    
    models = sorted(plot_df['Model'].unique())
    
    # === Plot 14a: Scatter plot - Output Error vs Absolute Input Perturbation (per model) ===
    for model in models:
        model_df = plot_df[plot_df['Model'] == model]
        
        if model_df.empty:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{model}: Attack Comparison by Absolute Perturbation', 
                    fontsize=14, fontweight='bold')
        
        # --- Left: Scatter plot ---
        ax1 = axes[0]
        
        for category in model_df['Category'].unique():
            cat_df = model_df[model_df['Category'] == category]
            color = category_colors.get(category, '#808080')
            
            for domain in cat_df['Domain'].unique():
                dom_df = cat_df[cat_df['Domain'] == domain]
                marker = domain_markers.get(domain, 'o')
                
                ax1.scatter(dom_df['IC_Pert_Abs'], 
                           dom_df['Adv_L2'] * 100,
                           c=color, marker=marker, s=80, alpha=0.7,
                           label=f'{category} ({domain})')
        
        ax1.set_xlabel('Absolute Input Perturbation (L2 Norm)', fontsize=11)
        ax1.set_ylabel('Output Relative L2 Error (%)', fontsize=11)
        ax1.set_title('Output Error vs Input Perturbation', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=8, loc='best')
        
        # --- Right: Efficiency (Output Error / Input Perturbation) ---
        ax2 = axes[1]
        
        # Group by category and compute mean efficiency
        efficiency_data = []
        for category in model_df['Category'].unique():
            cat_df = model_df[model_df['Category'] == category]
            for domain in cat_df['Domain'].unique():
                dom_df = cat_df[cat_df['Domain'] == domain]
                if len(dom_df) > 0:
                    # Efficiency = output error / input perturbation
                    efficiency = (dom_df['Adv_L2'] * 100) / (dom_df['IC_Pert_Abs'] + 1e-8)
                    efficiency_data.append({
                        'Category': category,
                        'Domain': domain,
                        'Mean_Efficiency': efficiency.mean(),
                        'Std_Efficiency': efficiency.std(),
                        'Color': category_colors.get(category, '#808080')
                    })
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            eff_df['Label'] = eff_df['Category'] + ' (' + eff_df['Domain'] + ')'
            eff_df = eff_df.sort_values('Mean_Efficiency', ascending=True)
            
            y_pos = np.arange(len(eff_df))
            bars = ax2.barh(y_pos, eff_df['Mean_Efficiency'], 
                           color=eff_df['Color'], alpha=0.8,
                           xerr=eff_df['Std_Efficiency'], capsize=3)
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(eff_df['Label'], fontsize=9)
            ax2.set_xlabel('Attack Efficiency\n(Output Error % / Input L2)', fontsize=11)
            ax2.set_title('Attack Efficiency Ranking\n(Higher = More Effective)', fontsize=12)
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"14a_abs_comparison_{model.lower()}.png"), dpi=150)
        plt.close()
        print(f"   [SAVED] 14a_abs_comparison_{model.lower()}.png")
    
    # === Plot 14b: Combined view - All models, binned by perturbation magnitude ===
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Output Error vs Absolute Input Perturbation (All Attack Types)', 
                fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = plot_df[plot_df['Model'] == model].copy()
        
        if model_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model)
            continue
        
        # Create bins based on absolute perturbation
        try:
            model_df['Pert_Bin'], bins = pd.qcut(model_df['IC_Pert_Abs'], q=5, 
                                                  labels=False, retbins=True, duplicates='drop')
        except ValueError:
            model_df['Pert_Bin'], bins = pd.cut(model_df['IC_Pert_Abs'], bins=5, 
                                                 labels=False, retbins=True)
        
        n_bins = len(bins) - 1
        x = np.arange(n_bins)
        
        # Get unique categories
        categories = sorted(model_df['Category'].unique())
        width = 0.8 / max(len(categories), 1)
        
        for i, category in enumerate(categories):
            cat_df = model_df[model_df['Category'] == category]
            if cat_df.empty:
                continue
            
            # Compute mean output error per bin
            bin_means = []
            bin_stds = []
            for b in range(n_bins):
                bin_df = cat_df[cat_df['Pert_Bin'] == b]
                if len(bin_df) > 0:
                    bin_means.append(bin_df['Adv_L2'].mean() * 100)
                    bin_stds.append(bin_df['Adv_L2'].std() * 100)
                else:
                    bin_means.append(0)
                    bin_stds.append(0)
            
            offset = (i - len(categories)/2 + 0.5) * width
            color = category_colors.get(category, '#808080')
            ax.bar(x + offset, bin_means, width, label=category, color=color, alpha=0.8,
                  yerr=bin_stds, capsize=2)
        
        # X-axis labels
        x_labels = [f'{bins[i]:.2e}-{bins[i+1]:.2e}' for i in range(n_bins)]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_xlabel('Absolute Input Perturbation (L2)', fontsize=10)
        ax.set_ylabel('Output Rel L2 Error (%)', fontsize=10)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "14b_abs_comparison_binned.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 14b_abs_comparison_binned.png")
    
    # === Plot 14c: Line plot - Output error vs perturbation for each category ===
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Attack Effectiveness Curves (Grouped by Absolute Perturbation)', 
                fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = plot_df[plot_df['Model'] == model]
        
        if model_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model)
            continue
        
        for category in model_df['Category'].unique():
            cat_df = model_df[model_df['Category'] == category].sort_values('IC_Pert_Abs')
            
            if len(cat_df) < 2:
                continue
            
            color = category_colors.get(category, '#808080')
            
            # Group by perturbation level and compute mean
            cat_df = cat_df.copy()
            cat_df['Pert_Group'] = pd.cut(cat_df['IC_Pert_Abs'], bins=10, labels=False)
            grouped = cat_df.groupby('Pert_Group').agg({
                'IC_Pert_Abs': 'mean',
                'Adv_L2': ['mean', 'std']
            }).dropna()
            
            if len(grouped) > 1:
                x_vals = grouped['IC_Pert_Abs']['mean']
                y_vals = grouped['Adv_L2']['mean'] * 100
                y_err = grouped['Adv_L2']['std'] * 100
                
                ax.plot(x_vals, y_vals, 'o-', color=color, label=category, 
                       linewidth=2, markersize=6)
                ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, 
                               color=color, alpha=0.2)
        
        ax.set_xlabel('Absolute Input Perturbation (L2)', fontsize=10)
        ax.set_ylabel('Output Rel L2 Error (%)', fontsize=10)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "14c_abs_comparison_curves.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 14c_abs_comparison_curves.png")


def print_summary_table(df):
    """Print summary statistics to console."""
    if df.empty:
        return
    
    std_df = df[df['Mode'] == 'standard']
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    summary = std_df.groupby(['Model', 'Category']).agg({
        'Adv_L2': ['mean', 'std', 'min', 'max'],
        'Amp_Ratio': 'mean'
    }).round(4)
    
    print(summary.to_string())
    print("=" * 80)


# === 6. MAIN ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather_only", action="store_true", help="Skip execution, just plot existing results")
    parser.add_argument("--output_dir", default="temporal/ns_2d/benchmark_results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.gather_only:
        for model_name, config_path in MODELS.items():
            print(f"\n=== PROCESSING MODEL: {model_name.upper()} ===")
            try:
                info = parse_configs(config_path)
            except Exception as e:
                print(f"Skipping {model_name}: {e}")
                continue
            
            run_attacks(config_path)
            evaluate_folder(model_name, info)
            evaluate_clean(model_name, info)

    # Aggregate
    df, df_time = aggregate_stats()
    
    if df.empty:
        print("[WARNING] No results to aggregate.")
        return
    
    # Save CSV
    csv_path = os.path.join(args.output_dir, "benchmark_metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    if not df_time.empty:
        df_time.to_csv(os.path.join(args.output_dir, "temporal_metrics.csv"), index=False)
        print(f"Temporal data saved to: {args.output_dir}/temporal_metrics.csv")
    
    print_summary_table(df)
    
    # Plot
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots in: {plot_dir}")
    
    try:
        plot_clean_baseline(df, plot_dir)           # 1
        plot_pgd_robustness(df, plot_dir)           # 2
        plot_noise_baseline(df, plot_dir)           # 3
        plot_temporal_drift(df_time, plot_dir)      # 4
        plot_spectral_cascade(df, plot_dir)         # 5
        plot_cross_resolution(df, plot_dir)         # 6
        plot_amplification_ranking(df, plot_dir)    # 7
        plot_summary_heatmap(df, plot_dir)          # 8
        plot_physics_residual(df_time, plot_dir)    # 9
        plot_injection_vs_drift(df_time, plot_dir)  # 10
        plot_sequential_attacks(df, df_time, plot_dir)  # 11
        plot_stealth_analysis(df, plot_dir)         # 12
        plot_frequency_analysis(df, plot_dir)       # 13
        plot_absolute_perturbation_comparison(df, plot_dir)  # 14
    except Exception as e:
        print(f"[WARNING] Plotting failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nBenchmark complete!")


if __name__ == "__main__":
    main()