import glob
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
import tomllib
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODELS = {
    'FNO':  'simple/darcy_2d/fno_attack_config.toml',
    'FFNO': 'simple/darcy_2d/ffno_attack_config.toml',
    'CNO':  'simple/darcy_2d/cno_attack_config.toml',
}

# ==========================================
# 2. EXECUTION ENGINE
# ==========================================
def parse_configs(attack_config_path):
    if not os.path.exists(attack_config_path):
        raise FileNotFoundError(f"Config not found: {attack_config_path}")

    with open(attack_config_path, "rb") as f:
        atk_cfg = tomllib.load(f)
    
    info = {
        "output_dir": atk_cfg['general']['output_dir'],
        "model_conf": atk_cfg['model']['model_config_path'],
        "model_path": atk_cfg['model']['model_path'],
        "data_conf": atk_cfg['model']['data_config_path'],
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


def run_attacks(attack_config_path):
    print(f"   -> Generating Attacks...")
    subprocess.run([
        "python", "-m", "simple.darcy_2d.attack",
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
        
        # Standard evaluation
        std_out = os.path.join(attack_dir, "eval_results", basename, "standard")
        subprocess.run([
            "python", "-m", "simple.darcy_2d.evaluate",
            "--model_config", info["model_conf"],
            "--model_path", info["model_path"],
            "--data_path", h5,
            "--output_dir", std_out,
            "--plot_samples", "10"
        ], check=True)
        
        # Cross-resolution evaluation for PGD pure attacks only (spatial/spectral, not stealth)
        is_cross_resolution = (("pgd_spatial" in basename.lower() or "pgd_spectral" in basename.lower()) 
                               and "stealth" not in basename.lower())
        
        if is_cross_resolution:
            base_res = info.get("base_res", 64)
            
            # Low resolution evaluation
            if info.get("low_res") and os.path.exists(info["low_res"]):
                low_out = os.path.join(attack_dir, "eval_results", basename, "cross_res_low")
                try:
                    subprocess.run([
                        "python", "-m", "simple.darcy_2d.evaluate",
                        "--model_config", info["model_conf"],
                        "--model_path", info["model_path"],
                        "--data_path", h5,
                        "--target_path", info["low_res"],
                        "--output_dir", low_out,
                        "--plot_samples", "0",
                        "--cross_res_mode", "low",
                        "--base_res", str(base_res)
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"      [WARNING] Low-res evaluation failed: {e}")
            
            # High resolution evaluation
            if info.get("high_res") and os.path.exists(info["high_res"]):
                high_out = os.path.join(attack_dir, "eval_results", basename, "cross_res_high")
                try:
                    subprocess.run([
                        "python", "-m", "simple.darcy_2d.evaluate",
                        "--model_config", info["model_conf"],
                        "--model_path", info["model_path"],
                        "--data_path", h5,
                        "--target_path", info["high_res"],
                        "--output_dir", high_out,
                        "--plot_samples", "0",
                        "--cross_res_mode", "high",
                        "--base_res", str(base_res)
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
            "python", "-m", "simple.darcy_2d.evaluate",
            "--model_config", info["model_conf"],
            "--model_path", info["model_path"],
            "--data_path", clean_data_path,
            "--output_dir", clean_out,
            "--plot_samples", "0"
        ], check=True)
        print(f"      [OK] Clean baseline evaluated")
    except subprocess.CalledProcessError as e:
        print(f"      [WARNING] Clean evaluation failed: {e}")


def classify_attack(attack_name):
    name_lower = attack_name.lower()
    if "clean" in name_lower or "baseline" in name_lower:
        return "Clean"
    elif "noise" in name_lower:
        return "Noise"
    elif "stealth" in name_lower:
        return "PGD-Stealth"
    elif "pgd" in name_lower:
        return "PGD-Pure"
    elif "mvmo" in name_lower:
        return "MVMO"
    elif "boundary" in name_lower:
        return "Boundary"
    else:
        return "Other"


def get_domain(attack_name):
    name_lower = attack_name.lower()
    if "spectral" in name_lower:
        return "Spectral"
    elif "spatial" in name_lower:
        return "Spatial"
    return "Unknown"


def load_attack_params(attack_dir, attack_name):
    json_path = os.path.join(attack_dir, f"{attack_name}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
        # Support both old (eps, target_l2) and new (rel_eps, rel_l2) naming
        return {
            "rel_eps": params.get("rel_eps", params.get("eps")),
            "rel_l2": params.get("rel_l2", params.get("target_l2")),
            "lambda_pde": params.get("lambda_pde", params.get("pde")),
            "lambda_bc": params.get("lambda_bc", params.get("bc"))
        }
    return {"rel_eps": None, "rel_l2": None, "lambda_pde": None, "lambda_bc": None}


def aggregate_stats():
    print("\n[BENCHMARK] Aggregating Metrics...")
    results = []
    
    search_dirs = []
    for m_name, cfg_path in MODELS.items():
        try:
            info = parse_configs(cfg_path)
            search_dirs.append((m_name, info["output_dir"]))
        except Exception as e:
            print(f"Skipping {m_name}: {e}")

    for model_name, base_dir in search_dirs:
        json_files = glob.glob(f"{base_dir}/**/benchmark_metrics.json", recursive=True)
        
        for jf in json_files:
            try:
                parts = jf.split(os.sep)
                attack_name = parts[-3]
                mode = parts[-2]
                
                with open(jf, 'r') as f:
                    m = json.load(f)
                
                attack_params = load_attack_params(base_dir, attack_name)
                metrics = m.get('metrics', {})
                spectral = m.get('spectral_changes', {})
                pert = m.get('perturbation', {})
                cross_res = m.get('cross_resolution', {}) or {}
                
                entry = {
                    'Model': model_name,
                    'Mode': mode,
                    'Attack_Name': attack_name,
                    'Category': classify_attack(attack_name),
                    'Domain': get_domain(attack_name),
                    # Attack parameters (relative)
                    'Rel_Eps': attack_params.get('rel_eps'),
                    'Rel_L2': attack_params.get('rel_l2'),
                    'Lambda_PDE': attack_params.get('lambda_pde'),
                    'Lambda_BC': attack_params.get('lambda_bc'),
                    # Output metrics
                    'Output_Rel_L2': metrics.get('avg_output_rel_l2', np.nan),
                    'Output_Abs_L2': metrics.get('avg_output_abs_l2', np.nan),
                    # Physics loss
                    'PDE_Loss': metrics.get('avg_pde_loss', np.nan),
                    'BC_Loss': metrics.get('avg_bc_loss', np.nan),
                    'Physics_Loss': metrics.get('avg_physics_loss', np.nan),
                    # Input perturbation
                    'Input_Abs_Pert_L2': metrics.get('avg_abs_input_pert_l2', np.nan),
                    'Input_Rel_Pert_L2': metrics.get('avg_rel_input_pert_l2', np.nan),
                    # Amplification
                    'Amp_Abs_Ratio': metrics.get('amplification_abs_ratio', np.nan),
                    'Amp_Rel_Ratio': metrics.get('amplification_rel_ratio', np.nan),
                    # Perturbation characteristics
                    'Pert_Max_Abs': pert.get('max_abs', np.nan),
                    'Pert_Linf': pert.get('avg_linf', np.nan),
                    'Pert_Sparsity': pert.get('sparsity', np.nan),
                    'Pert_Smoothness': pert.get('smoothness', np.nan),
                    'Pert_Low_Freq_Ratio': pert.get('low_freq_ratio', np.nan),
                    'Pert_High_Freq_Ratio': pert.get('high_freq_ratio', np.nan),
                    # Spectral changes
                    'In_Low_Freq_Chg': spectral.get('in_low_chg', np.nan),
                    'In_High_Freq_Chg': spectral.get('in_high_chg', np.nan),
                    'Out_Low_Freq_Diff': spectral.get('out_low_diff', np.nan),
                    'Out_High_Freq_Diff': spectral.get('out_high_diff', np.nan),
                    # Cross-resolution info
                    'Cross_Res_Mode': cross_res.get('mode'),
                    'Base_Res': cross_res.get('base_res'),
                    'Target_Res': cross_res.get('target_res'),
                }
                results.append(entry)
            except Exception as e:
                print(f"   [WARNING] Failed to parse {jf}: {e}")

    return pd.DataFrame(results)


# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_noise_impact(df, plot_dir):
    """
    Plot 1: Noise Impact - Separate charts for Spatial/Spectral
    Includes clean baseline as horizontal reference line.
    """
    noise_df = df[(df['Category'] == 'Noise') & (df['Mode'] == 'standard')].copy()
    clean_df = df[(df['Category'] == 'Clean') & (df['Mode'] == 'standard')].copy()
    
    if noise_df.empty:
        print("   [SKIP] No noise data.")
        return
    
    # Get clean baseline per model
    clean_baseline = {}
    for model in df['Model'].unique():
        model_clean = clean_df[clean_df['Model'] == model]
        if not model_clean.empty:
            clean_baseline[model] = model_clean['Output_Rel_L2'].mean() * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Noise Impact: Output Error by Target L2', fontsize=14, fontweight='bold')
    
    for idx, domain in enumerate(['Spatial', 'Spectral']):
        ax = axes[idx]
        domain_df = noise_df[noise_df['Domain'] == domain]
        
        if domain_df.empty:
            ax.text(0.5, 0.5, f'No {domain} noise data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{domain} Noise')
            continue
        
        pivot = domain_df.pivot_table(index='Rel_L2', columns='Model', values='Output_Rel_L2') * 100
        
        if not pivot.empty:
            pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_xlabel('Target L2')
            ax.set_ylabel('Output Relative L2 Error (%)')
            ax.set_title(f'{domain} Noise')
            ax.legend(title='Model')
            ax.set_xticklabels([f'{x:.3f}' for x in pivot.index], rotation=0)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', fontsize=8)
            
            # Add clean baseline lines for each model
            colors = plt.cm.tab10.colors
            for i, model in enumerate(pivot.columns):
                if model in clean_baseline:
                    ax.axhline(y=clean_baseline[model], color=colors[i % len(colors)], 
                              linestyle='--', linewidth=1.5, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "1_noise_impact.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 1_noise_impact.png")


def plot_pgd_by_model(df, plot_dir):
    """
    Plot 2: PGD Analysis - Separate files for Spatial and Spectral
      Row 1: By Rel_Eps (bar chart with value labels)
      Row 2: By Input Perturbation with noise baseline (annotated)
    Includes clean baseline as horizontal reference line.
    Creates: 2_pgd_spatial_analysis.png, 2_pgd_spectral_analysis.png
    """
    pgd_df = df[(df['Category'].isin(['PGD-Pure', 'PGD-Stealth'])) & (df['Mode'] == 'standard')].copy()
    noise_df = df[(df['Category'] == 'Noise') & (df['Mode'] == 'standard')].copy()
    clean_df = df[(df['Category'] == 'Clean') & (df['Mode'] == 'standard')].copy()
    
    if pgd_df.empty:
        print("   [SKIP] No PGD data.")
        return
    
    # Color scheme
    colors = {'PGD-Pure': '#E07A5F', 'PGD-Stealth': '#3D7C98'}
    
    # Get clean baseline per model
    clean_baseline = {}
    for model in pgd_df['Model'].unique():
        model_clean = clean_df[clean_df['Model'] == model]
        if not model_clean.empty:
            clean_baseline[model] = model_clean['Output_Rel_L2'].mean() * 100
    
    # Create separate plots for each domain
    for domain in ['Spatial', 'Spectral']:
        domain_df = pgd_df[pgd_df['Domain'] == domain]
        domain_noise = noise_df[noise_df['Domain'] == domain]
        
        if domain_df.empty:
            print(f"   [SKIP] No {domain} PGD data.")
            continue
        
        models = sorted(domain_df['Model'].unique())
        fig, axes = plt.subplots(2, len(models), figsize=(7 * len(models), 12))
        if len(models) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'PGD {domain} Attack Analysis', fontsize=16, fontweight='bold')
        
        # --- Row 1: By Rel_Eps (Grouped Bar Chart) ---
        for idx, model in enumerate(models):
            ax = axes[0, idx]
            model_df = domain_df[domain_df['Model'] == model].copy()
            model_noise = domain_noise[domain_noise['Model'] == model].copy()
            
            if model_df['Rel_Eps'].notna().any():
                # Aggregate by epsilon and category
                summary = model_df.groupby(['Rel_Eps', 'Category'])['Output_Rel_L2'].agg(['mean', 'std']).reset_index()
                summary.columns = ['Rel_Eps', 'Category', 'Mean', 'Std']
                
                epsilons = sorted(summary['Rel_Eps'].unique())
                x = np.arange(len(epsilons))
                width = 0.35
                
                for i, cat in enumerate(['PGD-Pure', 'PGD-Stealth']):
                    cat_data = summary[summary['Category'] == cat]
                    if not cat_data.empty:
                        # Match epsilon order
                        vals = [cat_data[cat_data['Rel_Eps'] == e]['Mean'].values[0] if e in cat_data['Rel_Eps'].values else 0 for e in epsilons]
                        stds = [cat_data[cat_data['Rel_Eps'] == e]['Std'].values[0] if e in cat_data['Rel_Eps'].values else 0 for e in epsilons]
                        
                        offset = -width/2 if i == 0 else width/2
                        bars = ax.bar(x + offset, [v * 100 for v in vals], width, 
                                     label=cat, color=colors[cat], alpha=0.8,
                                     yerr=[s * 100 for s in stds], capsize=3)
                        
                        # Add value labels on bars
                        for bar, val in zip(bars, vals):
                            height = bar.get_height()
                            ax.annotate(f'{val*100:.1f}%',
                                       xy=(bar.get_x() + bar.get_width()/2, height),
                                       xytext=(0, 3), textcoords='offset points',
                                       ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Add noise reference lines
                if not model_noise.empty:
                    noise_by_target = model_noise.groupby('Rel_L2')['Output_Rel_L2'].mean()
                    for target_l2, noise_val in noise_by_target.items():
                        ax.axhline(y=noise_val * 100, color='#808080', linestyle=':', 
                                  linewidth=2, alpha=0.8, label=f'Noise (L2={target_l2})')
                
                # Add clean baseline line
                if model in clean_baseline:
                    ax.axhline(y=clean_baseline[model], color='#2E7D32', linestyle='--', 
                              linewidth=2, alpha=0.7, label=f'Clean: {clean_baseline[model]:.1f}%')
                
                ax.set_xticks(x)
                ax.set_xticklabels([f'ε={e}' for e in epsilons], fontsize=10)
                ax.set_ylabel('Output Rel L2 Error (%)', fontsize=11)
                ax.set_title(f'{model}: By Rel_Eps', fontsize=13, fontweight='bold')
                
                # Clean up legend
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=9)
                
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
            else:
                ax.text(0.5, 0.5, 'No epsilon data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model}: By Rel_Eps')
        
        # --- Row 2: By Input Perturbation (Bucketed Bar Chart) ---
        for idx, model in enumerate(models):
            ax = axes[1, idx]
            model_df = domain_df[domain_df['Model'] == model].copy()
            model_noise = domain_noise[domain_noise['Model'] == model].copy()
            
            if model_df['Input_Rel_Pert_L2'].notna().any():
                # Create buckets
                try:
                    model_df['Bucket'], bins = pd.qcut(model_df['Input_Rel_Pert_L2'], q=5, labels=False, retbins=True, duplicates='drop')
                except ValueError:
                    model_df['Bucket'], bins = pd.cut(model_df['Input_Rel_Pert_L2'], bins=5, labels=False, retbins=True)
                
                n_buckets = len(bins) - 1
                
                # Aggregate by bucket and category
                summary = model_df.groupby(['Bucket', 'Category'])['Output_Rel_L2'].agg(['mean', 'std']).reset_index()
                
                # Check if noise overlaps with PGD range
                has_noise_in_range = False
                if not model_noise.empty and model_noise['Input_Rel_Pert_L2'].notna().any():
                    noise_range = (model_noise['Input_Rel_Pert_L2'].min(), model_noise['Input_Rel_Pert_L2'].max())
                    pgd_range = (bins[0], bins[-1])
                    has_noise_in_range = not (noise_range[1] < pgd_range[0] or noise_range[0] > pgd_range[1])
                
                x = np.arange(n_buckets)
                n_categories = 3 if has_noise_in_range else 2
                width = 0.8 / n_categories
                
                # Plot bars for each category
                for i, cat in enumerate(['PGD-Pure', 'PGD-Stealth']):
                    cat_data = summary[summary['Category'] == cat]
                    if not cat_data.empty:
                        vals = [cat_data[cat_data['Bucket'] == b]['mean'].values[0] if b in cat_data['Bucket'].values else 0 for b in range(n_buckets)]
                        stds = [cat_data[cat_data['Bucket'] == b]['std'].values[0] if b in cat_data['Bucket'].values else 0 for b in range(n_buckets)]
                        
                        offset = (i - n_categories/2 + 0.5) * width
                        bars = ax.bar(x + offset, [v * 100 for v in vals], width,
                                     label=cat, color=colors[cat], alpha=0.8,
                                     yerr=[s * 100 for s in stds], capsize=3)
                        
                        # Add value labels
                        for bar, val in zip(bars, vals):
                            if val > 0:
                                height = bar.get_height()
                                ax.annotate(f'{val*100:.1f}%',
                                           xy=(bar.get_x() + bar.get_width()/2, height),
                                           xytext=(0, 3), textcoords='offset points',
                                           ha='center', va='bottom', fontsize=8, rotation=45)
                
                # Add noise bars if in range
                if has_noise_in_range:
                    noise_vals = []
                    for b_idx in range(n_buckets):
                        bin_low, bin_high = bins[b_idx], bins[b_idx + 1]
                        matching_noise = model_noise[
                            (model_noise['Input_Rel_Pert_L2'] >= bin_low * 0.8) & 
                            (model_noise['Input_Rel_Pert_L2'] <= bin_high * 1.2)
                        ]
                        if not matching_noise.empty:
                            noise_vals.append(matching_noise['Output_Rel_L2'].mean() * 100)
                        else:
                            noise_vals.append(0)
                    
                    offset = (2 - n_categories/2 + 0.5) * width
                    bars = ax.bar(x + offset, noise_vals, width,
                                 label='Noise', color='#808080', alpha=0.8)
                    
                    for bar, val in zip(bars, noise_vals):
                        if val > 0:
                            height = bar.get_height()
                            ax.annotate(f'{val:.1f}%',
                                       xy=(bar.get_x() + bar.get_width()/2, height),
                                       xytext=(0, 3), textcoords='offset points',
                                       ha='center', va='bottom', fontsize=8, rotation=45)
                
                # If noise doesn't fit, show as horizontal line
                elif not model_noise.empty:
                    noise_avg = model_noise['Output_Rel_L2'].mean() * 100
                    noise_pert_avg = model_noise['Input_Rel_Pert_L2'].mean() * 100
                    ax.axhline(y=noise_avg, color='#808080', linestyle='--', linewidth=2,
                              label=f'Noise (IC={noise_pert_avg:.1f}%)')
                
                # Add clean baseline line
                if model in clean_baseline:
                    ax.axhline(y=clean_baseline[model], color='#2E7D32', linestyle=':', 
                              linewidth=2, alpha=0.7, label=f'Clean: {clean_baseline[model]:.1f}%')
                
                # X-axis labels: show input perturbation ranges
                x_labels = []
                for i in range(n_buckets):
                    low, high = bins[i] * 100, bins[i+1] * 100
                    x_labels.append(f'{low:.1f}-{high:.1f}%')
                
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, fontsize=9, rotation=20, ha='right')
                ax.set_xlabel('Input Perturbation (Rel L2)', fontsize=11)
                ax.set_ylabel('Output Rel L2 Error (%)', fontsize=11)
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


def plot_stealth_analysis(df, plot_dir):
    """
    Plot 3: Stealth Analysis - Lambda_PDE vs Output Error
    Simple scatter/line plot with Lambda_PDE on x-axis
    """
    pgd_df = df[(df['Category'].isin(['PGD-Pure', 'PGD-Stealth'])) & (df['Mode'] == 'standard')].copy()
    
    if pgd_df.empty:
        print("   [SKIP] No PGD data for stealth analysis.")
        return
    
    models = sorted(pgd_df['Model'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Stealth Attack: Output Error vs Physics Regularization (λ_PDE)', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = pgd_df[pgd_df['Model'] == model]
        stealth_df = model_df[model_df['Category'] == 'PGD-Stealth']
        
        if not stealth_df.empty and stealth_df['Lambda_PDE'].notna().any():
            # Group by epsilon and plot lines
            for eps in sorted(stealth_df['Rel_Eps'].dropna().unique()):
                eps_df = stealth_df[stealth_df['Rel_Eps'] == eps].sort_values('Lambda_PDE')
                if len(eps_df) > 0:
                    ax.plot(eps_df['Lambda_PDE'], eps_df['Output_Rel_L2'] * 100, 
                           marker='o', markersize=8, linewidth=2, label=f'ε = {eps:.3f}')
            
            ax.set_xlabel('λ_PDE (Physics Regularization)', fontsize=11)
            ax.set_ylabel('Output Rel L2 Error (%)', fontsize=11)
            ax.set_title(f'{model}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add annotation about the tradeoff
            ax.annotate('Higher λ_PDE → More physics-consistent\nbut potentially less effective attack', 
                       xy=(0.95, 0.95), xycoords='axes fraction',
                       ha='right', va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # Fallback: compare pure vs stealth
            pure_err = model_df[model_df['Category'] == 'PGD-Pure']['Output_Rel_L2'].mean() * 100
            stealth_err = stealth_df['Output_Rel_L2'].mean() * 100 if not stealth_df.empty else 0
            
            bars = ax.bar(['Pure PGD', 'Stealth PGD'], [pure_err, stealth_err], 
                         color=['coral', 'steelblue'], alpha=0.7)
            ax.bar_label(bars, fmt='%.1f%%')
            ax.set_ylabel('Output Rel L2 Error (%)')
            ax.set_title(f'{model}: Pure vs Stealth')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "3_stealth_analysis.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 3_stealth_analysis.png")


def plot_perturbation_characteristics(df, plot_dir):
    """
    Plot 4: Perturbation Characteristics - Compare Stealth vs Pure
    """
    pgd_df = df[(df['Category'].isin(['PGD-Pure', 'PGD-Stealth'])) & (df['Mode'] == 'standard')].copy()
    
    if pgd_df.empty:
        print("   [SKIP] No PGD data for perturbation analysis.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Perturbation Characteristics: Stealth vs Pure', fontsize=14, fontweight='bold')
    
    # 4a. Smoothness comparison
    ax = axes[0, 0]
    if pgd_df['Pert_Smoothness'].notna().any():
        pivot = pgd_df.pivot_table(index='Model', columns='Category', values='Pert_Smoothness')
        if not pivot.empty:
            pivot.plot(kind='bar', ax=ax, color={'PGD-Pure': 'coral', 'PGD-Stealth': 'steelblue'})
            ax.set_ylabel('Smoothness (higher = smoother)')
            ax.set_title('Perturbation Smoothness')
            ax.legend(title='Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    else:
        ax.text(0.5, 0.5, 'No smoothness data', ha='center', va='center', transform=ax.transAxes)
    
    # 4b. Low vs High Frequency Ratio
    ax = axes[0, 1]
    if pgd_df['Pert_Low_Freq_Ratio'].notna().any():
        # Grouped bar: Low and High freq for Pure and Stealth
        data = []
        for cat in ['PGD-Pure', 'PGD-Stealth']:
            cat_df = pgd_df[pgd_df['Category'] == cat]
            data.append({
                'Category': cat,
                'Low Freq': cat_df['Pert_Low_Freq_Ratio'].mean() * 100,
                'High Freq': cat_df['Pert_High_Freq_Ratio'].mean() * 100
            })
        freq_df = pd.DataFrame(data).set_index('Category')
        freq_df.plot(kind='bar', ax=ax, color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('Energy Ratio (%)')
        ax.set_title('Perturbation Frequency Distribution')
        ax.legend(title='Frequency Band')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    else:
        ax.text(0.5, 0.5, 'No frequency data', ha='center', va='center', transform=ax.transAxes)
    
    # 4c. Sparsity comparison
    ax = axes[1, 0]
    if pgd_df['Pert_Sparsity'].notna().any():
        pivot = pgd_df.pivot_table(index='Model', columns='Category', values='Pert_Sparsity') * 100
        if not pivot.empty:
            pivot.plot(kind='bar', ax=ax, color={'PGD-Pure': 'coral', 'PGD-Stealth': 'steelblue'})
            ax.set_ylabel('Sparsity (%)')
            ax.set_title('Perturbation Sparsity (% near-zero elements)')
            ax.legend(title='Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    else:
        ax.text(0.5, 0.5, 'No sparsity data', ha='center', va='center', transform=ax.transAxes)
    
    # 4d. L-inf norm comparison
    ax = axes[1, 1]
    if pgd_df['Pert_Linf'].notna().any():
        pivot = pgd_df.pivot_table(index='Model', columns='Category', values='Pert_Linf')
        if not pivot.empty:
            pivot.plot(kind='bar', ax=ax, color={'PGD-Pure': 'coral', 'PGD-Stealth': 'steelblue'})
            ax.set_ylabel('Avg L-inf Norm')
            ax.set_title('Perturbation L-inf Norm')
            ax.legend(title='Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    else:
        ax.text(0.5, 0.5, 'No L-inf data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "4_perturbation_characteristics.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 4_perturbation_characteristics.png")


def plot_physics_loss(df, plot_dir):
    """
    Plot 5: Physics Loss Analysis - Rows by category, columns by model
    Also includes amplification ratio with output error magnitude
    """
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty or not std_df['PDE_Loss'].notna().any():
        print("   [SKIP] No physics loss data.")
        return
    
    models = sorted(std_df['Model'].unique())
    categories = ['Noise', 'PGD-Pure', 'PGD-Stealth']  # Main categories
    categories = [c for c in categories if c in std_df['Category'].unique()]
    
    if not categories:
        print("   [SKIP] No recognized categories for physics loss.")
        return
    
    # 3 rows: one per category for PDE loss, plus 1 row for amplification
    n_rows = len(categories) + 1
    fig, axes = plt.subplots(n_rows, len(models), figsize=(6 * len(models), 4 * n_rows))
    if len(models) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Physics Loss & Amplification by Attack Configuration', fontsize=14, fontweight='bold')
    
    category_colors = {
        'Noise': 'gray',
        'PGD-Pure': 'coral',
        'PGD-Stealth': 'steelblue',
    }
    
    # --- Rows 0 to len(categories)-1: PDE Loss by Category ---
    for row_idx, category in enumerate(categories):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            cat_df = std_df[(std_df['Model'] == model) & (std_df['Category'] == category)].copy()
            
            if cat_df.empty:
                ax.text(0.5, 0.5, f'No {category} data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model} - {category}')
                continue
            
            # Create labels with key parameters
            def make_label(row):
                if category == 'Noise':
                    if pd.notna(row.get('Rel_L2')):
                        return f"L2={row['Rel_L2']:.1%}"
                    return 'Noise'
                else:
                    parts = []
                    if pd.notna(row.get('Rel_Eps')):
                        parts.append(f"ε={row['Rel_Eps']:.3f}")
                    if pd.notna(row.get('Lambda_PDE')) and row.get('Lambda_PDE', 0) > 0:
                        parts.append(f"λ={row['Lambda_PDE']:.0f}")
                    return '\n'.join(parts) if parts else category
            
            cat_df['Label'] = cat_df.apply(make_label, axis=1)
            
            # Sort by parameter values
            if category == 'Noise':
                cat_df = cat_df.sort_values('Rel_L2')
            else:
                cat_df = cat_df.sort_values(['Rel_Eps', 'Lambda_PDE'], na_position='first')
            
            # Plot
            labels = cat_df['Label'].tolist()
            values = cat_df['PDE_Loss'].tolist()
            
            bars = ax.barh(range(len(labels)), values, color=category_colors.get(category, 'gray'), alpha=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('PDE Residual Loss')
            ax.set_title(f'{model} - {category}')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(val + max(values) * 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', va='center', fontsize=8)
    
    # --- Last Row: Amplification Ratio with Output Error ---
    for col_idx, model in enumerate(models):
        ax = axes[-1, col_idx]
        model_df = std_df[std_df['Model'] == model].copy()
        
        if model_df.empty or not model_df['Amp_Rel_Ratio'].notna().any():
            ax.text(0.5, 0.5, 'No amplification data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model} - Amplification')
            continue
        
        # Group by category and compute means
        summary = model_df.groupby('Category').agg({
            'Output_Rel_L2': 'mean',
            'Amp_Rel_Ratio': 'mean'
        }).reset_index()
        
        x = np.arange(len(summary))
        width = 0.35
        
        # Plot output error and amplification ratio side by side
        bars1 = ax.bar(x - width/2, summary['Output_Rel_L2'] * 100, width, 
                      label='Output Error (%)', color='indianred', alpha=0.7)
        
        # Create secondary y-axis for amplification ratio
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, summary['Amp_Rel_Ratio'], width,
                       label='Amp Ratio', color='steelblue', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(summary['Category'], rotation=15, ha='right')
        ax.set_ylabel('Output Rel L2 Error (%)', color='indianred')
        ax2.set_ylabel('Amplification Ratio', color='steelblue')
        ax.set_title(f'{model} - Output Error & Amplification')
        
        # Add horizontal line at amplification = 1
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Amp=1')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        # Add value labels
        for bar, val in zip(bars1, summary['Output_Rel_L2'] * 100):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
        for bar, val in zip(bars2, summary['Amp_Rel_Ratio']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "5_physics_loss.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 5_physics_loss.png")


def plot_frequency_scatter(df, plot_dir):
    """
    Plot 6: Frequency Leakage Scatterplot
    """
    std_df = df[df['Mode'] == 'standard'].copy()
    spec_df = std_df.dropna(subset=['In_High_Freq_Chg', 'Out_Low_Freq_Diff'])
    
    if spec_df.empty:
        print("   [SKIP] No spectral data.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    markers = {'Noise': 'o', 'PGD-Pure': 's', 'PGD-Stealth': '^', 'MVMO': 'D', 'Boundary': 'v', 'Other': 'x'}
    colors = {'FNO': 'tab:blue', 'FFNO': 'tab:orange', 'CNO': 'tab:green'}
    
    for model in spec_df['Model'].unique():
        for cat in spec_df['Category'].unique():
            sub = spec_df[(spec_df['Model'] == model) & (spec_df['Category'] == cat)]
            if not sub.empty:
                ax.scatter(sub['In_High_Freq_Chg'] * 100, sub['Out_Low_Freq_Diff'] * 100,
                          c=colors.get(model, 'gray'), marker=markers.get(cat, 'o'),
                          s=100, alpha=0.7, label=f'{model} - {cat}')
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    corr = spec_df['In_High_Freq_Chg'].corr(spec_df['Out_Low_Freq_Diff'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Input High-Freq Change (%)', fontsize=12)
    ax.set_ylabel('Output Low-Freq Diff (%)', fontsize=12)
    ax.set_title('Frequency Leakage: High-Freq Input → Low-Freq Output', fontsize=14, fontweight='bold')
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=m) 
               for m, c in colors.items()]
    ax.legend(handles=handles, title='Model', loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "6_frequency_scatter.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 6_frequency_scatter.png")


def plot_amplification_ranking(df, plot_dir, top_n=15):
    """
    Plot 7: Amplification Ranking (per model)
    Shows experiments ranked by amplification ratio.
    One plot per model (CNO, FNO, FFNO).
    Shows input perturbation magnitude and output error magnitude side by side.
    Color-coded by attack type.
    Also outputs data as text/CSV files.
    
    Args:
        df: DataFrame with benchmark results
        plot_dir: Output directory for plots
        top_n: Number of top experiments to show (default 15)
    """
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty or not std_df['Amp_Rel_Ratio'].notna().any():
        print("   [SKIP] No amplification data for ranking.")
        return
    
    # Filter to rows with valid data
    plot_df = std_df[std_df['Amp_Rel_Ratio'].notna() & 
                     std_df['Input_Rel_Pert_L2'].notna() & 
                     std_df['Output_Rel_L2'].notna()].copy()
    
    if plot_df.empty:
        print("   [SKIP] No complete data for amplification ranking.")
        return
    
    # Color scheme by category
    category_colors = {
        'Noise': '#808080',        # Gray
        'PGD-Pure': '#E07A5F',     # Coral
        'PGD-Stealth': '#3D7C98',  # Teal
        'MVMO': '#81B29A',         # Sage green
        'Boundary': '#9B5DE5',     # Purple
        'Other': '#F4A261'         # Orange
    }
    
    models = sorted(plot_df['Model'].unique())
    
    # Also create a combined text report
    report_lines = []
    report_lines.append("=" * 110)
    report_lines.append("AMPLIFICATION RANKING REPORT")
    report_lines.append("=" * 110)
    report_lines.append("")
    
    for model in models:
        model_df = plot_df[plot_df['Model'] == model].copy()
        
        if model_df.empty:
            continue
        
        # Create experiment label with domain info (Spatial/Spectral)
        def make_label(row):
            parts = [row['Category']]
            # Add domain (Spatial/Spectral) - abbreviated
            if pd.notna(row.get('Domain')) and row.get('Domain') != 'Unknown':
                parts.append(row['Domain'][:4])  # "Spat" or "Spec"
            if pd.notna(row.get('Rel_Eps')) and row['Category'] not in ['Noise']:
                parts.append(f"ε={row['Rel_Eps']:.3f}")
            if pd.notna(row.get('Lambda_PDE')) and row.get('Lambda_PDE', 0) > 0:
                parts.append(f"λ={row['Lambda_PDE']:.0f}")
            if pd.notna(row.get('Rel_L2')) and row['Category'] == 'Noise':
                parts.append(f"L2={row['Rel_L2']:.1%}")
            return ' | '.join(parts)
        
        model_df['Label'] = model_df.apply(make_label, axis=1)
        
        # Sort by amplification ratio (descending - most vulnerable first)
        model_df = model_df.sort_values('Amp_Rel_Ratio', ascending=False)
        
        # === Generate Full Text Report for this model ===
        report_lines.append(f"{'=' * 110}")
        report_lines.append(f"MODEL: {model}")
        report_lines.append(f"{'=' * 110}")
        report_lines.append("")
        report_lines.append(f"{'Rank':<6} {'Experiment':<50} {'Input Pert %':<14} {'Output Err %':<14} {'Amplification':<14}")
        report_lines.append(f"{'-' * 6} {'-' * 50} {'-' * 14} {'-' * 14} {'-' * 14}")
        
        for rank, (idx, row) in enumerate(model_df.iterrows(), 1):
            input_pct = row['Input_Rel_Pert_L2'] * 100
            output_pct = row['Output_Rel_L2'] * 100
            amp = row['Amp_Rel_Ratio']
            label = row['Label'][:48] + '..' if len(row['Label']) > 50 else row['Label']
            
            report_lines.append(f"{rank:<6} {label:<50} {input_pct:<14.2f} {output_pct:<14.2f} {amp:<14.2f}x")
        
        report_lines.append("")
        
        # Summary stats
        report_lines.append(f"Summary for {model}:")
        report_lines.append(f"  Total experiments: {len(model_df)}")
        report_lines.append(f"  Avg Input Perturbation: {model_df['Input_Rel_Pert_L2'].mean() * 100:.2f}%")
        report_lines.append(f"  Avg Output Error: {model_df['Output_Rel_L2'].mean() * 100:.2f}%")
        report_lines.append(f"  Avg Amplification: {model_df['Amp_Rel_Ratio'].mean():.2f}x")
        report_lines.append(f"  Max Amplification: {model_df['Amp_Rel_Ratio'].max():.2f}x ({model_df.loc[model_df['Amp_Rel_Ratio'].idxmax(), 'Label']})")
        report_lines.append(f"  Min Amplification: {model_df['Amp_Rel_Ratio'].min():.2f}x ({model_df.loc[model_df['Amp_Rel_Ratio'].idxmin(), 'Label']})")
        report_lines.append("")
        
        # === Save Full CSV for this model ===
        csv_df = model_df[['Label', 'Category', 'Domain', 'Rel_Eps', 'Lambda_PDE', 'Rel_L2', 
                           'Input_Rel_Pert_L2', 'Output_Rel_L2', 'Amp_Rel_Ratio']].copy()
        csv_df['Input_Rel_Pert_L2'] = csv_df['Input_Rel_Pert_L2'] * 100
        csv_df['Output_Rel_L2'] = csv_df['Output_Rel_L2'] * 100
        csv_df.columns = ['Experiment', 'Category', 'Domain', 'Rel_Eps', 'Lambda_PDE', 'Rel_L2',
                         'Input_Pert_%', 'Output_Err_%', 'Amplification']
        csv_filename = f"7_amplification_{model.lower()}.csv"
        csv_df.to_csv(os.path.join(plot_dir, csv_filename), index=False)
        print(f"   [SAVED] {csv_filename}")
        
        # === Generate Plot (TOP N only) ===
        # Take top N by amplification for plotting
        plot_model_df = model_df.head(top_n).copy()
        
        # Re-sort for plotting (ascending for bottom-to-top display)
        plot_model_df = plot_model_df.sort_values('Amp_Rel_Ratio', ascending=True)
        
        n_experiments = len(plot_model_df)
        fig_height = max(6, n_experiments * 0.5)
        
        fig, axes = plt.subplots(1, 3, figsize=(14, fig_height), 
                                 gridspec_kw={'width_ratios': [2, 2, 1.5]})
        
        title_suffix = f" (Top {top_n})" if len(model_df) > top_n else ""
        fig.suptitle(f'{model}: Experiments Ranked by Amplification{title_suffix}', 
                    fontsize=14, fontweight='bold')
        
        y_pos = np.arange(n_experiments)
        bar_height = 0.7
        
        # Get colors for each row
        colors = [category_colors.get(cat, '#808080') for cat in plot_model_df['Category']]
        
        # === Panel 1: Input Perturbation ===
        ax1 = axes[0]
        input_vals = plot_model_df['Input_Rel_Pert_L2'].values * 100
        bars1 = ax1.barh(y_pos, input_vals, height=bar_height, color=colors, alpha=0.8, edgecolor='white')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(plot_model_df['Label'], fontsize=10)
        ax1.set_xlabel('Input Perturbation (Rel L2 %)', fontsize=11)
        ax1.set_title('Input Perturbation', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # === Panel 2: Output Error ===
        ax2 = axes[1]
        output_vals = plot_model_df['Output_Rel_L2'].values * 100
        bars2 = ax2.barh(y_pos, output_vals, height=bar_height, color=colors, alpha=0.8, edgecolor='white')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])  # Hide y labels (shown in panel 1)
        ax2.set_xlabel('Output Error (Rel L2 %)', fontsize=11)
        ax2.set_title('Output Error', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        # === Panel 3: Amplification Ratio ===
        ax3 = axes[2]
        amp_vals = plot_model_df['Amp_Rel_Ratio'].values
        
        # Color based on amplification (green < 1, red > 1)
        amp_colors = ['#2E7D32' if v < 1 else '#C62828' for v in amp_vals]
        bars3 = ax3.barh(y_pos, amp_vals, height=bar_height, color=amp_colors, alpha=0.8, edgecolor='white')
        
        # Add vertical line at 1.0
        ax3.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([])  # Hide y labels
        ax3.set_xlabel('Amplification Ratio', fontsize=11)
        ax3.set_title('Amplification', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        ax3.set_axisbelow(True)
        
        # === Legend ===
        present_categories = plot_model_df['Category'].unique()
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=category_colors.get(c, '#808080'), alpha=0.8) 
                         for c in present_categories]
        legend_labels = list(present_categories)
        
        # Add amplification color legend
        legend_handles.extend([
            plt.Rectangle((0, 0), 1, 1, color='#2E7D32', alpha=0.8),
            plt.Rectangle((0, 0), 1, 1, color='#C62828', alpha=0.8)
        ])
        legend_labels.extend(['Amp < 1 (Robust)', 'Amp > 1 (Vulnerable)'])
        
        fig.legend(legend_handles, legend_labels, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 5), fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        # Save with model name in filename (now using 7_ prefix)
        filename = f"7_amplification_{model.lower()}.png"
        plt.savefig(os.path.join(plot_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   [SAVED] {filename}")
    
    # Save combined text report
    report_lines.append("=" * 110)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 110)
    
    report_filename = "7_amplification_report.txt"
    with open(os.path.join(plot_dir, report_filename), 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"   [SAVED] {report_filename}")


def plot_cross_resolution(df, plot_dir):
    """
    Plot 10: Cross-Resolution Analysis
    Shows how PGD pure attacks transfer across different resolutions.
    Only for PGD pure attacks (pgd_spatial, pgd_spectral - not stealth).
    """
    # Filter for cross-resolution results
    cross_res_df = df[df['Mode'].astype(str).str.contains('cross_res', na=False)].copy()
    
    if cross_res_df.empty:
        print("   [SKIP] No cross-resolution data.")
        return
    
    # Get ALL noise results (standard + cross_res)
    # Get models from cross-res data
    models = sorted(cross_res_df['Model'].unique())
    
    # Prepare data for plotting - group by model, attack, target_l2
    plot_data = []
    
    for model in models:
        # Use FULL df, not filtered by category - match by attack name instead
        model_df = df[df['Model'] == model]
        
        # Get unique attack names from cross-res results for this model
        model_cross = cross_res_df[cross_res_df['Model'] == model]
        attack_names = model_cross['Attack_Name'].unique()
        
        for attack in attack_names:
            # Get ALL results for this attack (any mode)
            attack_df = model_df[model_df['Attack_Name'] == attack]
            
            # Get standard (base resolution) error
            std_results = attack_df[attack_df['Mode'] == 'standard']
            std_err = std_results['Output_Rel_L2'].mean() if not std_results.empty else np.nan
            
            # Get low resolution error
            low_results = attack_df[attack_df['Mode'] == 'cross_res_low']
            low_err = low_results['Output_Rel_L2'].mean() if not low_results.empty else np.nan
            
            # Get high resolution error
            high_results = attack_df[attack_df['Mode'] == 'cross_res_high']
            high_err = high_results['Output_Rel_L2'].mean() if not high_results.empty else np.nan
            
            # Get epsilon (for PGD) and resolution info
            epsilon = attack_df['Rel_Eps'].iloc[0] if not attack_df.empty and pd.notna(attack_df['Rel_Eps'].iloc[0]) else np.nan
            base_res = attack_df['Base_Res'].dropna().iloc[0] if attack_df['Base_Res'].notna().any() else np.nan
            
            domain = 'Spectral' if 'spectral' in attack.lower() else 'Spatial'
            
            plot_data.append({
                'Model': model,
                'Attack': attack,
                'Domain': domain,
                'Rel_Eps': epsilon,
                'Base_Res': base_res,
                'Err_Standard': std_err,
                'Err_Low_Res': low_err,
                'Err_High_Res': high_err,
            })
    
    cross_df = pd.DataFrame(plot_data)
    
    if cross_df.empty:
        print("   [SKIP] No cross-resolution comparison data.")
        return
    
    # Debug output
    print(f"   Cross-resolution data summary:")
    print(f"   - Total entries: {len(cross_df)}")
    print(f"   - Standard non-null: {cross_df['Err_Standard'].notna().sum()}")
    print(f"   - Low Res non-null: {cross_df['Err_Low_Res'].notna().sum()}")
    print(f"   - High Res non-null: {cross_df['Err_High_Res'].notna().sum()}")
    
    # Save CSV
    csv_path = os.path.join(plot_dir, "10_cross_resolution.csv")
    cross_df.to_csv(csv_path, index=False)
    print(f"   [SAVED] 10_cross_resolution.csv")
    
    # === Create Plot ===
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 8))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Cross-Resolution Analysis: PGD Attack Transferability', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = cross_df[cross_df['Model'] == model].copy()
        
        if model_data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model}')
            continue
        
        # Sort by domain then epsilon
        model_data = model_data.sort_values(['Domain', 'Rel_Eps'])
        
        n_attacks = len(model_data)
        x = np.arange(n_attacks)
        width = 0.25
        
        # Plot three bars per attack: Low, Standard, High
        has_low = model_data['Err_Low_Res'].notna().any()
        has_std = model_data['Err_Standard'].notna().any()
        has_high = model_data['Err_High_Res'].notna().any()
        
        if has_low:
            low_vals = model_data['Err_Low_Res'].fillna(0) * 100
            bars_low = ax.bar(x - width, low_vals, width, 
                             label='Low Res', color='#90BE6D', alpha=0.8)
            # Add value labels
            for bar, val in zip(bars_low, low_vals):
                if val > 0:
                    ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=7, rotation=90)
        
        if has_std:
            std_vals = model_data['Err_Standard'].fillna(0) * 100
            bars_std = ax.bar(x, std_vals, width,
                             label='Standard (Base)', color='#577590', alpha=0.8)
            # Add value labels
            for bar, val in zip(bars_std, std_vals):
                if val > 0:
                    ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=7, rotation=90)
        
        if has_high:
            high_vals = model_data['Err_High_Res'].fillna(0) * 100
            bars_high = ax.bar(x + width, high_vals, width,
                              label='High Res', color='#F94144', alpha=0.8)
            # Add value labels
            for bar, val in zip(bars_high, high_vals):
                if val > 0:
                    ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=7, rotation=90)
        
        # X-axis labels
        x_labels = []
        for _, row in model_data.iterrows():
            domain = row['Domain']
            eps = row['Rel_Eps']
            if pd.notna(eps):
                x_labels.append(f"{domain}\nε={eps:.3f}")
            else:
                x_labels.append(domain)
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel('Output Rel L2 Error (%)', fontsize=11)
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "10_cross_resolution.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 10_cross_resolution.png")
    
    # === Generate text report ===
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("CROSS-RESOLUTION ANALYSIS REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("This report shows how PGD pure perturbations transfer across resolutions.")
    report_lines.append("Lower error at different resolutions suggests the perturbation is")
    report_lines.append("resolution-dependent and may not transfer well.")
    report_lines.append("")
    
    for model in models:
        model_data = cross_df[cross_df['Model'] == model]
        
        report_lines.append(f"\n{'=' * 100}")
        report_lines.append(f"MODEL: {model}")
        report_lines.append(f"{'=' * 100}")
        report_lines.append("")
        report_lines.append(f"{'Attack':<40} {'Rel_Eps':<12} {'Low Res %':<12} {'Std Res %':<12} {'High Res %':<12}")
        report_lines.append(f"{'-' * 40} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
        
        for _, row in model_data.iterrows():
            attack = row['Attack'][:38] + '..' if len(row['Attack']) > 40 else row['Attack']
            eps = f"{row['Rel_Eps']:.3f}" if pd.notna(row['Rel_Eps']) else 'N/A'
            low = f"{row['Err_Low_Res']*100:.2f}%" if pd.notna(row['Err_Low_Res']) else 'N/A'
            std = f"{row['Err_Standard']*100:.2f}%" if pd.notna(row['Err_Standard']) else 'N/A'
            high = f"{row['Err_High_Res']*100:.2f}%" if pd.notna(row['Err_High_Res']) else 'N/A'
            report_lines.append(f"{attack:<40} {eps:<12} {low:<12} {std:<12} {high:<12}")
    
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 100)
    
    report_path = os.path.join(plot_dir, "10_cross_resolution_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"   [SAVED] 10_cross_resolution_report.txt")


def plot_clean_baseline(df, plot_dir):
    """
    Plot 11: Clean Baseline Comparison
    Shows clean (non-adversarial) error as baseline compared to attacks.
    """
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty:
        print("   [SKIP] No data for clean baseline plot.")
        return
    
    # Check if we have clean data
    clean_df = std_df[std_df['Category'] == 'Clean']
    if clean_df.empty:
        print("   [SKIP] No clean baseline data found.")
        return
    
    models = sorted(std_df['Model'].unique())
    
    # Get clean baseline per model
    clean_baseline = clean_df.groupby('Model')['Output_Rel_L2'].mean() * 100
    
    # Get attack categories (excluding Clean)
    attack_df = std_df[std_df['Category'] != 'Clean']
    categories = sorted(attack_df['Category'].unique())
    
    # Color scheme
    category_colors = {
        'Clean': '#2E7D32',        # Green
        'Noise': '#808080',        # Gray
        'PGD-Pure': '#E07A5F',     # Coral
        'PGD-Stealth': '#3D7C98',  # Teal
        'MVMO': '#81B29A',         # Sage green
        'Boundary': '#9B5DE5',     # Purple
        'Other': '#F4A261'         # Orange
    }
    
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 8))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Output Error: Clean Baseline vs Adversarial Attacks', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = attack_df[attack_df['Model'] == model]
        
        # Get clean baseline for this model
        model_clean = clean_baseline.get(model, np.nan)
        
        # Group by category
        cat_errors = model_df.groupby('Category')['Output_Rel_L2'].agg(['mean', 'std', 'min', 'max']) * 100
        
        # Add clean as first category
        all_cats = ['Clean'] + list(cat_errors.index)
        n_cats = len(all_cats)
        x = np.arange(n_cats)
        
        # Prepare data
        means = [model_clean] + list(cat_errors['mean'])
        stds = [0] + list(cat_errors['std'].fillna(0))
        colors = [category_colors.get(c, '#808080') for c in all_cats]
        
        # Plot bars
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5, edgecolor='white')
        
        # Add horizontal line for clean baseline
        if pd.notna(model_clean):
            ax.axhline(y=model_clean, color='#2E7D32', linestyle='--', linewidth=2, alpha=0.7, label='Clean Baseline')
        
        # Add value labels
        for bar, val in zip(bars, means):
            if pd.notna(val) and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + max(means)*0.02,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_cats, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Output Rel L2 Error (%)', fontsize=11)
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "11_clean_baseline.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 11_clean_baseline.png")
    
    # === Generate CSV ===
    csv_data = []
    for model in models:
        model_clean = clean_baseline.get(model, np.nan)
        csv_data.append({
            'Model': model,
            'Category': 'Clean',
            'Mean_Error_%': model_clean,
            'Std_Error_%': 0,
            'Error_Increase_%': 0
        })
        
        model_attack = attack_df[attack_df['Model'] == model]
        for cat in categories:
            cat_data = model_attack[model_attack['Category'] == cat]
            if not cat_data.empty:
                mean_err = cat_data['Output_Rel_L2'].mean() * 100
                std_err = cat_data['Output_Rel_L2'].std() * 100
                increase = mean_err - model_clean if pd.notna(model_clean) else np.nan
                csv_data.append({
                    'Model': model,
                    'Category': cat,
                    'Mean_Error_%': mean_err,
                    'Std_Error_%': std_err,
                    'Error_Increase_%': increase
                })
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(os.path.join(plot_dir, "11_clean_baseline.csv"), index=False)
    print(f"   [SAVED] 11_clean_baseline.csv")
    
    # === Generate text report ===
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CLEAN BASELINE COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for model in models:
        model_clean = clean_baseline.get(model, np.nan)
        report_lines.append(f"\nMODEL: {model}")
        report_lines.append("-" * 40)
        report_lines.append(f"Clean Baseline Error: {model_clean:.2f}%")
        report_lines.append("")
        report_lines.append(f"{'Category':<20} {'Mean Error %':<15} {'Increase %':<15}")
        report_lines.append(f"{'-'*20} {'-'*15} {'-'*15}")
        
        model_attack = attack_df[attack_df['Model'] == model]
        for cat in categories:
            cat_data = model_attack[model_attack['Category'] == cat]
            if not cat_data.empty:
                mean_err = cat_data['Output_Rel_L2'].mean() * 100
                increase = mean_err - model_clean if pd.notna(model_clean) else np.nan
                report_lines.append(f"{cat:<20} {mean_err:<15.2f} {'+' if increase > 0 else ''}{increase:<15.2f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    with open(os.path.join(plot_dir, "11_clean_baseline_report.txt"), 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"   [SAVED] 11_clean_baseline_report.txt")


def plot_summary_by_model(df, plot_dir):
    """
    Plot 7: Summary - Relative Amplification Ratio by Attack Type per Model
    """
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty:
        print("   [SKIP] No data for summary.")
        return
    
    models = sorted(std_df['Model'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Relative Amplification Ratio by Attack Type', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = std_df[std_df['Model'] == model]
        
        cat_summary = model_df.groupby('Category')['Amp_Rel_Ratio'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        
        if not cat_summary.empty:
            colors = ['green' if v < 1 else 'red' for v in cat_summary['mean']]
            bars = ax.barh(cat_summary.index, cat_summary['mean'], 
                          xerr=cat_summary['std'], color=colors, alpha=0.7, capsize=3)
            ax.axvline(x=1.0, color='k', linestyle='--', linewidth=2)
            ax.set_xlabel('Relative Amplification Ratio')
            ax.set_title(f'{model}')
            
            for bar, val in zip(bars, cat_summary['mean']):
                ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
                       f'{val:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "7_summary_by_model.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 7_summary_by_model.png")


def plot_summary_heatmap(df, plot_dir):
    """
    Plot 8: Summary Heatmap
    """
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty:
        print("   [SKIP] No data for heatmap.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pivot = std_df.pivot_table(index='Category', columns='Model', values='Amp_Rel_Ratio', aggfunc='mean')
    
    if not pivot.empty:
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', center=1.0, ax=ax,
                   cbar_kws={'label': 'Relative Amplification Ratio'})
        ax.set_title('Relative Amplification Ratio: Attack Type vs Model\n(Green < 1 = Robust, Red > 1 = Vulnerable)', 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "8_summary_heatmap.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 8_summary_heatmap.png")


def plot_absolute_perturbation_comparison(df, plot_dir):
    """
    Plot 12: Absolute Perturbation Comparison
    Compare all attack types on the same absolute input perturbation scale.
    Creates scatter plots and efficiency curves showing output error vs absolute input L2.
    """
    std_df = df[df['Mode'] == 'standard'].copy()
    
    if std_df.empty:
        print("   [SKIP] No data for absolute comparison.")
        return
    
    # Filter to rows with valid perturbation data
    plot_df = std_df[std_df['Input_Abs_Pert_L2'].notna() & 
                     std_df['Output_Rel_L2'].notna()].copy()
    
    if plot_df.empty:
        print("   [SKIP] No perturbation data for absolute comparison.")
        return
    
    # Color scheme by category
    category_colors = {
        'Noise': '#808080',
        'PGD-Pure': '#E07A5F',
        'PGD-Stealth': '#3D7C98',
        'MVMO': '#81B29A',
        'Boundary': '#9B5DE5',
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
    
    # === Plot 12a: Scatter plot - Output Error vs Absolute Input Perturbation ===
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
                
                ax1.scatter(dom_df['Input_Abs_Pert_L2'], 
                           dom_df['Output_Rel_L2'] * 100,
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
                    efficiency = (dom_df['Output_Rel_L2'] * 100) / (dom_df['Input_Abs_Pert_L2'] + 1e-8)
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
        plt.savefig(os.path.join(plot_dir, f"12a_abs_comparison_{model.lower()}.png"), dpi=150)
        plt.close()
        print(f"   [SAVED] 12a_abs_comparison_{model.lower()}.png")
    
    # === Plot 12b: Combined view - All models, binned by perturbation magnitude ===
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    fig.suptitle('Output Error vs Absolute Input Perturbation (All Attack Types)', 
                fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = plot_df[plot_df['Model'] == model]
        
        if model_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model)
            continue
        
        # Create bins based on absolute perturbation
        try:
            model_df['Pert_Bin'], bins = pd.qcut(model_df['Input_Abs_Pert_L2'], q=5, 
                                                  labels=False, retbins=True, duplicates='drop')
        except ValueError:
            model_df['Pert_Bin'], bins = pd.cut(model_df['Input_Abs_Pert_L2'], bins=5, 
                                                 labels=False, retbins=True)
        
        n_bins = len(bins) - 1
        x = np.arange(n_bins)
        
        # Get unique categories
        categories = sorted(model_df['Category'].unique())
        width = 0.8 / len(categories)
        
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
                    bin_means.append(bin_df['Output_Rel_L2'].mean() * 100)
                    bin_stds.append(bin_df['Output_Rel_L2'].std() * 100)
                else:
                    bin_means.append(0)
                    bin_stds.append(0)
            
            offset = (i - len(categories)/2 + 0.5) * width
            color = category_colors.get(category, '#808080')
            ax.bar(x + offset, bin_means, width, label=category, color=color, alpha=0.8,
                  yerr=bin_stds, capsize=2)
        
        # X-axis labels
        x_labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(n_bins)]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha='right')
        ax.set_xlabel('Absolute Input Perturbation (L2)', fontsize=10)
        ax.set_ylabel('Output Rel L2 Error (%)', fontsize=10)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "12b_abs_comparison_binned.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 12b_abs_comparison_binned.png")
    
    # === Plot 12c: Line plot - Output error vs perturbation for each category ===
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
            cat_df = model_df[model_df['Category'] == category].sort_values('Input_Abs_Pert_L2')
            
            if len(cat_df) < 2:
                continue
            
            color = category_colors.get(category, '#808080')
            
            # Group by perturbation level and compute mean
            cat_df['Pert_Group'] = pd.cut(cat_df['Input_Abs_Pert_L2'], bins=10, labels=False)
            grouped = cat_df.groupby('Pert_Group').agg({
                'Input_Abs_Pert_L2': 'mean',
                'Output_Rel_L2': ['mean', 'std']
            }).dropna()
            
            if len(grouped) > 1:
                x_vals = grouped['Input_Abs_Pert_L2']['mean']
                y_vals = grouped['Output_Rel_L2']['mean'] * 100
                y_err = grouped['Output_Rel_L2']['std'] * 100
                
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
    plt.savefig(os.path.join(plot_dir, "12c_abs_comparison_curves.png"), dpi=150)
    plt.close()
    print(f"   [SAVED] 12c_abs_comparison_curves.png")


def print_summary_table(df):
    """Print formatted summary."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    std_df = df[df['Mode'] == 'standard']
    
    if std_df.empty:
        print("No results found.")
        return
    
    # Model summary
    print("\n--- Model Performance ---")
    model_summary = std_df.groupby('Model').agg({
        'Input_Rel_Pert_L2': 'mean',
        'Output_Rel_L2': 'mean',
        'Amp_Rel_Ratio': ['mean', 'std'],
        'PDE_Loss': 'mean'
    })
    model_summary.columns = ['Avg Input Pert', 'Avg Output Error', 'Amp Ratio (Mean)', 'Amp Ratio (Std)', 'Avg PDE Loss']
    model_summary['Avg Input Pert'] = (model_summary['Avg Input Pert'] * 100).round(2).astype(str) + '%'
    model_summary['Avg Output Error'] = (model_summary['Avg Output Error'] * 100).round(2).astype(str) + '%'
    print(model_summary.round(4).to_string())
    
    # Perturbation characteristics
    pert_cols = ['Pert_Smoothness', 'Pert_Low_Freq_Ratio', 'Pert_Sparsity']
    if all(col in std_df.columns for col in pert_cols):
        print("\n--- Perturbation Characteristics (Stealth vs Pure) ---")
        pert_summary = std_df[std_df['Category'].isin(['PGD-Pure', 'PGD-Stealth'])].groupby('Category').agg({
            'Pert_Smoothness': 'mean',
            'Pert_Low_Freq_Ratio': 'mean',
            'Pert_Sparsity': 'mean'
        })
        pert_summary['Pert_Low_Freq_Ratio'] = (pert_summary['Pert_Low_Freq_Ratio'] * 100).round(1).astype(str) + '%'
        pert_summary['Pert_Sparsity'] = (pert_summary['Pert_Sparsity'] * 100).round(1).astype(str) + '%'
        print(pert_summary.round(2).to_string())
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather_only", action="store_true")
    parser.add_argument("--output_dir", default="simple/darcy_2d/benchmark_results")
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

    df = aggregate_stats()
    
    if df.empty:
        print("[WARNING] No results to aggregate.")
        return
    
    csv_path = os.path.join(args.output_dir, "benchmark_metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    print_summary_table(df)
    
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots in: {plot_dir}")
    
    try:
        plot_noise_impact(df, plot_dir)               # 1
        plot_pgd_by_model(df, plot_dir)               # 2
        plot_stealth_analysis(df, plot_dir)           # 3
        plot_perturbation_characteristics(df, plot_dir) # 4
        plot_physics_loss(df, plot_dir)               # 5
        plot_frequency_scatter(df, plot_dir)          # 6
        plot_summary_by_model(df, plot_dir)           # 7
        plot_summary_heatmap(df, plot_dir)            # 8
        plot_amplification_ranking(df, plot_dir)      # 9
        plot_cross_resolution(df, plot_dir)           # 10
        plot_clean_baseline(df, plot_dir)             # 11
        plot_absolute_perturbation_comparison(df, plot_dir)  # 12
    except Exception as e:
        print(f"[WARNING] Plotting failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nBenchmark complete!")


if __name__ == "__main__":
    main()