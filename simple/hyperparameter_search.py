"""
Hyperparameter search for FNO, FFNO, and CNO models using random search.

Usage:
    python3 -m simple.hyperparameter_search \
        --problem="simple_darcy_2d" \
        --model_type="fno" \
        --base_model_config="simple/darcy_2d/fno_model.toml" \
        --data_config="simple/darcy_2d/data.toml" \
        --output_dir="simple/darcy_2d/hp_search" \
        --n_trials=50 \
        --epochs_per_trial=50
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class SearchSpace:
    """Define hyperparameter search spaces for different model types."""
    
    # Common parameters for all models
    LEARNING_RATE = [1e-4, 5e-4, 1e-3, 2e-3]
    BATCH_SIZE = [16, 32, 64]
    WEIGHT_DECAY = [0, 1e-6, 1e-5, 1e-4]
    STEP_SIZE = [20, 30, 50]
    GAMMA = [0.3, 0.5, 0.7]
    
    # FNO/FFNO
    FNO_MODES = [12, 16, 20, 24, 32]
    FNO_WIDTH = [32, 64, 128]
    FNO_LAYERS = [3, 4, 5, 6]
    
    # CNO
    CNO_LAYERS = [3, 4, 5, 6]
    
    @staticmethod
    def sample_fno_config() -> Dict[str, Any]:
        """Sample hyperparameters for FNO/FFNO."""
        return {
            'modes': random.choice(SearchSpace.FNO_MODES),
            'width': random.choice(SearchSpace.FNO_WIDTH),
            'n_layers': random.choice(SearchSpace.FNO_LAYERS),
            'learning_rate': random.choice(SearchSpace.LEARNING_RATE),
            'batch_size': random.choice(SearchSpace.BATCH_SIZE),
            'weight_decay': random.choice(SearchSpace.WEIGHT_DECAY),
            'step_size': random.choice(SearchSpace.STEP_SIZE),
            'gamma': random.choice(SearchSpace.GAMMA),
        }
    
    @staticmethod
    def sample_cno_config() -> Dict[str, Any]:
        """Sample hyperparameters for CNO."""
        return {
            'n_layers': random.choice(SearchSpace.CNO_LAYERS),
            'learning_rate': random.choice(SearchSpace.LEARNING_RATE),
            'batch_size': random.choice(SearchSpace.BATCH_SIZE),
            'weight_decay': random.choice(SearchSpace.WEIGHT_DECAY),
            'step_size': random.choice(SearchSpace.STEP_SIZE),
            'gamma': random.choice(SearchSpace.GAMMA),
        }


class HyperparameterSearcher:
    """Perform hyperparameter search for neural operator models."""
    
    def __init__(
        self,
        problem: str,
        model_type: str,
        base_config_path: str,
        data_config_path: str,
        output_dir: str,
        n_trials: int = 50,
        epochs_per_trial: int = 50,
        device: str = 'cuda'
    ):
        self.problem = problem
        self.model_type = model_type.lower()
        self.base_config_path = base_config_path
        self.data_config_path = data_config_path
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        self.epochs_per_trial = epochs_per_trial
        self.device = device
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.output_dir / "configs"
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        
        self.configs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Load base configuration
        with open(self.base_config_path, 'rb') as f:
            self.base_config = tomllib.load(f)
        
        # Results storage
        self.results = []
        
        print(f"\n{'='*80}")
        print(f"Hyperparameter Search Configuration")
        print(f"{'='*80}")
        print(f"Problem: {self.problem}")
        print(f"Model Type: {self.model_type}")
        print(f"Number of Trials: {self.n_trials}")
        print(f"Epochs per Trial: {self.epochs_per_trial}")
        print(f"Output Directory: {self.output_dir}")
        print(f"NOTE: physics_importance and boundary_condition_weight fixed at 0.0")
        print(f"{'='*80}\n")
    
    def sample_config(self) -> Dict[str, Any]:
        """Sample a random configuration based on model type."""
        if self.model_type in ['fno', 'fno2d', 'ffno', 'ffno2d']:
            return SearchSpace.sample_fno_config()
        elif self.model_type in ['cno', 'cno2d']:
            return SearchSpace.sample_cno_config()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def create_trial_config(self, trial_id: int, sampled_params: Dict[str, Any]) -> Path:
        """
        Create a TOML config file for this trial by updating the base config.
        
        Args:
            trial_id: Trial number
            sampled_params: Dictionary of sampled hyperparameters
        
        Returns:
            Path to the created config file
        """
        # Deep copy base config
        import copy
        trial_config = copy.deepcopy(self.base_config)
        
        # Update config with sampled parameters
        config_section = trial_config['config']
        
        # Update architecture parameters
        if 'modes' in sampled_params:
            config_section['modes'] = sampled_params['modes']
        if 'width' in sampled_params:
            config_section['width'] = sampled_params['width']
        if 'n_layers' in sampled_params:
            config_section['n_layers'] = sampled_params['n_layers']
        
        # Update training parameters
        config_section['learning_rate'] = sampled_params['learning_rate']
        config_section['batch_size'] = sampled_params['batch_size']
        config_section['weight_decay'] = sampled_params['weight_decay']
        config_section['step_size'] = sampled_params['step_size']
        config_section['gamma'] = sampled_params['gamma']
        
        # Fix physics_importance and boundary_condition_weight at 0
        config_section['physics_importance'] = 0.0
        config_section['boundary_condition_weight'] = 0.0
        
        # Set epochs for this trial
        config_section['epochs'] = self.epochs_per_trial
        
        # Save to file
        config_path = self.configs_dir / f"trial_{trial_id:03d}_config.toml"
        
        # Write TOML
        with open(config_path, 'w') as f:
            f.write(f'title = "{trial_config.get("title", "Trial Config")}"\n\n')
            f.write('[config]\n')
            for key, value in config_section.items():
                if isinstance(value, str):
                    f.write(f'{key} = "{value}"\n')
                else:
                    f.write(f'{key} = {value}\n')
        
        return config_path
    
    def run_trial(self, trial_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single training trial with the given configuration.
        
        Args:
            trial_id: Trial number
            config: Sampled hyperparameters
        
        Returns:
            Dictionary containing trial results
        """
        print(f"\n{'='*80}")
        print(f"Trial {trial_id + 1}/{self.n_trials}")
        print(f"{'='*80}")
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
        
        # Create config file for this trial
        config_path = self.create_trial_config(trial_id, config)
        model_path = self.models_dir / f"trial_{trial_id:03d}_model.pth"
        
        # Build training command
        train_cmd = [
            sys.executable, '-m', 'simple.train',
            f'--problem={self.problem}',
            f'--model_config={config_path}',
            f'--data_config={self.data_config_path}',
            f'--output_model={model_path}'
        ]
        
        # Run training
        start_time = time.time()
        
        try:
            result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per trial
            )
            
            training_time = time.time() - start_time
            success = (result.returncode == 0)
            
            if not success:
                print(f"Training failed for trial {trial_id}")
                print(f"Error output:\n{result.stderr}")
                return {
                    'trial_id': trial_id,
                    'config': config,
                    'success': False,
                    'error': result.stderr,
                    'training_time': training_time
                }
            
        except subprocess.TimeoutExpired:
            print(f"Training timed out for trial {trial_id}")
            return {
                'trial_id': trial_id,
                'config': config,
                'success': False,
                'error': 'Timeout',
                'training_time': 3600
            }
        except Exception as e:
            print(f"Error running trial {trial_id}: {e}")
            return {
                'trial_id': trial_id,
                'config': config,
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
        
        # Parse results from training log (CSV file)
        log_path = model_path.with_suffix('.csv')
        
        try:
            df = pd.read_csv(log_path)
            
            # Get best validation metrics
            best_epoch = df['val_rel_l2'].idxmin()
            best_val_l2 = df.loc[best_epoch, 'val_rel_l2']
            best_val_mse = df.loc[best_epoch, 'val_mse']
            final_train_loss = df.iloc[-1]['total']
            
            # Get final metrics
            final_val_l2 = df.iloc[-1]['val_rel_l2']
            final_val_mse = df.iloc[-1]['val_mse']
            
            print(f"\n✓ Trial {trial_id} completed successfully!")
            print(f"  Best Val L2: {best_val_l2:.6f} (epoch {best_epoch})")
            print(f"  Final Val L2: {final_val_l2:.6f}")
            print(f"  Training time: {training_time:.1f}s")
            
            return {
                'trial_id': trial_id,
                'config': config,
                'success': True,
                'best_val_l2': float(best_val_l2),
                'best_val_mse': float(best_val_mse),
                'best_epoch': int(best_epoch),
                'final_val_l2': float(final_val_l2),
                'final_val_mse': float(final_val_mse),
                'final_train_loss': float(final_train_loss),
                'training_time': training_time,
                'config_path': str(config_path),
                'model_path': str(model_path)
            }
            
        except Exception as e:
            print(f"Error parsing results for trial {trial_id}: {e}")
            return {
                'trial_id': trial_id,
                'config': config,
                'success': False,
                'error': f"Error parsing results: {e}",
                'training_time': training_time
            }
    
    def run_search(self):
        """Run the complete hyperparameter search."""
        print(f"\n{'='*80}")
        print(f"Starting Hyperparameter Search")
        print(f"{'='*80}\n")
        
        for trial_id in range(self.n_trials):
            # Sample configuration
            config = self.sample_config()
            
            # Run trial
            result = self.run_trial(trial_id, config)
            
            # Store result
            self.results.append(result)
            
            # Save intermediate results after each trial
            self.save_results()
        
        # Final summary
        self.print_summary()
        self.save_best_config()
    
    def save_results(self):
        """Save all search results to JSON and CSV."""
        # Save detailed JSON
        results_json = self.output_dir / "search_results.json"
        with open(results_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for r in self.results:
            row = {
                'trial_id': r['trial_id'],
                'success': r['success'],
                'training_time': r.get('training_time', None),
            }
            
            # Add config parameters
            if 'config' in r:
                row.update(r['config'])
            
            # Add metrics
            if r['success']:
                row['best_val_l2'] = r.get('best_val_l2', None)
                row['best_val_mse'] = r.get('best_val_mse', None)
                row['best_epoch'] = r.get('best_epoch', None)
                row['final_val_l2'] = r.get('final_val_l2', None)
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        summary_csv = self.output_dir / "search_summary.csv"
        df.to_csv(summary_csv, index=False)
    
    def print_summary(self):
        """Print summary of search results."""
        successful_trials = [r for r in self.results if r.get('success', False)]
        
        if not successful_trials:
            print("\nNo successful trials!")
            return
        
        # Find best trial
        best_trial = min(successful_trials, key=lambda x: x['best_val_l2'])
        
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER SEARCH SUMMARY")
        print(f"{'='*80}")
        print(f"Total Trials: {len(self.results)}")
        print(f"Successful Trials: {len(successful_trials)}")
        print(f"Failed Trials: {len(self.results) - len(successful_trials)}")
        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATION (Trial {best_trial['trial_id']})")
        print(f"{'='*80}")
        print(f"Best Validation L2 Error: {best_trial['best_val_l2']:.6f}")
        print(f"Best Epoch: {best_trial['best_epoch']}")
        print(f"Training Time: {best_trial['training_time']:.1f}s")
        print(f"\nHyperparameters:")
        for key, value in best_trial['config'].items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
    
    def save_best_config(self):
        """Save the best configuration to a separate file."""
        successful_trials = [r for r in self.results if r.get('success', False)]
        
        if not successful_trials:
            print("No successful trials to save best config.")
            return
        
        best_trial = min(successful_trials, key=lambda x: x['best_val_l2'])
        
        # Copy the best config
        best_config_source = Path(best_trial['config_path'])
        best_config_dest = self.output_dir / "best_config.toml"
        shutil.copy(best_config_source, best_config_dest)
        
        # Copy the best model
        best_model_source = Path(best_trial['model_path'])
        best_model_dest = self.output_dir / "best_model.pth"
        shutil.copy(best_model_source, best_model_dest)
        
        # Save best config info
        best_info = {
            'trial_id': best_trial['trial_id'],
            'best_val_l2': best_trial['best_val_l2'],
            'config': best_trial['config'],
            'config_path': str(best_config_dest),
            'model_path': str(best_model_dest)
        }
        
        with open(self.output_dir / "best_config_info.json", 'w') as f:
            json.dump(best_info, f, indent=2)
        
        print(f"\n✓ Best configuration saved to:")
        print(f"   Config: {best_config_dest}")
        print(f"   Model: {best_model_dest}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for neural operator models"
    )
    
    parser.add_argument('--problem', type=str, required=True,
                       help='Problem name (e.g., simple_darcy_2d)')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['fno', 'fno2d', 'ffno', 'ffno2d', 'cno', 'cno2d'],
                       help='Model type')
    parser.add_argument('--base_model_config', type=str, required=True,
                       help='Path to base model config TOML')
    parser.add_argument('--data_config', type=str, required=True,
                       help='Path to data config TOML')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save search results')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials to run (default: 50)')
    parser.add_argument('--epochs_per_trial', type=int, default=50,
                       help='Number of epochs per trial (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.base_model_config).exists():
        print(f"Error: Base model config not found: {args.base_model_config}")
        sys.exit(1)
    
    if not Path(args.data_config).exists():
        print(f"Error: Data config not found: {args.data_config}")
        sys.exit(1)
    
    # Create and run searcher
    searcher = HyperparameterSearcher(
        problem=args.problem,
        model_type=args.model_type,
        base_config_path=args.base_model_config,
        data_config_path=args.data_config,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        device=args.device
    )
    
    searcher.run_search()
    
    print("\nHyperparameter search complete!")


if __name__ == "__main__":
    main()
