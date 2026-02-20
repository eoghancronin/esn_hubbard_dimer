import optuna
import numpy as np

def load_optimal_hyperparameters(study_name, db_file=None):
    """
    Load optimal hyperparameters from an Optuna study database.
    
    Returns:
        dict: Dictionary containing the optimal hyperparameters
    """
    if db_file is None:
        db_file = f"{study_name}.db"
    
    storage = f"sqlite:///{db_file}"
    study = optuna.load_study(
        study_name=study_name,
        storage=storage
    )
    
    optimal_params = {
        'sr': study.best_params['sr'],
        'units': study.best_params['units'],
        'lr/dt': 5 * study.best_params['lr'],
        'ridge': study.best_params['ridge'],
    }
    
    return optimal_params

def format_sig_figs(val, sig_figs=2):
    """Format a number to specified significant figures."""
    if val == 0:
        return "0"
    return f"{val:.{sig_figs}g}"

# Load all parameters
params = {
    '3a': {
        'direct': load_optimal_hyperparameters('test1_direct_study', db_file='sec3a/test1_direct_study.db'),
        'augmented': load_optimal_hyperparameters('test1_augmented_study', db_file='sec3a/test1_augmented_study.db')
    },
    '3b': {
        'direct': load_optimal_hyperparameters('test2_direct_study', db_file='sec3b/test2_direct_study.db'),
        'augmented': load_optimal_hyperparameters('test2_augmented_study', db_file='sec3b/test2_augmented_study.db')
    },
    '3c': {
        'direct': load_optimal_hyperparameters('test3_direct_study', db_file='sec3c/test3_direct_study.db'),
        'augmented': load_optimal_hyperparameters('test3_augmented_study', db_file='sec3c/test3_augmented_study.db')
    },
    '3d': {
        'direct': load_optimal_hyperparameters('test4_direct_1200', db_file='sec3d/test4_direct_1200.db'),
        'augmented': None  # No augmented for 3d
    }
}

print("TABLE III. Optimized hyperparameters for each investigation.")
print()

# Print table
header = f"{'Section':<8} | {'SR (D)':<8} {'Units (D)':<10} {'LR/DT (D)':<10} {'Ridge (D)':<10} | {'SR (A)':<8} {'Units (A)':<10} {'LR/DT (A)':<10} {'Ridge (A)':<10}"
print(header)
print("-" * len(header))

for sec in ['3a', '3b', '3c', '3d']:
    d = params[sec]['direct']
    a = params[sec]['augmented']
    
    row = f"{sec:<8} | "
    row += f"{format_sig_figs(d['sr']):<8} {d['units']:<10} {format_sig_figs(d['lr/dt']):<10} {format_sig_figs(d['ridge']):<10} | "
    
    if a:
        row += f"{format_sig_figs(a['sr']):<8} {a['units']:<10} {format_sig_figs(a['lr/dt']):<10} {format_sig_figs(a['ridge']):<10}"
    else:
        row += f"{'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10}"
    
    print(row)
    
import json

def format_sig_figs(val, sig_figs=2):
    """Format a number to specified significant figures."""
    if val == 0:
        return "0"
    return f"{val:.{sig_figs}g}"

def load_rmse_results(filepath):
    """Load RMSE results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return {
        'train_max': results['train']['max_rmse'],
        'train_avg': results['train']['avg_rmse'],
        'test_max': results['test']['max_rmse'],
        'test_avg': results['test']['avg_rmse']
    }

# Load all results
results = {
    '3a': {
        'direct': load_rmse_results('sec3a/test1_direct_results.json'),
        'augmented': load_rmse_results('sec3a/test1_augmented_results.json')
    },
    '3b': {
        'direct': load_rmse_results('sec3b/test2_direct_results.json'),
        'augmented': load_rmse_results('sec3b/test2_augmented_results.json')
    },
    '3c': {
        'direct': load_rmse_results('sec3c/test3_direct_results.json'),
        'augmented': load_rmse_results('sec3c/test3_augmented_results.json')
    },
    '3d': {
        'direct': load_rmse_results('sec3d/test4_direct_1200.json'),
        'augmented': None
    }
}

# Print table header
print("TABLE II. Summary of RMSEs of each investigation.")
print()
header = (f"{'Section':<12} | "
          f"{'Max RMSE (D)':<12} {'Max RMSE (D)':<12} {'Avg RMSE (D)':<12} {'Avg RMSE (D)':<12} | "
          f"{'Max RMSE (A)':<12} {'Max RMSE (A)':<12} {'Avg RMSE (A)':<12} {'Avg RMSE (A)':<12}")
subheader = (f"{'Investigation':<12} | "
             f"{'Train':<12} {'Test':<12} {'Train':<12} {'Test':<12} | "
             f"{'Train':<12} {'Test':<12} {'Train':<12} {'Test':<12}")

# Simplified header
print(f"{'Investigation':<12} | {'n^ESN(t)':<50} | {'n^ESN_ω(t)':<50}")
print(f"{'':<12} | {'Max RMSE':<24} {'Avg RMSE':<24} | {'Max RMSE':<24} {'Avg RMSE':<24}")
print(f"{'':<12} | {'Train':<12}{'Test':<12}{'Train':<12}{'Test':<12} | {'Train':<12}{'Test':<12}{'Train':<12}{'Test':<12}")
print("-" * 130)

for sec in ['3a', '3b', '3c', '3d']:
    d = results[sec]['direct']
    a = results[sec]['augmented']
    
    # Apply factor of 2 for sections 3a, 3b, 3c (not 3d)
    factor = 2 if sec != '3d' else 1
    
    row = f"Sec.III {sec.upper():<4} | "
    row += f"{format_sig_figs(d['train_max']*factor):<12}"
    row += f"{format_sig_figs(d['test_max']*factor):<12}"
    row += f"{format_sig_figs(d['train_avg']*factor):<12}"
    row += f"{format_sig_figs(d['test_avg']*factor):<12} | "
    
    if a:
        row += f"{format_sig_figs(a['train_max']*factor):<12}"
        row += f"{format_sig_figs(a['test_max']*factor):<12}"
        row += f"{format_sig_figs(a['train_avg']*factor):<12}"
        row += f"{format_sig_figs(a['test_avg']*factor):<12}"
    else:
        row += f"{'-':<12}{'-':<12}{'-':<12}{'-':<12}"
    
    print(row)
