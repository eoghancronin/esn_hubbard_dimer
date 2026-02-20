import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.observables import rmse, rsquare
from reservoirpy.nodes import Reservoir, Ridge, ESN
import time
import io
from contextlib import redirect_stdout, redirect_stderr
import optuna
import json


def load_data(filename):
    """Load data from JSON file and prepare arrays."""
    with open(filename, mode='r') as f:
        data = json.load(f)
    
    L = data['L']
    num_sys = data['num_sys']
    tmax = data['tmax']
    dt = data['dt']
    steps = int(tmax/dt)
    t_array = np.linspace(0, tmax, steps)
    v_array = np.array(data['v_array'])[:,:,1,None]  # Shape: (num_sys, steps)
    n_array = np.array(data['n_array'])         # Shape: (num_sys, steps, 4)
    # Prepare input data - stack the 4 input arrays
    X = v_array
    Y = n_array  # Shape: (num_sys, steps, 4)
    
    return {
        'X': X,
        'Y': Y,
        'L': L,
        'num_sys': num_sys,
        'tmax': tmax,
        'dt': dt,
        'steps': steps,
        't_array': t_array
    }

def evaluate_esn(esn, X, Y, t_on, dt, warmup=None, plot=False, plot_title=""):
    """
    Evaluate ESN on given data and return metrics and predictions.
    
    Returns:
    --------
    dict with keys:
        - max_rmse: Maximum RMSE across all systems
        - avg_rmse: Average RMSE across all systems
        - avg_r2: Average R2 across all systems
        - system_rmses: Array of RMSE for each system
        - system_r2s: Array of R2 for each system
        - y_pred: Predictions array of shape (num_sys, steps, L)
    """
    num_sys = X.shape[0]
    L = Y.shape[2]
    steps = X.shape[1]
    
    if warmup is None:
        warmup = int(t_on[0]/(2*dt))
    
    # Convert to list of sequences for parallel processing
    X_list = [X[i] for i in range(num_sys)]
    
    # Run all systems in parallel (uses workers=-1 from ESN construction)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        predictions = esn.run(X_list, reset=True)
    
    # Convert predictions list to array
    predictions_array = np.array(predictions)
    
    # Calculate metrics for all systems
    system_rmses = np.array([rmse(Y[i, warmup:], predictions[i][warmup:]) for i in range(num_sys)])
    system_r2s = np.array([rsquare(Y[i, warmup:], predictions[i][warmup:]) for i in range(num_sys)])
    
    if plot:
        t_array = np.linspace(0, steps*dt, steps)
        for system in range(num_sys):
            fig, axes = plt.subplots(L, 1, figsize=(12, 2*L), sharex=True)
            
            for site in range(L):
                axes[site].plot(t_array[warmup:], Y[system,warmup:,site], linewidth=1.5, 
                               label=f'ρ_{site}(t) - True')
                axes[site].plot(t_array[warmup:], predictions[system][warmup:,site], linewidth=1.5, 
                               label=f'ρ_{site}(t) - Predicted')
                axes[site].set_ylabel(f'ρ_{site}(t)')
                axes[site].legend(loc='upper right')
                axes[site].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Time')
            plt.suptitle(f'{plot_title} - System {system} - R2: {system_r2s[system]:.4f}, RMSE: {system_rmses[system]:.4f}')
            plt.tight_layout()
            plt.show()
    
    return {
        'max_rmse': np.max(system_rmses),
        'avg_rmse': np.mean(system_rmses),
        'avg_r2': np.mean(system_r2s),
        'system_rmses': system_rmses,
        'system_r2s': system_r2s,
        'y_pred': predictions_array
    }

def train_esn(X, Y, params, t_on, dt, verbose=0):
    """
    Train ESN with given parameters.
    
    Returns:
    --------
    trained_esn : ESN
        Trained ESN model
    """
    warmup = int(t_on[0]/(2*dt))
    
    # Create ESN with parameters
    reservoir = Reservoir(
        units=params['units'], 
        lr=params['lr'], 
        sr=params['sr'],
        input_scaling=params['input_scaling'],
        input_connectivity=params['input_connectivity'],
        rc_connectivity=params['rc_connectivity'],
        input_bias=True,
        bias_scaling = params['bias_scaling'],
        seed=0
    )
    readout = Ridge(output_dim=Y.shape[2], ridge=params['ridge'])
    esn = ESN(reservoir=reservoir, readout=readout, workers=-1)
    
    if verbose > 0:
        print("Training ESN with parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
    
    start = time.time()
    esn = esn.fit(X, Y, warmup=warmup, reset=True)
    if verbose > 1:
        print(f"Training time: {time.time() - start:.2f} seconds")
    return esn

def objective(trial, X_train, Y_train, t_on, dt):
    """
    Objective function for Optuna to optimize the ESN hyperparameters.
    """
    units = trial.suggest_int('units', 100, 1000)
    params = {
        'units': units,
        'sr': trial.suggest_float('sr', 1E-2, 2.0, log=True),
        'lr': trial.suggest_float('lr', 1E-5, 1.0, log=True),
        'ridge': trial.suggest_float('ridge', 1e-8, 1e-3, log=True),
        'input_scaling': trial.suggest_float('input_scaling', 0.01, 1.0, log=True),
        'input_connectivity': trial.suggest_float('input_connectivity', 1E-3, 1.0, log=False),
        'rc_connectivity': trial.suggest_float('rc_connectivity', 1E-3, 0.3, log=False),
        'bias_scaling': trial.suggest_float('bias_scaling', 1E-5,1,log=True)
    }
    
    # Train ESN
    esn = train_esn(X_train, Y_train, params, t_on, dt, verbose=0)
    
    # Evaluate on training data
    results = evaluate_esn(esn, X_train, Y_train, t_on, dt)
    
    return results['avg_rmse']

def train_optimal_esn_with_test(study_name, n_trials, tmax, load_if_exists=True):
    """
    Perform hyperparameter optimization on train data and evaluate on both train and test data.
    
    Parameters:
    -----------
    study_name : str
        Name of the Optuna study
    n_trials : int
        Number of optimization trials (0 to just load existing study)
    load_if_exists : bool
        Whether to load existing study if it exists
    
    Returns:
    --------
    dict with 'train' and 'test' keys, each containing:
        - max_rmse, avg_rmse, avg_r2
        - system_rmses, system_r2s
        - y_pred (predictions array)
    """
    
    # Load training data
    print("Loading training data...")
    train_data = load_data(f'test4_train_data_{tmax}.json')
    X_train = train_data['X']
    Y_train = train_data['Y']
    
    # Load test data
    print("Loading test data...")
    test_data = load_data(f'test4_test_data_{tmax}.json')
    X_test = test_data['X']
    Y_test = test_data['Y']
    
    print(f'Train X shape: {X_train.shape}, Train Y shape: {Y_train.shape}')
    print(f'Test X shape: {X_test.shape}, Test Y shape: {Y_test.shape}')
    
    # Set t_on (adjust if needed)
    t_on_train = 150 * np.ones(train_data['num_sys'])
    t_on_test = 150 * np.ones(test_data['num_sys'])
    
    # Create or load Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        load_if_exists=load_if_exists,
        storage=f"sqlite:///{study_name}.db"
    )
    
    # Optimize hyperparameters if n_trials > 0
    if n_trials > 0:
        print(f"\nStarting hyperparameter optimization with {n_trials} trials...")
        study.optimize(
            lambda trial: objective(trial, X_train, Y_train, t_on_train, train_data['dt']), 
            n_trials=n_trials
        )
        print("\nOptimization completed!")
    else:
        print("\nLoading existing study (no new trials)...")
    
    # Print best results
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for param_name, param_value in study.best_params.items():
        print(f"  {param_name}: {param_value}")
    
    # Visualizations
    try:
        if n_trials > 0:  # Only show if we ran new trials
            fig = optuna.visualization.plot_optimization_history(study)
            fig.show()
            
            fig = optuna.visualization.plot_param_importances(study)
            fig.show()
    except (ImportError, RuntimeError) as e:
        print(f"Visualization error: {e}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_esn = train_esn(X_train, Y_train, study.best_params, t_on_train, train_data['dt'], verbose=1)
    
    # Evaluate on training data
    print("\nEvaluating on training data...")
    train_results = evaluate_esn(
        final_esn, X_train, Y_train, t_on_train, train_data['dt'], 
        plot=True, plot_title="Training Data"
    )
    print(f"Training - Max RMSE: {train_results['max_rmse']:.4f}, "
          f"Avg RMSE: {train_results['avg_rmse']:.4f}, "
          f"Avg R2: {train_results['avg_r2']:.4f}")
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_results = evaluate_esn(
        final_esn, X_test, Y_test, t_on_test, test_data['dt'], 
        plot=True, plot_title="Test Data"
    )
    print(f"Test - Max RMSE: {test_results['max_rmse']:.4f}, "
          f"Avg RMSE: {test_results['avg_rmse']:.4f}, "
          f"Avg R2: {test_results['avg_r2']:.4f}")
    
    return {
        'train': train_results,
        'test': test_results
    }

# Main execution
if __name__ == "__main__":
    tmax=1200
    results = train_optimal_esn_with_test(
        study_name=f'test4_direct_{tmax}', 
        n_trials=0,  # Set to 0 to load existing study, change to higher number for optimization
        tmax=tmax,
        load_if_exists=True
    )
    
    print(f"\nFinal training predictions array shape: {results['train']['y_pred'].shape}")
    print(f"Final testing predictions array shape: {results['test']['y_pred'].shape}")
    
    # Prepare results dictionary with both training and testing metrics
    final_results = {
        'train': {
            'max_rmse': results['train']['max_rmse'],
            'avg_rmse': results['train']['avg_rmse'],
            'avg_r2': results['train']['avg_r2'],
            'system_rmses': results['train']['system_rmses'].tolist(),
            'system_r2s': results['train']['system_r2s'].tolist(),
            'y_pred_all': results['train']['y_pred'].tolist()
        },
        'test': {
            'max_rmse': results['test']['max_rmse'],
            'avg_rmse': results['test']['avg_rmse'], 
            'avg_r2': results['test']['avg_r2'],
            'system_rmses': results['test']['system_rmses'].tolist(),
            'system_r2s': results['test']['system_r2s'].tolist(),
            'y_pred_all': results['test']['y_pred'].tolist()
        }
    }
    
    # Save results to JSON file
    with open(f'test4_direct_{tmax}.json', mode='w') as f:
        json.dump(final_results, f)
    
    print("\nResults saved to 'test4_single_results_bias.json'")
