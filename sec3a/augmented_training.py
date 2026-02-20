import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.observables import rmse, rsquare
from reservoirpy.nodes import Reservoir, Ridge, ESN
import time
import io
from contextlib import redirect_stdout, redirect_stderr
import optuna
import json



def load_data(filename, test_index):
    """Load data from JSON file and prepare arrays."""
    # For compatibility with script #2's data format
    from ESN_data_gen_v1_3_3 import json_file_to_arrays
    
    # Load the data using the original function
    X, Y, Z, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, omega01_t, omega02_t, omega12_t = json_file_to_arrays(filename, observables=True)
    
    # Calculate omega values for the sin signals
    def H_(dv,t):
        h = np.zeros((3,3))
        h[0,0] = 2*(dv/2) + 1
        h[0,1] = -np.sqrt(2)*t
        h[1,0] = -np.sqrt(2)*t
        h[1,2] = -np.sqrt(2)*t
        h[2,2] = -2*(dv/2) + 1
        h[2,1] = -np.sqrt(2)*t
        return h
    
    num_sys = len(X)
    steps = len(X[0])
    
    if test_index == 3: # Modified approach to frequency augmentation for near resonant dynamics
        vals = np.linalg.eigvalsh(H_(0.04, 0.05))  # Using T=0.05 as in script #2
        omega01 = vals[1] - vals[0]  # Renamed from omega1
        omega12 = vals[2] - vals[1]  # Renamed from omega3 (omega2 removed)
        
        dt = 0.2
        tmax = 650
        t_array = np.linspace(0, tmax-dt, steps)
        
        # Create omega signals - now only omega01 and omega12
        omega01_t = np.array([np.sin(omega01*t_array) for i in range(num_sys)])
        omega12_t = np.array([np.sin(omega12*t_array) for i in range(num_sys)])
    
    # Prepare input data - stack the 3 input arrays (v_ext, omega01, omega12)
    v_array = np.array(X)[:, :, np.newaxis]  # Shape: (num_sys, steps, 1)
    omega01_t = omega01_t[:, :, np.newaxis]
    omega12_t = omega12_t[:, :, np.newaxis]
    
    X_multi = np.concatenate([v_array, omega01_t, omega12_t], axis=2)  # Shape: (num_sys, steps, 3)
    Y_output = np.array(Y)[:, :, np.newaxis]  # Shape: (num_sys, steps, 1) for single output
    
    return {
        'X': X_multi,
        'Y': Y_output,
        'L': 1,  # Output dimension is 1 for script #2
        'num_sys': num_sys,
        'steps': steps,
    }

def create_custom_input_weights(units, input_dims, input_scalings, input_connectivities, rng=None):
    """
    Create custom input weight matrix with different connectivity for each input channel.
    
    Args:
        units: Number of reservoir units
        input_dims: List of dimensions for each input channel
        input_scalings: Array of scaling factors for each channel
        input_connectivities: Array of connectivity probabilities for each channel
        rng: Random number generator (numpy RandomState) for reproducibility
    
    Returns:
        Win: Custom input weight matrix
    """
    if rng is None:
        rng = np.random.RandomState()
    
    total_input_dim = sum(input_dims)
    Win = np.zeros((units, total_input_dim))
    
    input_start = 0
    for i, (dim, scaling, connectivity) in enumerate(zip(input_dims, input_scalings, input_connectivities)):
        input_end = input_start + dim
        
        # Generate weights for this input channel using the provided RNG
        channel_weights = rng.randn(units, dim) * scaling
        
        # Apply connectivity mask using the provided RNG
        connectivity_mask = rng.rand(units, dim) < connectivity
        channel_weights = channel_weights * connectivity_mask
        
        # Place in the full weight matrix
        Win[:, input_start:input_end] = channel_weights
        input_start = input_end
    
    return Win

def evaluate_esn(esn, X, Y, warm_up, plot=False, plot_title=""):
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
    L = Y.shape[2]  # Will be 1 for this script
    steps = X.shape[1]
    dt = 0.2
    
    # Convert X to list of sequences for parallel processing
    X_list = [X[i] for i in range(num_sys)]
    
    # Run all systems in parallel at once
    esn.reset()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        predictions = esn.run(X_list, reset=True)
    
    # Convert predictions list to array
    predictions_array = np.array(predictions)
    
    # Calculate metrics for each system
    r2s = []
    rmses = []
    
    for system in range(num_sys):
        y_pred = predictions_array[system]
        
        system_r2 = rsquare(Y[system, warm_up:], y_pred[warm_up:])
        system_rmse = rmse(Y[system, warm_up:], y_pred[warm_up:])
        
        r2s.append(system_r2)
        rmses.append(system_rmse)
        
        if plot:  # Only plot first system to avoid clutter
            t_array = np.linspace(0, steps*dt, steps)
            
            if L == 1:  # Single output case
                fig, axes = plt.subplots(2, 1, figsize=(15, 10))
                
                # Top subplot: Density comparison
                axes[0].plot(t_array[warm_up:], Y[system, warm_up:, 0], linewidth=2, label='True')
                axes[0].plot(t_array[warm_up:], y_pred[warm_up:, 0], linewidth=2, alpha=0.8, label='Predicted')
                axes[0].set_xlabel('Time')
                axes[0].set_ylabel('Density')
                axes[0].set_title(f'{plot_title} - System {system} - R²={system_r2:.4f}, RMSE={system_rmse:.4f}')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Bottom subplot: Input signals
                channel_names = ['v_ext', 'sin(ω₀₁t)', 'sin(ω₁₂t)']
                colors = ['blue', 'red', 'green']
                
                for channel in range(X.shape[2]):
                    axes[1].plot(t_array[warm_up:], X[system, warm_up:, channel], 
                                linewidth=1.5, color=colors[channel], label=channel_names[channel])
                
                axes[1].set_xlabel('Time')
                axes[1].set_ylabel('Input Signal Amplitude')
                axes[1].set_title(f'Input Signals for System {system}')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            else:  # Multiple outputs case
                fig, axes = plt.subplots(L, 1, figsize=(12, 2*L), sharex=True)
                
                for site in range(L):
                    axes[site].plot(t_array[warm_up:], Y[system,warm_up:,site], linewidth=1.5, 
                                   label=f'ρ_{site}(t) - True')
                    axes[site].plot(t_array[warm_up:], y_pred[warm_up:,site], linewidth=1.5, 
                                   label=f'ρ_{site}(t) - Predicted')
                    axes[site].set_ylabel(f'ρ_{site}(t)')
                    axes[site].legend(loc='upper right')
                    axes[site].grid(True, alpha=0.3)
                
                axes[-1].set_xlabel('Time')
                plt.suptitle(f'{plot_title} - System {system} - R2: {system_r2:.4f}, RMSE: {system_rmse:.4f}')
                plt.tight_layout()
                plt.show()
    
    # Convert to arrays
    system_rmses = np.array(rmses)
    system_r2s = np.array(r2s)
    
    return {
        'max_rmse': np.max(system_rmses),
        'avg_rmse': np.mean(system_rmses),
        'avg_r2': np.mean(system_r2s),
        'system_rmses': system_rmses,
        'system_r2s': system_r2s,
        'y_pred': predictions_array
    }

def train_esn(X, Y, params, warm_up, verbose=0):
    """
    Train ESN with given parameters.
    
    Returns:
    --------
    trained_esn : ESN
        Trained ESN model
    """
    
    seed = 0  # Always use seed=0 as requested
    rng = np.random.RandomState(seed)
    
    # Create custom input weight matrix with different connectivity for each channel
    input_dims = [1, 1, 1]  # Each channel is 1-dimensional (3 channels now)
    Win = create_custom_input_weights(
        units=params['units'],  # Now using optimized units
        input_dims=input_dims,
        input_scalings=params['input_scaling'],
        input_connectivities=params['input_connectivity'],
        rng=rng  # Pass the RNG for reproducibility
    )
    
    # Create ESN with parameters
    reservoir = Reservoir(
        units=params['units'],  # Using optimized units
        lr=params['lr'], 
        sr=params['sr'],
        Win=Win,
        rc_connectivity=params['rc_connectivity'],
        input_bias=True,
        bias_scaling=params['bias_scaling'],
        seed=seed  # Always use seed=0
    )
    readout = Ridge(output_dim=Y.shape[2], ridge=params['ridge'])
    esn = ESN(reservoir=reservoir, readout=readout, workers=-1)
    
    if verbose > 0:
        print("Training ESN with parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
    
    start = time.time()
    esn = esn.fit(X, Y, warmup=warm_up, reset=True)
    if verbose > 1:
        print(f"Training time: {time.time() - start:.2f} seconds")
    
    return esn

def objective(trial, X_train, Y_train, warm_up):
    """
    Objective function for Optuna to optimize the ESN hyperparameters.
    """
    # Optimize units as in script #2
    units = trial.suggest_int('units', 100, 1000)
    
    params = {
        'units': units,  # Optimizing units
        'sr': trial.suggest_float('sr', 1E-2, 2.0, log=True),
        'lr': trial.suggest_float('lr', 1E-5, 1.0, log=True),
        'ridge': trial.suggest_float('ridge', 1e-8, 1e-3, log=True),
        'rc_connectivity': trial.suggest_float('rc_connectivity', 1E-3, 1.0, log=False),
        'bias_scaling': trial.suggest_float('bias_scaling', 1E-5, 1.0, log=True)
    }
    
    input_scaling_list = []
    input_connectivity_list = []
    # Now only 3 channels: v_ext, sin_omega01, sin_omega12
    channel_names = ['v_ext', 'sin_omega01', 'sin_omega12']
    
    for i, channel_name in enumerate(channel_names):
        scaling = trial.suggest_float(f'input_scaling_{channel_name}', 1E-5, 1E2, log=True)
        connectivity = trial.suggest_float(f'input_connectivity_{channel_name}', 2E-3, 1.0, log=False)
        input_scaling_list.append(scaling)
        input_connectivity_list.append(connectivity)
    
    params['input_scaling'] = np.array(input_scaling_list)
    params['input_connectivity'] = np.array(input_connectivity_list)
    
    # Train ESN
    esn = train_esn(X_train, Y_train, params, warm_up, verbose=0)
    
    # Evaluate on training data
    results = evaluate_esn(esn, X_train, Y_train, warm_up)
    
    # Return max_rmse as in script #2 (minimizing max RMSE)
    return results['max_rmse']

def train_optimal_esn_with_test(study_name, test_index, n_trials=0, load_if_exists=True):
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
    file_name_train = 'test'+f'{test_index}'+'_train_data.json'
    file_name_test = 'test'+f'{test_index}'+'_test_data.json'
    
    # Load training data
    print("Loading training data...")
    train_data = load_data(file_name_train, test_index)
    X_train = train_data['X']
    Y_train = train_data['Y']
    
    # Load test data
    print("Loading test data...")
    test_data = load_data(file_name_test, test_index)
    X_test = test_data['X']
    Y_test = test_data['Y']
    
    print(f'Train X shape: {X_train.shape}, Train Y shape: {Y_train.shape}')
    print(f'Test X shape: {X_test.shape}, Test Y shape: {Y_test.shape}')
    
    warm_ups = [int(75/dt), int(500/dt), int(75/dt)]
    warm_up = warm_ups[test_index-1]
    
    # Create or load Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # Minimize max RMSE as in script #2
        load_if_exists=load_if_exists,
        storage=f"sqlite:///{study_name}.db"
    )
    
    # Optimize hyperparameters if n_trials > 0
    if n_trials > 0:
        print(f"\nStarting hyperparameter optimization with {n_trials} trials...")
        print("Optimizing for minimum MAXIMUM RMSE across all training systems...")
        study.optimize(
            lambda trial: objective(trial, X_train, Y_train, warm_up), 
            n_trials=n_trials
        )
        print("\nOptimization completed!")
    else:
        print("\nLoading existing study (no new trials)...")
    
    # Print best results
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best max RMSE: {study.best_value:.6f}")
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
    
    input_scaling_list = []
    input_connectivity_list = []
    channel_names = ['v_ext', 'sin_omega01', 'sin_omega12']
    
    for channel_name in channel_names:
        input_scaling_list.append(study.best_params[f'input_scaling_{channel_name}'])
        input_connectivity_list.append(study.best_params[f'input_connectivity_{channel_name}'])
    
    best_params = {
        'units': study.best_params['units'],  # Using optimized units
        'sr': study.best_params['sr'],
        'lr': study.best_params['lr'],
        'ridge': study.best_params['ridge'],
        'input_scaling': np.array(input_scaling_list),
        'input_connectivity': np.array(input_connectivity_list),
        'rc_connectivity': study.best_params['rc_connectivity'],
        'bias_scaling': study.best_params['bias_scaling']
    }
    
    final_esn = train_esn(X_train, Y_train, best_params, warm_up, verbose=1)
    
    # Evaluate on training data
    print("\nEvaluating on training data...")
    train_results = evaluate_esn(
        final_esn, X_train, Y_train, warm_up, 
        plot=True, plot_title="Training Data"
    )
    print(f"Training - Max RMSE: {train_results['max_rmse']:.4f}, "
          f"Avg RMSE: {train_results['avg_rmse']:.4f}, "
          f"Avg R2: {train_results['avg_r2']:.4f}")
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_results = evaluate_esn(
        final_esn, X_test, Y_test, warm_up, 
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
    dt=0.2
    test_index = 1
    results = train_optimal_esn_with_test(
        study_name='test'+f'{test_index}'+'_augmented_study',
        test_index=test_index,
        n_trials=0,  # Set to 0 to load existing study, change to higher number for optimization
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
    with open('test'+f'{test_index}'+'_augmented_results.json', mode='w') as f:
        json.dump(final_results, f)