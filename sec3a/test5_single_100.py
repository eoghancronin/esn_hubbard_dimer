import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.observables import rmse, rsquare
from reservoirpy.nodes import Reservoir, Ridge, ESN
import time
import io
from contextlib import redirect_stdout, redirect_stderr
import optuna
from ESN_data_gen_v1_3_3 import dataset, json_file_to_arrays
import json

def H_(dv,t):
    h = np.zeros((3,3))
    h[0,0] = 2*(dv/2) + 1
    h[0,1] = -np.sqrt(2)*t
    h[1,0] = -np.sqrt(2)*t
    h[1,2] = -np.sqrt(2)*t
    h[2,2] = -2*(dv/2) + 1
    h[2,1] = -np.sqrt(2)*t
    return h

def v_sig(dv0,dv1,gamma,t_on):
    def v(t):
        return dv0 + dv1*(1/(1+np.exp(-gamma*(t-t_on))))
    return v

def sig(gamma, t_on):
    def sig_(t):
        return (1/(1+np.exp(-gamma*(t-t_on))))
    return sig_

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

# Generate TRAINING dataset for both hyperparameter optimization and fitting
np.random.seed(0)
num_sys_train = 201
T_train = 0.05*np.ones(num_sys_train)
delta_v0_train = -1.4 + np.linspace(0, 1, num_sys_train)
amplitude_train = -1*delta_v0_train
dt = 0.2
t_on_train = 150*np.ones(num_sys_train)
tmax = 250
gamma_off_train = 1.0 + 0.0*np.random.random(num_sys_train)
steps = int(tmax/dt)
t_array = np.linspace(0, tmax-dt, steps)

v_func_list_train = []
for i in range(num_sys_train):
    v_func_list_train.append(v_sig(delta_v0_train[i], amplitude_train[i], gamma_off_train[i], t_on_train[i]))

# Generate training dataset
file_name_train = 'test5_train_data.json'
#dataset(T_train, v_func_list_train, t_array, mode='to_file', file_name=file_name_train, observables=True)

# Generate TESTING dataset with midpoint values
np.random.seed(1)  # Different seed for test data
num_sys_test = 200
T_test = 0.05*np.ones(num_sys_test)
# Calculate midpoints between consecutive training delta_v0 values
delta_v0_test = np.zeros(num_sys_test)
for i in range(num_sys_test):
    delta_v0_test[i] = (delta_v0_train[i] + delta_v0_train[i+1])/2

amplitude_test = -1*delta_v0_test
t_on_test = 150*np.ones(num_sys_test)
gamma_off_test = 1.0 + 0.0*np.random.random(num_sys_test)

v_func_list_test = []
for i in range(num_sys_test):
    v_func_list_test.append(v_sig(delta_v0_test[i], amplitude_test[i], gamma_off_test[i], t_on_test[i]))

# Generate testing dataset
file_name_test = 'test5_test_data.json'
#dataset(T_test, v_func_list_test, t_array, mode='to_file', file_name=file_name_test, observables=True)

# Configuration: Input channels = external potential + 3 natural frequency sin signals
NUM_INPUT_CHANNELS = 4  # v_ext + sin(ω₁t) + sin(ω₂t) + sin(ω₃t)

# Load training data
X_train, Y_train, Z_train, TplusU_train, TplusUplusV_train, psi0_overlap_train, psi1_overlap_train, psi2_overlap_train, omega1_t_train, omega2_t_train, omega3_t_train = json_file_to_arrays('./test5_train_data.json', observables=True)

# Load testing data
X_test, Y_test, Z_test, TplusU_test, TplusUplusV_test, psi0_overlap_test, psi1_overlap_test, psi2_overlap_test, omega1_t_test, omega2_t_test, omega3_t_test = json_file_to_arrays('./test5_test_data.json', observables=True)

# Calculate eigenvalues for omega calculations
vals_train = np.linalg.eigvalsh(H_(0.00, T_train[0]))
omega1_train_val = vals_train[1] - vals_train[0]
omega2_train_val = vals_train[2] - vals_train[0]
omega3_train_val = vals_train[2] - vals_train[1]

vals_test = np.linalg.eigvalsh(H_(0.00, T_test[0]))
omega1_test_val = vals_test[1] - vals_test[0]
omega2_test_val = vals_test[2] - vals_test[0]
omega3_test_val = vals_test[2] - vals_test[1]

# Zero out all frequency augmentation for the study without frequency augmentation
omega1_t_train = np.zeros(omega1_t_train.shape)
omega1_t_test = np.zeros(omega1_t_test.shape)

omega2_t_train = np.zeros(omega2_t_train.shape)
omega2_t_test = np.zeros(omega2_t_test.shape)

omega3_t_train = np.zeros(omega3_t_train.shape)
omega3_t_test = np.zeros(omega3_t_test.shape)

print(f'Training dataset - X shape: {X_train[0].shape}')
print(f'Training dataset - X range: [{np.min(X_train)}, {np.max(X_train)}], Y range: [{np.min(Y_train)}, {np.max(Y_train)}]')
print(f'Training delta_v0 range: [{delta_v0_train[0]:.4f}, {delta_v0_train[-1]:.4f}]')
print(f'Testing dataset - X shape: {X_test[0].shape}')
print(f'Testing dataset - X range: [{np.min(X_test)}, {np.max(X_test)}], Y range: [{np.min(Y_test)}, {np.max(Y_test)}]')
print(f'Testing delta_v0 range: [{delta_v0_test[0]:.4f}, {delta_v0_test[-1]:.4f}]')

tsteps = len(X_train[0])
print(f'Total input channels: {NUM_INPUT_CHANNELS} (v_ext + sin(ω₁t) + sin(ω₂t) + sin(ω₃t))')

def Ready_Data(X, Y, omega1_t, omega2_t, omega3_t, num_sys):
    """
    Prepare data with external potential and three natural frequency signals for training only.
    
    Args:
        X, Y: Original potential and density data
        omega1_t, omega2_t, omega3_t: Time-dependent natural frequencies
        num_sys: Number of systems in the dataset
    
    Returns:
        X_train_multi_input, y_train
    """    
    X_data = np.vstack([X[i] for i in range(num_sys)])
    y_data = np.vstack([Y[i] for i in range(num_sys)])
    
    # Extract omega arrays for all systems
    omega1_data = np.vstack([omega1_t[i] for i in range(num_sys)])
    omega2_data = np.vstack([omega2_t[i] for i in range(num_sys)])
    omega3_data = np.vstack([omega3_t[i] for i in range(num_sys)])
    
    # Use the natural frequencies directly - they represent the instantaneous transition frequencies
    sin_omega1_data = omega1_data
    sin_omega2_data = omega2_data
    sin_omega3_data = omega3_data
    
    # Stack all inputs: [v_ext, sin(ω₁t), sin(ω₂t), sin(ω₃t)]
    X_multi_input = np.stack([
        X_data,
        sin_omega1_data,
        sin_omega2_data,
        sin_omega3_data
    ], axis=-1)

    return X_multi_input, y_data

def train_and_test_esn(X_train, Y_train, omega1_t_train, omega2_t_train, omega3_t_train, t_on_train, num_sys_train,
                       X_test, Y_test, omega1_t_test, omega2_t_test, omega3_t_test, t_on_test, num_sys_test,
                       params=None, plot=False, verbose=0, seed=None):
    """
    Train ESN on training data and evaluate on both training and testing data.
    Returns metrics for both datasets.
    
    Args:
        seed: Random seed for reproducible weight initialization
    """
    if params is None:
        # Default parameters if optimization is skipped
        default_input_scaling = np.ones(NUM_INPUT_CHANNELS)  # v_ext + 3 omega signals
        default_input_connectivity = np.ones(NUM_INPUT_CHANNELS) * 0.1  # Default connectivity for each channel
        params = {
            'sr': 1.693,
            'units': 200,
            'lr': 0.028,
            'ridge': 2.973e-8,
            'input_scaling': default_input_scaling,
            'input_connectivity': default_input_connectivity,
            'rc_connectivity': 0.1,
            'fb_scaling': 0.0,
            'fb_connectivity': 0.0
        }
    
    # Prepare training data
    X_train_prepared, y_train_prepared = Ready_Data(X_train, Y_train, omega1_t_train, omega2_t_train, omega3_t_train, num_sys_train)
    
    # Prepare testing data
    X_test_prepared, y_test_prepared = Ready_Data(X_test, Y_test, omega1_t_test, omega2_t_test, omega3_t_test, num_sys_test)

    print("\nTraining ESN with parameters:")
    if seed is not None:
        print(f"  Random seed: {seed}")
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
        elif isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    print(f"Training input shape: {X_train_prepared.shape}")
    print(f"Testing input shape: {X_test_prepared.shape}")
    
    warmup = int(t_on_train[0]/(2*dt))
    
    # Create a RandomState object for reproducibility
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Create custom input weight matrix with different connectivity for each channel
    input_dims = [1, 1, 1, 1]  # Each channel is 1-dimensional
    Win = create_custom_input_weights(
        units=params['units'],
        input_dims=input_dims,
        input_scalings=params['input_scaling'],
        input_connectivities=params['input_connectivity'],
        rng=rng  # Pass the RNG for reproducibility
    )
    
    # Create ESN with parameters and custom input weights
    # Use the same seed for the Reservoir to ensure consistent W matrix
    reservoir = Reservoir(
        units=params['units'], 
        lr=params['lr'], 
        sr=params['sr'],
        Win=Win,  # Use our custom input weight matrix
        input_bias=False,
        rc_connectivity=params['rc_connectivity'],
        fb_scaling=params['fb_scaling'],
        fb_connectivity=params['fb_connectivity'],
        seed=seed  # Pass seed to Reservoir for consistent W matrix
    )
    readout = Ridge(output_dim=1, ridge=params['ridge'])
    esn = ESN(reservoir=reservoir, readout=readout, workers=-1)
    
    # Train on training data
    start = time.time()
    esn = esn.fit(X_train_prepared, y_train_prepared[:, :, np.newaxis], warmup=warmup, reset=True)
    if verbose > 1:
        print("Training time:", "{:.2f}".format(time.time() - start), "seconds")
    
    # Evaluate on TRAINING data
    esn.reset()
    train_r2s = []
    train_rmses = []
    y_pred_train = np.zeros((num_sys_train, steps))
    
    X_train_systems = X_train_prepared.reshape(num_sys_train, steps, NUM_INPUT_CHANNELS)
    y_train_systems = y_train_prepared.reshape(num_sys_train, steps)
    
    print("\n--- Training Data Performance ---")
    for system in range(num_sys_train):
        esn.reset()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            y_pred = esn.run(X_train_systems[system, :, :], reset=True)[:, 0]
        
        y_pred_train[system, :] = y_pred
        
        system_r2 = rsquare(y_train_systems[system, warmup:], y_pred[warmup:])
        system_rmse = rmse(y_train_systems[system, warmup:], y_pred[warmup:])
        
        train_r2s.append(system_r2)
        train_rmses.append(system_rmse)
        if verbose > 0: 
            print(f'Train System {system}. R2: {system_r2:.6f}. RMSE: {system_rmse:.6f}')
    
    train_avg_r2 = np.mean(train_r2s)
    train_avg_rmse = np.mean(train_rmses)
    train_max_rmse = np.max(train_rmses)
    print(f'Training - Average RMSE: {train_avg_rmse:.6f}, Max RMSE: {train_max_rmse:.6f}, Average R2: {train_avg_r2:.6f}')
    
    # Evaluate on TESTING data
    esn.reset()
    test_r2s = []
    test_rmses = []
    y_pred_test = np.zeros((num_sys_test, steps))
    
    X_test_systems = X_test_prepared.reshape(num_sys_test, steps, NUM_INPUT_CHANNELS)
    y_test_systems = y_test_prepared.reshape(num_sys_test, steps)
    
    print("\n--- Testing Data Performance ---")
    for system in range(num_sys_test):
        esn.reset()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            y_pred = esn.run(X_test_systems[system, :, :], reset=True)[:, 0]
        
        y_pred_test[system, :] = y_pred
        
        system_r2 = rsquare(y_test_systems[system, warmup:], y_pred[warmup:])
        system_rmse = rmse(y_test_systems[system, warmup:], y_pred[warmup:])
        
        test_r2s.append(system_r2)
        test_rmses.append(system_rmse)
        if verbose > 0: 
            print(f'Test System {system}. R2: {system_r2:.6f}. RMSE: {system_rmse:.6f}')
        
        if plot:  # Plot every 10th system
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Top subplot: Density comparison
            ax1.plot(t_array[warmup:], y_test_systems[system, warmup:], label='True', linewidth=2)
            ax1.plot(t_array[warmup:], y_pred[warmup:], label='Predicted', linewidth=2, alpha=0.8)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Density')
            ax1.set_title(f'Test System {system} - R²={system_r2:.4f}, RMSE={system_rmse:.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Inset for detailed view
            ax_inset = ax1.inset_axes([0.6, 0.6, 0.35, 0.35])
            start_idx = int((t_on_test[0])/dt) + np.random.choice(len(t_array[int(t_on_test[0]/dt):int(800/dt)]))
            stop_idx = start_idx + int(30/dt)
            ax_inset.plot(t_array[start_idx:stop_idx], y_test_systems[system, start_idx:stop_idx], linewidth=2)
            ax_inset.plot(t_array[start_idx:stop_idx], y_pred[start_idx:stop_idx], linewidth=2, alpha=0.8)
            ax_inset.grid(True, alpha=0.3)
            ax_inset.set_title('Detail View', fontsize=10)
            
            # Bottom subplot: Input signals
            channel_names = ['v_ext', 'ω₁', 'ω₂', 'ω₃']
            colors = ['blue', 'red', 'green', 'orange']
            
            for channel in range(NUM_INPUT_CHANNELS):
                ax2.plot(t_array[warmup:], X_test_systems[system, warmup:, channel], 
                        label=channel_names[channel], linewidth=1.5, color=colors[channel])
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Input Signal Amplitude')
            ax2.set_title(f'Input Signals for Test System {system}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    test_avg_r2 = np.mean(test_r2s)
    test_avg_rmse = np.mean(test_rmses)
    test_max_rmse = np.max(test_rmses)
    print(f'Testing - Average RMSE: {test_avg_rmse:.6f}, Max RMSE: {test_max_rmse:.6f}, Average R2: {test_avg_r2:.6f}')
    
    # Return all metrics
    return {
        'train': {
            'max_rmse': train_max_rmse,
            'avg_rmse': train_avg_rmse,
            'avg_r2': train_avg_r2,
            'system_rmses': np.array(train_rmses),
            'system_r2s': np.array(train_r2s),
            'y_pred': y_pred_train
        },
        'test': {
            'max_rmse': test_max_rmse,
            'avg_rmse': test_avg_rmse,
            'avg_r2': test_avg_r2,
            'system_rmses': np.array(test_rmses),
            'system_r2s': np.array(test_r2s),
            'y_pred': y_pred_test
        }
    }

def objective(trial, X_train, Y_train, omega1_t_train, omega2_t_train, omega3_t_train, t_on_train, num_sys_train):
    """
    Objective function for Optuna to optimize the ESN hyperparameters.
    Optimizes for minimum maximum RMSE on training data.
    """
    units = trial.suggest_int('units', 100, 100)  # Now optimizing units between 100 and 1000
    
    # Create parameter dictionary
    params = {
        'sr': trial.suggest_float('sr', 1E-2, 2.0, log=True),
        'units': units,
        'lr': trial.suggest_float('lr', 1E-5, 1.0, log=True),
        'ridge': trial.suggest_float('ridge', 1e-8, 1e-3, log=True),
        'rc_connectivity': trial.suggest_float('rc_connectivity', 1E-3, 1.0, log=False),
        'fb_scaling': trial.suggest_float('fb_scaling', 1E-5, 1E3, log=True),
        'fb_connectivity': trial.suggest_float('fb_connectivity', 2E-3, 1.0, log=False)
    }
    
    # Add input scaling and connectivity for each channel
    input_scaling_list = []
    input_connectivity_list = []
    channel_names = ['v_ext', 'sin_omega1', 'sin_omega2', 'sin_omega3']
    
    for i, channel_name in enumerate(channel_names):
        scaling = trial.suggest_float(f'input_scaling_{channel_name}', 1E-5, 1E2, log=True)
        connectivity = trial.suggest_float(f'input_connectivity_{channel_name}', 2E-3, 1.0, log=False)
        input_scaling_list.append(scaling)
        input_connectivity_list.append(connectivity)
    
    params['input_scaling'] = np.array(input_scaling_list)
    params['input_connectivity'] = np.array(input_connectivity_list)
    
    # Use trial number as seed for reproducibility within this trial
    trial_seed = 42 + trial.number  # Base seed + trial number for unique but reproducible seeds
    
    # Store the seed in trial user attributes for later retrieval
    trial.set_user_attr("seed", trial_seed)
    
    # For optimization, we only evaluate on training data (no need to run test data each trial)
    # Create a simplified training function for optimization
    X_prepared, y_prepared = Ready_Data(X_train, Y_train, omega1_t_train, omega2_t_train, omega3_t_train, num_sys_train)
    
    warmup = int(t_on_train[0]/(2*dt))
    
    # Create a RandomState object for reproducibility
    rng = np.random.RandomState(trial_seed)
    
    input_dims = [1, 1, 1, 1]
    Win = create_custom_input_weights(
        units=params['units'],
        input_dims=input_dims,
        input_scalings=params['input_scaling'],
        input_connectivities=params['input_connectivity'],
        rng=rng  # Pass the RNG for reproducibility
    )
    
    reservoir = Reservoir(
        units=params['units'], 
        lr=params['lr'], 
        sr=params['sr'],
        Win=Win,
        input_bias=False,
        rc_connectivity=params['rc_connectivity'],
        fb_scaling=params['fb_scaling'],
        fb_connectivity=params['fb_connectivity'],
        seed=trial_seed  # Pass seed to Reservoir for consistent W matrix
    )
    readout = Ridge(output_dim=1, ridge=params['ridge'])
    esn = ESN(reservoir=reservoir, readout=readout, workers=-1)
    
    esn = esn.fit(X_prepared, y_prepared[:, :, np.newaxis], warmup=warmup, reset=True)
    
    # Calculate max RMSE on training data
    esn.reset()
    rmses = []
    X_systems = X_prepared.reshape(num_sys_train, steps, NUM_INPUT_CHANNELS)
    y_systems = y_prepared.reshape(num_sys_train, steps)
    
    for system in range(num_sys_train):
        esn.reset()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            y_pred = esn.run(X_systems[system, :, :], reset=True)[:, 0]
        
        system_rmse = rmse(y_systems[system, warmup:], y_pred[warmup:])
        rmses.append(system_rmse)
    
    max_rmse = np.max(rmses)
    return max_rmse

def train_optimal_esn_with_test(study_name, n_trials=500, load_if_exists=True):
    """
    Perform hyperparameter optimization on training data and evaluate on both training and testing data.
    """
    # Create or load an Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # Minimize max RMSE
        load_if_exists=load_if_exists,
        storage=f"sqlite:///{study_name}.db"
    )
    
    # Optimize hyperparameters using the training dataset
    print(f"Starting hyperparameter optimization with {n_trials} trials on training dataset...")
    print("Optimizing for minimum MAXIMUM RMSE across all training systems...")
    study.optimize(
        lambda trial: objective(trial, X_train, Y_train, omega1_t_train, omega2_t_train, omega3_t_train, t_on_train, num_sys_train), 
        n_trials=n_trials
    )
    
    # Print optimization results
    print("\nOptimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best max RMSE on training dataset: {study.best_value:.6f}")
    
    # Retrieve the seed used for the best trial
    best_trial_seed = study.best_trial.user_attrs.get("seed", 42 + study.best_trial.number)
    print(f"Best trial seed: {best_trial_seed}")
    
    print("\nBest hyperparameters:")
    for param_name, param_value in study.best_params.items():
        print(f"  {param_name}: {param_value}")
    
    # Reconstruct the best parameters for final training and testing
    channel_names = ['v_ext', 'sin_omega1', 'sin_omega2', 'sin_omega3']
    input_scaling_list = []
    input_connectivity_list = []
    for channel_name in channel_names:
        input_scaling_list.append(study.best_params[f'input_scaling_{channel_name}'])
        input_connectivity_list.append(study.best_params[f'input_connectivity_{channel_name}'])
    
    best_params = {
        'sr': study.best_params['sr'],
        'units': study.best_params['units'],  # Use the optimized number of units
        'lr': study.best_params['lr'],
        'ridge': study.best_params['ridge'],
        'input_scaling': np.array(input_scaling_list),
        'input_connectivity': np.array(input_connectivity_list),
        'rc_connectivity': study.best_params['rc_connectivity'],
        'fb_scaling': study.best_params['fb_scaling'],
        'fb_connectivity': study.best_params['fb_connectivity']
    }
    
    # Create visualization plots
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.show()
        
        fig = optuna.visualization.plot_slice(study)
        fig.show()
    except (ImportError, RuntimeError) as e:
        print(f"Visualization error: {e}")
        print("Optuna visualizations require plotly to be installed.")
    
    # Train final model and evaluate on both training and testing data using the best seed
    print(f"\nTraining final model with best parameters using seed {best_trial_seed}...")
    print("This ensures the same weight matrices as the best trial.")
    print("Input connectivity per channel:", best_params['input_connectivity'])
    
    results = train_and_test_esn(
        X_train, Y_train, omega1_t_train, omega2_t_train, omega3_t_train, t_on_train, num_sys_train,
        X_test, Y_test, omega1_t_test, omega2_t_test, omega3_t_test, t_on_test, num_sys_test,
        params=best_params, plot=True, seed=best_trial_seed
    )
    
    return results

# Run the optimization and evaluation
results = train_optimal_esn_with_test(
    study_name='test5_single_study_100', 
    n_trials=200,  # Set to 0 to load existing study, change to higher number for optimization
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
#with open('test1_single_results.json', mode='w') as f:
#    json.dump(final_results, f)

print("\nResults saved to 'train_test_results.json'")
print("\nSummary:")
print(f"Training - Max RMSE: {results['train']['max_rmse']:.6f}, Avg R²: {results['train']['avg_r2']:.6f}")
print(f"Testing  - Max RMSE: {results['test']['max_rmse']:.6f}, Avg R²: {results['test']['avg_r2']:.6f}")