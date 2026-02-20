#!/usr/bin/env python3
"""
Multi-electron eigenstate analysis for 1D chain dynamics.
Creates separate training and test datasets where test points are interpolated between training points.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.interpolate import interp1d
import json

def sigmoid(t, t_on):
    """Sigmoid function for smooth turn-on."""
    return 1 / (1 + np.exp(0.2 * (t - t_on)))

def v_sin(v0_static, v1_amp, omega_freq, t_on_time):
    """Create time-dependent potential function."""
    def v_t(t):
        v_td = v0_static.copy()
        envelope = sigmoid(t, t_on_time)
        driving_term = envelope * 2 * v1_amp * np.sin(omega_freq * t)
        v_td += driving_term
        return v_td
    return v_t

def p2d(v0, L, T, tmax, dt):
    N = L // 2      # Number of electrons (occupied states)
    t_on = 150.0     # Time to turn on driving
    
    def H_(v_sites, T_hop):
        """Build 1D chain Hamiltonian with PBC."""
        H = np.zeros((L, L), dtype=complex)
        
        # On-site energies
        for i in range(L):
            H[i, i] = v_sites[i]
        
        # Hopping terms with PBC
        for i in range(L):
            next_site = (i + 1) % L
            H[i, next_site] = T_hop
            H[next_site, i] = T_hop
        H[L-1,0] = 0.0
        H[0,L-1] = 0.0
            
        return H
    
    # Get initial eigenstates
    v_vec = np.zeros(L)
    #v_vec[1] = v0
    H0 = H_(v_vec, T)
    eigenvals, eigenvecs = eigh(H0)
    print(H0)
    # Sort by energy
    idx = np.argsort(eigenvals)
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    occupied_states = eigenvecs[:, :N].astype(complex)  # Shape: (L, N)
    print(f"\nOccupied orbital energies: {eigenvals[:N]}")
    print(f"Initial occupied states shape: {occupied_states.shape}")
    
    # =============================================================================
    # TIME-DEPENDENT POTENTIAL
    # =============================================================================
    def sigmoid_(t_off, gamma_off):
        def sigmoid__(t):
            return 1 / (1 + np.exp(-1*gamma_off * (t - t_off)))
        return sigmoid__
    
    def delta_kick(v0, t_off, gamma_off):
        sigmoid_temp = sigmoid_(t_off, gamma_off)
        def v_t(t):
            v = v0*sigmoid_temp(t)*np.ones(L)
            v[L//2:] = 0
            v[0] = 0
            return v
        return v_t
    
    # Create time-dependent potential
    v_t = delta_kick(v0, 150, 0.2)
    
    # =============================================================================
    # TIME EVOLUTION
    # =============================================================================
    
    def f(t, y):
        """Time-dependent Schrödinger equation for multiple states."""
        # Reshape y to (L, N) matrix
        psi_matrix = y.reshape((L, N))
        
        # Apply Hamiltonian to each column (state)
        H_t = H_(v_t(t), T)
        dpsi_dt = -1j * H_t @ psi_matrix
        
        # Flatten back to 1D array
        return dpsi_dt.flatten()
    
    # Time array
    t_array = np.arange(0, tmax, dt)
    
    print(f"\nSolving time-dependent Schrödinger equation...")
    print(f"Time points: {len(t_array)}")
    print(f"Total states to propagate: {N}")
    
    # Initial condition: flatten the occupied states matrix
    y0 = occupied_states.flatten()
    
    # Solve ODE
    sln = solve_ivp(
        f, 
        [t_array[0], t_array[-1]], 
        np.complex128(y0), 
        method='DOP853', 
        t_eval=t_array, 
        rtol=1e-12, 
        atol=1e-12
    )
    psi_t = sln.y.reshape((L, N, len(sln.t)))
    print(f"Evolution completed. Final time: {sln.t[-1]:.2f}")
    total_density_t = np.sum(np.abs(psi_t)**2, axis=1)  # Shape: (L, time_points)
    v_array = np.array([v_t(t) for t in t_array])
    return v_array, total_density_t.T

def construct_dataset(v0, L, T, tmax, dt):
    num_sys = v0.shape[0]
    timesteps = int(tmax/dt)
    v_array = np.zeros((num_sys, timesteps, L))
    n_array = np.zeros((num_sys, timesteps, L))
    for i in range(num_sys):
        print(f"\n=== Processing system {i+1}/{num_sys} with v0={v0[i]:.4f} ===")
        result = p2d(v0[i], L, T, tmax, dt)
        v_array[i] = result[0]
        n_array[i] = result[1]
    return v_array, n_array

def create_frequency_interpolators(T, L, v_range=(-0.2, 0.2), n_points=10001): 
    def H_(v_sites, T_hop):
        """Build 1D chain Hamiltonian with PBC."""
        H = np.zeros((L, L), dtype=complex)
        
        # On-site energies
        for i in range(L):
            H[i, i] = v_sites[i]
        
        # Hopping terms with PBC
        for i in range(L):
            next_site = (i + 1) % L
            H[i, next_site] = T_hop
            H[next_site, i] = T_hop
        
        H[L-1,0] = 0.0
        H[0,L-1] = 0.0
            
        return H
    
    v_static_array = np.linspace(v_range[0], v_range[1], n_points)
    v_array = np.zeros((n_points,L))
    omega_array = np.zeros((n_points, L-1))
    v_array[:,1] = v_static_array
    for i in range(n_points):
        eigs = np.linalg.eigvalsh(H_(v_array[i], T))
        omega_array[i] = eigs[1:] - eigs[:-1]
    kind = 'linear'
    # Create interpolation functions
    omega1_interp = interp1d(v_static_array, omega_array[:,0], kind=kind, 
                            bounds_error=False, fill_value='extrapolate')
    omega2_interp = interp1d(v_static_array, omega_array[:,1], kind=kind, 
                            bounds_error=False, fill_value='extrapolate')
    omega3_interp = interp1d(v_static_array, omega_array[:,2], kind=kind, 
                            bounds_error=False, fill_value='extrapolate')
    
    return omega1_interp, omega2_interp, omega3_interp

# Parameters
L = 4
T = 0.05
tmax = 1200
dt = 0.2
num_sys_train = 101
num_sys_test = num_sys_train - 1

# Create training dataset v0 values
v0_train = np.linspace(0.0, 0.1, num_sys_train)

# Create test dataset v0 values (interpolated between training points)
v0_test = np.zeros(num_sys_test)
for i in range(num_sys_test):
    v0_test[i] = (v0_train[i] + v0_train[i+1]) / 2

print("="*60)
print("DATASET GENERATION")
print("="*60)
print(f"\nTraining v0 values: {v0_train}")
print(f"Test v0 values: {v0_test}")
print(f"\nNumber of training systems: {num_sys_train}")
print(f"Number of test systems: {num_sys_test}")

# Generate training dataset
print("\n" + "="*60)
print("GENERATING TRAINING DATASET")
print("="*60)
v_array_train, n_array_train = construct_dataset(v0_train, L, T, tmax, dt)

# Generate test dataset
print("\n" + "="*60)
print("GENERATING TEST DATASET")
print("="*60)
v_array_test, n_array_test = construct_dataset(v0_test, L, T, tmax, dt)

# Create frequency interpolators
print("\n" + "="*60)
print("CREATING FREQUENCY INTERPOLATORS")
print("="*60)
omega1_interp, omega2_interp, omega3_interp = create_frequency_interpolators(
    T=T, L=L, v_range=(-0.2, 0.2), n_points=10001
)

t_array = np.linspace(0, tmax-dt, int(tmax/dt))

# Process training frequencies
omega1_t_train = omega1_interp(v_array_train[:,:,1])
omega2_t_train = omega2_interp(v_array_train[:,:,1])
omega3_t_train = omega3_interp(v_array_train[:,:,1])

omega1_t_train = np.sin(omega1_t_train * t_array)
omega2_t_train = np.sin(omega2_t_train * t_array)
omega3_t_train = np.sin(omega3_t_train * t_array)

# Process test frequencies
omega1_t_test = omega1_interp(v_array_test[:,:,1])
omega2_t_test = omega2_interp(v_array_test[:,:,1])
omega3_t_test = omega3_interp(v_array_test[:,:,1])

omega1_t_test = np.sin(omega1_t_test * t_array)
omega2_t_test = np.sin(omega2_t_test * t_array)
omega3_t_test = np.sin(omega3_t_test * t_array)

# Create training data dictionary
train_data = {
    'v_array': v_array_train.tolist(),
    'n_array': n_array_train.tolist(),
    'omega1_t': omega1_t_train.tolist(),
    'omega2_t': omega2_t_train.tolist(),
    'omega3_t': omega3_t_train.tolist(),
    'v0_values': v0_train.tolist(),
    'L': L,
    'num_sys': num_sys_train,
    'T': T,
    'dt': dt,
    'tmax': tmax
}

# Create test data dictionary
test_data = {
    'v_array': v_array_test.tolist(),
    'n_array': n_array_test.tolist(),
    'omega1_t': omega1_t_test.tolist(),
    'omega2_t': omega2_t_test.tolist(),
    'omega3_t': omega3_t_test.tolist(),
    'v0_values': v0_test.tolist(),
    'L': L,
    'num_sys': num_sys_test,
    'T': T,
    'dt': dt,
    'tmax': tmax
}

# Save training data
print("\n" + "="*60)
print("SAVING DATASETS")
print("="*60)
with open(f'test4_train_data_{tmax}.json', mode='w') as f:
    json.dump(train_data, f)
print("Training data saved to: test4_train_data.json")

# Save test data
with open(f'test4_test_data_{tmax}.json', mode='w') as f:
    json.dump(test_data, f)
print("Test data saved to: test4_test_data.json")

print("\n" + "="*60)
print("DATASET GENERATION COMPLETE")
print("="*60)
print(f"\nSummary:")
print(f"- Training systems: {num_sys_train} with v0 in [{v0_train.min():.4f}, {v0_train.max():.4f}]")
print(f"- Test systems: {num_sys_test} with v0 interpolated between training points")
print(f"- Each system has {int(tmax/dt)} timesteps")
print(f"- System size: L = {L}")
print(f"- Hopping parameter: T = {T}")