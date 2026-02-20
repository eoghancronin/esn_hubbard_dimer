"""
Publication Figure Generation Script
=====================================
This script generates all figures for the manuscript comparing Echo State Network (ESN)
predictions with exact quantum dynamics simulations.

Figures generated:
- Figure 3: Eigenstate properties and input signals
- Figure 4: Time series comparison with FFT analysis
- Figure 5: Reservoir size comparison
- Figure 6: Decay rate comparison
- Figure 7: Driving frequency sweep (contour plots)
- Figure 8: Four-site lattice dynamics (contour plots)

Author: [Your name]
Date: [Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import json
from ESN_data_gen_v1_3_3 import dataset, json_file_to_arrays

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def H_(dv, t):
    """
    Construct the Hamiltonian matrix for the three-level system.
    
    Parameters:
    -----------
    dv : float
        Detuning parameter
    t : float
        Hopping parameter
    
    Returns:
    --------
    h : ndarray (3, 3)
        Hamiltonian matrix
    """
    h = np.zeros((3, 3))
    h[0, 0] = 2 * (dv / 2) + 1
    h[0, 1] = -np.sqrt(2) * t
    h[1, 0] = -np.sqrt(2) * t
    h[1, 2] = -np.sqrt(2) * t
    h[2, 2] = -2 * (dv / 2) + 1
    h[2, 1] = -np.sqrt(2) * t
    return h

def H__(v_sites, T_hop, L):
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


def wf2d_(wf):
    """
    Calculate density difference from wavefunction.
    
    Parameters:
    -----------
    wf : ndarray
        Wavefunction vector
    
    Returns:
    --------
    float
        Density difference 2*(|ψ₀|² - |ψ₂|²)
    """
    dm = np.outer(np.conjugate(wf), wf)
    return 2 * (dm[0, 0] - dm[2, 2]).real


def load_results(filename):
    """
    Load results from JSON file and convert to proper scale.
    
    Parameters:
    -----------
    filename : str
        Path to JSON results file
    
    Returns:
    --------
    dict
        Results dictionary with predictions scaled to [-2, 2]
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    # Convert predictions from [0,1] to [-2,2] scale
    results['test']['y_pred_all'] = 2 * (np.array(results['test']['y_pred_all']) - 1)
    if 'train' in results:
        results['train']['y_pred_all'] = 2 * (np.array(results['train']['y_pred_all']) - 1)
    
    return results


# ============================================================================
# GLOBAL PLOTTING PARAMETERS
# ============================================================================

# Font sizes
TITLE_SIZE = 14
LABEL_SIZE = 14
TICK_SIZE = 14
LEGEND_SIZE = 12
INSET_LABEL_SIZE = 12
INSET_TICK_SIZE = 8

# Time parameters
dt = 0.2  # Time step
warmup_time = 75  # Warm-up period duration

# Hopping parameter
T_HOPPING = 0.05


# ============================================================================
# FIGURE 3: EIGENSTATE PROPERTIES AND INPUT SIGNALS
# ============================================================================

def generate_figure_3():
    """
    Generate Figure 3 showing:
    (a) Density difference vs detuning for three eigenstates
    (b) Energy levels vs detuning
    (c) Input signal time series
    """
    print("Generating Figure 3...")
    
    # Load test data
    data_arrays = json_file_to_arrays('./test1_test_data.json', observables=True)
    X, Y, Z, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, omega1_t, omega2_t, omega3_t = data_arrays
    Y = 2 * (Y - 1)  # Convert to [-2, 2] scale
    
    # Calculate eigenstate properties over detuning range
    num_points = 1000
    detuning_array = np.linspace(-2, 2, num_points)
    
    # Arrays to store results
    density_psi0 = np.zeros(num_points)
    density_psi1 = np.zeros(num_points)
    density_psi2 = np.zeros(num_points)
    energy_0 = np.zeros(num_points)
    energy_1 = np.zeros(num_points)
    energy_2 = np.zeros(num_points)
    
    for idx, detuning in enumerate(detuning_array):
        eigenvalues, eigenvectors = np.linalg.eigh(H_(detuning, T_HOPPING))
        density_psi0[idx] = wf2d_(eigenvectors[:, 0])
        density_psi1[idx] = wf2d_(eigenvectors[:, 1])
        density_psi2[idx] = wf2d_(eigenvectors[:, 2])
        energy_0[idx] = eigenvalues[0]
        energy_1[idx] = eigenvalues[1]
        energy_2[idx] = eigenvalues[2]
    
    # Create figure with three subplots
    fig, (ax_density, ax_energy, ax_signals) = plt.subplots(3, 1, figsize=(6, 9))
    
    # Time array
    t_array = np.arange(len(X[0])) * dt
    test_index = 19  # Example trajectory to plot
    
    # -------------------------------------------------------------------------
    # Panel (a): Density difference vs detuning
    # -------------------------------------------------------------------------
    ax_density.plot(detuning_array, density_psi0, label=r'$\Delta n (\Psi_0)$')
    ax_density.plot(detuning_array, density_psi1, label=r'$\Delta n (\Psi_1)$')
    ax_density.plot(detuning_array, density_psi2, label=r'$\Delta n (\Psi_2)$')
    ax_density.set_xlabel(r'$\Delta v$', fontsize=LABEL_SIZE)
    ax_density.set_ylabel(r'$\Delta n$', fontsize=LABEL_SIZE)
    ax_density.legend(fontsize=LEGEND_SIZE, loc='lower left')
    ax_density.tick_params(labelsize=TICK_SIZE)
    
    # Add inset zooming into small detuning region
    inset_density = inset_axes(ax_density, width="40%", height="40%", loc='upper right')
    inset_density.plot(detuning_array, density_psi0)
    inset_density.plot(detuning_array, density_psi1)
    inset_density.plot(detuning_array, density_psi2)
    inset_density.set_xlim(-0.02, 0.02)
    inset_density.set_xticks([-0.02, 0.0, 0.02])
    inset_density.set_xlabel(r'$\Delta v$', fontsize=INSET_LABEL_SIZE)
    inset_density.set_ylabel(r'$\Delta n$', fontsize=INSET_LABEL_SIZE, labelpad=-4.0)
    inset_density.tick_params(labelsize=INSET_TICK_SIZE)
    
    # -------------------------------------------------------------------------
    # Panel (b): Energy levels vs detuning
    # -------------------------------------------------------------------------
    ax_energy.plot(detuning_array, energy_0, label=r'$E_0$')
    ax_energy.plot(detuning_array, energy_1, label=r'$E_1$')
    ax_energy.plot(detuning_array, energy_2, label=r'$E_2$')
    ax_energy.set_xlabel(r'$\Delta v$', fontsize=LABEL_SIZE)
    ax_energy.set_ylabel(r'$E_i$', fontsize=LABEL_SIZE)
    ax_energy.legend(fontsize=LEGEND_SIZE, loc='center left')
    ax_energy.tick_params(labelsize=TICK_SIZE)
    
    # -------------------------------------------------------------------------
    # Panel (c): Input signals time series
    # -------------------------------------------------------------------------
    ax_signals.plot(t_array, omega1_t[test_index], c='C3', 
                   label=r'$\sin(\omega_{0,1}[\Delta v](t) \cdot t)$')
    ax_signals.plot(t_array, omega3_t[test_index], c='C4', 
                   label=r'$\sin(\omega_{1,2}[\Delta v](t) \cdot t)$')
    ax_signals.plot(t_array, X[test_index], c='C5', 
                   label=r'$\Delta v (t)$')
    ax_signals.set_xlim(50, 250)
    ax_signals.legend(loc='lower right', ncols=2, labelspacing=0.05, 
                     framealpha=0.6, markerscale=0.1, handlelength=1.5,
                     columnspacing=1.0, handletextpad=0.2, 
                     borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
    ax_signals.set_xlabel('$t$', fontsize=LABEL_SIZE)
    ax_signals.set_ylabel(r'$\mathbf{u}(t)$', fontsize=LABEL_SIZE)
    ax_signals.tick_params(labelsize=TICK_SIZE)
    
    # Add panel labels
    ax_density.text(0.0, 1.02, '(a)', transform=ax_density.transAxes, 
                   fontsize=TITLE_SIZE, fontweight='bold')
    ax_energy.text(0.0, 1.02, '(b)', transform=ax_energy.transAxes, 
                  fontsize=TITLE_SIZE, fontweight='bold')
    ax_signals.text(0.0, 1.02, '(c)', transform=ax_signals.transAxes, 
                   fontsize=TITLE_SIZE, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92,
                       wspace=0.12, hspace=0.25)
    
    # Save figure
    plt.savefig('new_plots/Fig3.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved!")


# ============================================================================
# FIGURE 4: TIME SERIES AND FFT COMPARISON
# ============================================================================

def generate_figure_4():
    """
    Generate Figure 4 showing:
    (a) Input detuning time series for two test cases
    (b) Predicted vs exact density time series
    (c) FFT analysis comparing frequency content
    """
    print("Generating Figure 4...")
    
    # Load results
    results_single = load_results('test1_direct_results.json')
    results_multi = load_results('test1_augmented_results.json')
    
    # Load test data
    data_arrays = json_file_to_arrays('./test1_test_data.json', observables=True)
    X, Y, Z, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, omega1_t, omega2_t, omega3_t = data_arrays
    Y = 2 * (Y - 1)
    
    # Extract predictions
    y_pred_single = results_single['test']['y_pred_all'][:,:,0]
    y_pred_multi = results_multi['test']['y_pred_all'][:,:,0]
    
    # Test system parameters
    delta_v0_test = X[:, 0]
    
    # Select two example trajectories
    idx_1 = 19
    idx_2 = 99
    
    # Pre-calculate rounded detuning values for labels
    dv_1_str = str(np.round(delta_v0_test[idx_1], 2))
    dv_2_str = str(np.round(delta_v0_test[idx_2], 2))
    
    # Time array
    t_array = np.arange(len(X[0])) * dt
    
    # Calculate FFT for first trajectory (after warmup)
    fft_start_idx = int(150 / dt)
    fft_exact = np.fft.rfft(Y[idx_1, fft_start_idx:], norm="forward")
    fft_single = np.fft.rfft(y_pred_single[idx_1, fft_start_idx:], norm="forward")
    fft_multi = np.fft.rfft(y_pred_multi[idx_1, fft_start_idx:], norm="forward")
    fft_freq = np.fft.rfftfreq(len(Y[idx_1, fft_start_idx:]), dt)
    
    # Calculate characteristic frequencies at zero detuning
    eigenvalues = np.linalg.eigvalsh(H_(0.00, T_HOPPING))
    omega_01 = eigenvalues[1] - eigenvalues[0]
    omega_02 = eigenvalues[2] - eigenvalues[0]
    omega_12 = eigenvalues[2] - eigenvalues[1]
    
    # Print RMSE statistics
    print('Multi-input ESN:')
    print(f'  Max RMSE (train): {2*results_multi["train"]["max_rmse"]:.6f}')
    print(f'  Max RMSE (test):  {2*results_multi["test"]["max_rmse"]:.6f}')
    print(f'  Avg RMSE (train): {2*results_multi["train"]["avg_rmse"]:.6f}')
    print(f'  Avg RMSE (test):  {2*results_multi["test"]["avg_rmse"]:.6f}')
    print('\nSingle-input ESN:')
    print(f'  Max RMSE (train): {2*results_single["train"]["max_rmse"]:.6f}')
    print(f'  Max RMSE (test):  {2*results_single["test"]["max_rmse"]:.6f}')
    print(f'  Avg RMSE (train): {2*results_single["train"]["avg_rmse"]:.6f}')
    print(f'  Avg RMSE (test):  {2*results_single["test"]["avg_rmse"]:.6f}')
    
    # Create figure
    scale = 1.1
    fig, (ax_input, ax_timeseries, ax_fft) = plt.subplots(3, 1, figsize=(scale*7, scale*10))
    
    # Color scheme
    color_exact_1 = '#F18F01'  # Orange
    color_exact_2 = '#048A81'  # Teal
    color_multi_1 = '#2E86AB'  # Steel blue
    color_multi_2 = '#C73E1D'  # Red
    color_single_1 = '#A23B72'  # Magenta
    color_single_2 = '#592E83'  # Purple
    
    # -------------------------------------------------------------------------
    # Panel (a): Input detuning time series
    # -------------------------------------------------------------------------
    ax_input.plot(t_array, X[idx_1], label=rf'$\Delta v_0 = {dv_1_str}$', c=color_exact_1)
    ax_input.plot(t_array, X[idx_2], label=rf'$\Delta v_0 = {dv_2_str}$', c=color_exact_2)
    ax_input.set_xlabel(r'$t$', fontsize=LABEL_SIZE)
    ax_input.set_ylabel(r'$\Delta v(t)$', fontsize=LABEL_SIZE)
    ax_input.set_xlim(0, 650)
    ax_input.tick_params(labelsize=TICK_SIZE)
    ax_input.legend(loc='upper right', ncols=1, labelspacing=0.1, 
                   framealpha=0.6, markerscale=0.5, fontsize=LEGEND_SIZE)
    
    # -------------------------------------------------------------------------
    # Panel (b): Time series comparison
    # -------------------------------------------------------------------------
    # Trajectory 1
    ax_timeseries.plot(t_array, y_pred_multi[idx_1], 
                      label=rf'$\Delta n^{{\text{{ESN}}}}_\omega(t)[\Delta v_0 = {dv_1_str}]$',
                      c=color_multi_1, linewidth=1.5)
    ax_timeseries.plot(t_array, y_pred_single[idx_1],
                      label=rf'$\Delta n^{{\text{{ESN}}}}(t)[\Delta v_0 = {dv_1_str}]$',
                      c=color_single_1, linewidth=1.5)
    ax_timeseries.plot(t_array, Y[idx_1],
                      label=rf'$\Delta n^{{\text{{Exact}}}}(t)[\Delta v_0 = {dv_1_str}]$',
                      c=color_exact_1, linestyle=(0, (10, 10)), linewidth=1.5)
    
    # Trajectory 2
    ax_timeseries.plot(t_array, y_pred_multi[idx_2],
                      label=rf'$\Delta n^{{\text{{ESN}}}}_\omega(t)[\Delta v_0 = {dv_2_str}]$',
                      c=color_multi_2, linewidth=1.5)
    ax_timeseries.plot(t_array, y_pred_single[idx_2],
                      label=rf'$\Delta n^{{\text{{ESN}}}}(t)[\Delta v_0 = {dv_2_str}]$',
                      c=color_single_2, linewidth=1.5)
    ax_timeseries.plot(t_array, Y[idx_2],
                      label=rf'$\Delta n^{{\text{{Exact}}}}(t)[\Delta v_0 = {dv_2_str}]$',
                      c=color_exact_2, linestyle=(0, (10, 10)), linewidth=1.5)
    
    ax_timeseries.set_xlabel(r'$t$', fontsize=LABEL_SIZE)
    ax_timeseries.set_ylabel(r'$\Delta n(t)$', fontsize=LABEL_SIZE)
    ax_timeseries.tick_params(labelsize=TICK_SIZE)
    ax_timeseries.set_ylim((-2.0, 2.2))
    ax_timeseries.set_xlim(0, 650)
    ax_timeseries.axvline(x=warmup_time, color='black', linestyle='dotted', 
                         alpha=0.8, linewidth=2, label=r'$t_{\mathrm{warm-up}}$')
    ax_timeseries.legend(loc='lower left', ncols=2, labelspacing=0.05,
                        framealpha=0.6, markerscale=0.1, handlelength=1.5,
                        columnspacing=1.0, handletextpad=0.2,
                        borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
    
    # Add inset for detailed view
    ax_inset = ax_timeseries.inset_axes([0.6, 0.62, 0.35, 0.35])
    inset_start_idx = int(400 / dt)
    inset_stop_idx = inset_start_idx + int(30 / dt)
    ax_inset.plot(t_array[inset_start_idx:inset_stop_idx],
                 Y[idx_1, inset_start_idx:inset_stop_idx],
                 linewidth=2, c=color_exact_1)
    ax_inset.plot(t_array[inset_start_idx:inset_stop_idx],
                 y_pred_multi[idx_1, inset_start_idx:inset_stop_idx],
                 linewidth=2, c=color_multi_1)
    ax_inset.plot(t_array[inset_start_idx:inset_stop_idx],
                 y_pred_single[idx_1, inset_start_idx:inset_stop_idx],
                 linewidth=2, c=color_single_1)
    ax_inset.set_xlabel(r'$t$', labelpad=0.1, fontsize=INSET_LABEL_SIZE)
    ax_inset.set_ylabel(r'$\Delta n (t)$', labelpad=0.1, fontsize=INSET_LABEL_SIZE)
    
    # -------------------------------------------------------------------------
    # Panel (c): FFT analysis
    # -------------------------------------------------------------------------
    # Reference frequencies
    ax_fft.axvline(omega_01, label=r'$\omega_{0,1}[\Delta v = 0]$',
                  linestyle='solid', c='black')
    ax_fft.axvline(omega_02, label=r'$\omega_{0,2}[\Delta v = 0]$',
                  linestyle='dotted', c='black')
    ax_fft.axvline(omega_12, label=r'$\omega_{1,2}[\Delta v = 0]$',
                  linestyle='dashed', c='black')
    
    # FFT data
    ax_fft.plot(2*np.pi*fft_freq, np.abs(fft_exact),
               label=rf'$\Delta \tilde{{n}}^{{\text{{Exact}}}}(t) [\Delta v_0 = {dv_1_str}]$',
               c=color_exact_1)
    ax_fft.plot(2*np.pi*fft_freq, np.abs(fft_multi),
               label=rf'$\Delta \tilde{{n}}_\omega^{{\text{{ESN}}}}(t) [\Delta v_0 = {dv_1_str}]$',
               c=color_multi_1)
    ax_fft.plot(2*np.pi*fft_freq, np.abs(fft_single),
               label=rf'$\Delta \tilde{{n}}^{{\text{{ESN}}}}(t) [\Delta v_0 = {dv_1_str}]$',
               c=color_single_1)
    
    ax_fft.set_xlabel(r'$\omega$', fontsize=LABEL_SIZE)
    ax_fft.set_xscale('log')
    ax_fft.set_yscale('log')
    ax_fft.legend(loc='upper center', ncols=2, labelspacing=0.05,
                 framealpha=0.6, markerscale=0.1, handlelength=1.5,
                 columnspacing=1.0, handletextpad=0.2,
                 borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
    ax_fft.set_xlim(5E-3, 4)
    ax_fft.set_ylabel(r'$\Delta \tilde{n} (\omega)$', fontsize=LABEL_SIZE)
    ax_fft.tick_params(labelsize=TICK_SIZE)
    
    # Add FFT inset
    ax_fft_inset = ax_fft.inset_axes([0.2, 0.15, 0.35, 0.35])
    ax_fft_inset.axvline(omega_01, linestyle='solid', c='black', linewidth=1)
    ax_fft_inset.axvline(omega_02, linestyle='dotted', c='black', linewidth=1)
    ax_fft_inset.axvline(omega_12, linestyle='dashed', c='black', linewidth=1)
    ax_fft_inset.plot(2*np.pi*fft_freq, np.abs(fft_exact), c=color_exact_1, linewidth=2)
    ax_fft_inset.plot(2*np.pi*fft_freq, np.abs(fft_multi), c=color_multi_1, linewidth=2)
    ax_fft_inset.plot(2*np.pi*fft_freq, np.abs(fft_single), c=color_single_1, linewidth=2)
    ax_fft_inset.set_xlim(0.99, 1.04)
    ax_fft_inset.set_yscale('log')
    ax_fft_inset.set_xlabel(r'$\omega$', labelpad=0.1, fontsize=INSET_LABEL_SIZE)
    ax_fft_inset.set_ylabel(r'$\Delta \tilde{n} (\omega)$', labelpad=0.1, fontsize=INSET_LABEL_SIZE)
    ax_fft_inset.tick_params(labelsize=INSET_TICK_SIZE)
    
    # Add panel labels
    ax_input.text(-0.1, 1.02, '(a)', transform=ax_input.transAxes,
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax_timeseries.text(-0.1, 1.02, '(b)', transform=ax_timeseries.transAxes,
                      fontsize=TITLE_SIZE, fontweight='bold')
    ax_fft.text(-0.1, 1.02, '(c)', transform=ax_fft.transAxes,
               fontsize=TITLE_SIZE, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92,
                       wspace=0.2, hspace=0.3)
    
    # Save figure
    plt.savefig('new_plots/Fig4.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved!")


# ============================================================================
# FIGURE 5: RESERVOIR SIZE COMPARISON
# ============================================================================

def generate_figure_5():
    """
    Generate Figure 5 comparing ESN performance for different reservoir sizes.
    Shows time series predictions for N=100, 200, and 300 neurons.
    """
    print("Generating Figure 5...")
    
    # Load results for different reservoir sizes
    results_100 = load_results('test5_single_results_100.json')
    results_200 = load_results('test5_single_results_200.json')
    results_300 = load_results('test5_single_results_300.json')
    
    # Extract predictions
    y_pred_100 = results_100['test']['y_pred_all']
    y_pred_200 = results_200['test']['y_pred_all']
    y_pred_300 = results_300['test']['y_pred_all']
    
    # Load test data
    data_arrays = json_file_to_arrays('./test5_test_data.json', observables=True)
    X, Y, Z, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, omega1_t, omega2_t, omega3_t = data_arrays
    Y = 2 * (Y - 1)
    
    # Print statistics
    print(f'N=100 - Max RMSE (test): {results_100["test"]["max_rmse"]:.4f}')
    print(f'N=200 - Max RMSE (test): {results_200["test"]["max_rmse"]:.4f}')
    print(f'N=300 - Max RMSE (test): {results_300["test"]["max_rmse"]:.4f}')
    print(f'N=100 - Avg RMSE (test): {results_100["test"]["avg_rmse"]:.4f}')
    print(f'N=200 - Avg RMSE (test): {results_200["test"]["avg_rmse"]:.4f}')
    print(f'N=300 - Avg RMSE (test): {results_300["test"]["avg_rmse"]:.4f}')
    
    # Time array
    t_array = np.arange(len(X[0])) * dt
    
    # Select example trajectory
    idx = 19
    
    # Create figure
    scale = 1.1
    plt.figure(figsize=(scale*7, scale*3.3))
    
    plt.plot(t_array, Y[idx], label=r'$\Delta n^{\text{Exact}}(t)$', linewidth=1.5)
    plt.plot(t_array, y_pred_100[idx], label=r'$\Delta n^{\text{ESN}}(t)[N=100]$', linewidth=1.5)
    plt.plot(t_array, y_pred_200[idx], label=r'$\Delta n^{\text{ESN}}(t)[N=200]$', linewidth=1.5)
    plt.plot(t_array, y_pred_300[idx], label=r'$\Delta n^{\text{ESN}}(t)[N=300]$', linewidth=1.5)
    
    plt.xlabel(r'$t$', fontsize=LABEL_SIZE)
    plt.ylabel(r'$\Delta n(t)$', fontsize=LABEL_SIZE)
    plt.tick_params(labelsize=TICK_SIZE)
    plt.ylim((0.9, 2.0))
    plt.xlim(125, 250)
    plt.legend(loc='lower left', ncols=2, labelspacing=0.05,
              framealpha=0.6, markerscale=0.1, handlelength=1.5,
              columnspacing=1.0, handletextpad=0.2,
              borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
    
    plt.savefig('new_plots/Fig5.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 5 saved!")


# ============================================================================
# FIGURE 6: DECAY RATE COMPARISON
# ============================================================================

def generate_figure_6():
    """
    Generate Figure 6 comparing dynamics for different decay rates (gamma_off).
    Shows input signal and predicted vs exact density time series.
    """
    print("Generating Figure 6...")
    
    # Load results
    results_multi = load_results('test2_multi_results.json')
    y_pred_multi = results_multi['test']['y_pred_all']
    
    # Load test data
    data_arrays = json_file_to_arrays('./test2_test_data.json', observables=True)
    X, Y, Z, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, omega1_t, omega2_t, omega3_t = data_arrays
    Y = 2 * (Y - 1)
    
    # Calculate gamma_off test values (interpolated between training points)
    num_sys_train = 201
    num_sys_test = 200
    gamma_off_train = np.linspace(0.01, 0.1, num_sys_train)
    gamma_off_test = np.zeros(num_sys_test)
    for i in range(num_sys_test):
        gamma_off_test[i] = (gamma_off_train[i] + gamma_off_train[i+1]) / 2
    
    # Print statistics
    print('Multi-input ESN:')
    print(f'  Max RMSE (train): {2*results_multi["train"]["max_rmse"]:.4f}')
    print(f'  Max RMSE (test):  {2*results_multi["test"]["max_rmse"]:.4f}')
    print(f'  Avg RMSE (train): {2*results_multi["train"]["avg_rmse"]:.4f}')
    print(f'  Avg RMSE (test):  {2*results_multi["test"]["avg_rmse"]:.4f}')
    
    # Time array
    t_array = np.arange(len(X[0])) * dt
    
    # Select two example trajectories
    idx_1 = 22
    idx_2 = -1  # Last trajectory
    
    # Pre-calculate rounded gamma values for labels
    gamma_1_str = str(np.round(gamma_off_test[idx_1], 3))
    gamma_2_str = str(np.round(gamma_off_test[idx_2], 3))
    
    # Create figure
    scale = 1.1
    fig, (ax_input, ax_density) = plt.subplots(2, 1, figsize=(scale*7, scale*6))
    
    # Color scheme
    color_1 = '#F18F01'  # Orange
    color_2 = '#048A81'  # Teal
    color_pred_1 = '#2E86AB'  # Steel blue
    color_pred_2 = '#C73E1D'  # Red
    
    warmup_long = 800
    
    # -------------------------------------------------------------------------
    # Panel (a): Input detuning time series
    # -------------------------------------------------------------------------
    ax_input.plot(t_array, X[idx_1], label=rf'$\gamma_{{\mathrm{{off}}}} = {gamma_1_str}$', c=color_1)
    ax_input.plot(t_array, X[idx_2], label=rf'$\gamma_{{\mathrm{{off}}}} = {gamma_2_str}$', c=color_2)
    ax_input.set_xlabel('$t$', fontsize=LABEL_SIZE)
    ax_input.set_ylabel(r'$\Delta v(t)$', fontsize=LABEL_SIZE)
    ax_input.set_xlim(600, 1400)
    ax_input.tick_params(labelsize=TICK_SIZE)
    ax_input.axvline(x=warmup_long, color='black', linestyle='dotted',
                    alpha=0.8, linewidth=2, label=r'$t_{\mathrm{warm-up}}$')
    ax_input.legend(loc='upper left', ncols=1, labelspacing=0.1,
                   framealpha=0.6, markerscale=0.5, fontsize=LEGEND_SIZE)
    
    # -------------------------------------------------------------------------
    # Panel (b): Density time series comparison
    # -------------------------------------------------------------------------
    ax_density.plot(t_array, Y[idx_2],
                   label=rf'$\Delta n^{{\text{{Exact}}}}(t) [\gamma_{{\text{{off}}}} = {gamma_2_str}]$',
                   c=color_2)
    ax_density.plot(t_array, y_pred_multi[idx_2],
                   label=rf'$\Delta n^{{ESN}}_\omega(t)[\gamma_{{\text{{off}}}} = {gamma_2_str}]$',
                   c=color_pred_2)
    ax_density.plot(t_array, y_pred_multi[idx_1],
                   label=rf'$\Delta n^{{ESN}}_\omega(t)[\gamma_{{\text{{off}}}} = {gamma_1_str}]$',
                   c=color_pred_1, linewidth=1.5)
    ax_density.plot(t_array, Y[idx_1],
                   label=rf'$\Delta n^{{\text{{Exact}}}}(t) [\gamma_{{\text{{off}}}} = {gamma_1_str}]$',
                   c=color_1, linestyle=(0, (10, 10)), linewidth=1.5)
    
    ax_density.set_xlabel('$t$', fontsize=LABEL_SIZE)
    ax_density.set_ylabel(r'$\Delta n(t)$', fontsize=LABEL_SIZE)
    ax_density.tick_params(labelsize=TICK_SIZE)
    ax_density.set_ylim((-0.85, 2.0))
    ax_density.set_xlim(600, 1400)
    ax_density.axvline(x=warmup_long, color='black', linestyle='dotted',
                      alpha=0.8, linewidth=2, label=r'$t_{\mathrm{warm-up}}$')
    ax_density.legend(loc='lower left', ncols=2, labelspacing=0.05,
                     framealpha=0.6, markerscale=0.1, handlelength=1.5,
                     columnspacing=1.0, handletextpad=0.2,
                     borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
    
    # Add panel labels
    ax_input.text(-0.1, 1.02, '(a)', transform=ax_input.transAxes,
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax_density.text(-0.1, 1.02, '(b)', transform=ax_density.transAxes,
                   fontsize=TITLE_SIZE, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92,
                       wspace=0.2, hspace=0.3)
    
    # Save figure
    plt.savefig('new_plots/Fig6.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 6 saved!")


# ============================================================================
# FIGURE 7: DRIVING FREQUENCY SWEEP (CONTOUR PLOTS)
# ============================================================================

def generate_figure_7():
    """
    Generate Figure 7 showing density evolution as a function of time and driving frequency.
    Creates contour plots comparing:
    (a) Exact test data
    (b) Multi-input ESN predictions (test)
    (c) Single-input ESN predictions (test)
    (d) Single-input ESN predictions (train)
    """
    print("Generating Figure 7...")
    
    # Load results
    results_multi = load_results('test3_augmented_results.json')
    results_single = load_results('test3_direct_results.json')
    
    # Extract predictions
    y_pred_multi_test = results_multi['test']['y_pred_all'][:,:,0]
    y_pred_single_test = results_single['test']['y_pred_all'][:,:,0]
    y_pred_single_train = results_single['train']['y_pred_all'][:,:,0]
    print(y_pred_multi_test.shape)
    # Load test data
    data_arrays = json_file_to_arrays('./test3_test_data.json', observables=True)
    X, Y, Z, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, omega1_t, omega2_t, omega3_t = data_arrays
    Y_exact = 2 * (Y - 1)
    
    # Calculate driving frequencies (interpolated between training points)
    num_sys_train = 201
    num_sys_test = 200
    omega_dr_train = np.linspace(0.965, 1.065, num_sys_train)
    omega_dr_test = np.zeros(num_sys_test)
    for i in range(num_sys_test):
        omega_dr_test[i] = (omega_dr_train[i] + omega_dr_train[i+1]) / 2
    
    # Calculate reference frequencies at detuning = 0.04
    eigenvalues = np.linalg.eigvalsh(H_(0.04, 0.05))
    omega_01_ref = eigenvalues[1] - eigenvalues[0]
    omega_02_ref = eigenvalues[2] - eigenvalues[0]
    omega_12_ref = eigenvalues[2] - eigenvalues[1]
    
    # Print statistics
    print('Multi-input ESN:')
    print(f'  Max RMSE (train): {2*results_multi["train"]["max_rmse"]:.6f}')
    print(f'  Max RMSE (test):  {2*results_multi["test"]["max_rmse"]:.6f}')
    print(f'  Avg RMSE (train): {2*results_multi["train"]["avg_rmse"]:.6f}')
    print(f'  Avg RMSE (test):  {2*results_multi["test"]["avg_rmse"]:.6f}')
    print('Single-input ESN:')
    print(f'  Max RMSE (train): {2*results_single["train"]["max_rmse"]:.6f}')
    print(f'  Max RMSE (test):  {2*results_single["test"]["max_rmse"]:.6f}')
    print(f'  Avg RMSE (train): {2*results_single["train"]["avg_rmse"]:.6f}')
    print(f'  Avg RMSE (test):  {2*results_single["test"]["avg_rmse"]:.6f}')
    
    # Set up time and frequency arrays
    tmax = 650
    n_timesteps = int(tmax / dt)
    t_array = np.linspace(0, tmax - dt, n_timesteps)
    
    # Create meshgrids for contour plots
    time_mesh_test, freq_mesh_test = np.meshgrid(t_array, omega_dr_test)
    time_mesh_train, freq_mesh_train = np.meshgrid(t_array, omega_dr_train)
    
    # Prepare density data (clip to plotting range and truncate time)
    density_exact = np.clip(Y_exact[:, :n_timesteps], -2, 2)
    density_multi_test = np.clip(y_pred_multi_test[:, :n_timesteps], -2, 2)
    density_single_test = np.clip(y_pred_single_test[:, :n_timesteps], -2, 2)
    density_single_train = np.clip(y_pred_single_train[:, :n_timesteps], -2, 2)
    
    # Contour levels
    density_levels = np.linspace(-2, 2, 101)
    
    # Create figure
    scale = 0.8
    fig, axes = plt.subplots(3, 1, figsize=(scale*11, scale*11))
    ax_exact, ax_multi, ax_single_test = axes
    
    # Increase font sizes for this figure
    TITLE_SIZE_FIG7 = 15
    LABEL_SIZE_FIG7 = 15
    TICK_SIZE_FIG7 = 15
    LEGEND_SIZE_FIG7 = 15
    
    # -------------------------------------------------------------------------
    # Panel (a): Exact test data
    # -------------------------------------------------------------------------
    contour_exact = ax_exact.contourf(time_mesh_test, freq_mesh_test, density_exact,
                                     levels=density_levels, cmap='viridis', vmin=-2, vmax=2)
    ax_exact.set_xlabel(r'$t$', fontsize=LABEL_SIZE_FIG7, labelpad=0.1)
    ax_exact.set_ylabel(r'$\omega_{\mathrm{dr}}$', fontsize=LABEL_SIZE_FIG7)
    ax_exact.set_title(r'$\Delta n^{\mathrm{Exact}} (t)$', fontsize=TITLE_SIZE_FIG7)
    ax_exact.tick_params(labelsize=TICK_SIZE_FIG7)
    
    # Add reference lines
    ax_exact.axvline(x=warmup_time, color='white', linestyle='--',
                    alpha=0.8, linewidth=2, label=r'$t_{\mathrm{warm-up}}$')
    ax_exact.axhline(y=omega_01_ref, color='red', linestyle='-',
                    alpha=0.8, linewidth=2, label=r'$\omega_{0,1}[\Delta v = 0.04]$')
    ax_exact.axhline(y=omega_02_ref, color='orange', linestyle='-',
                    alpha=0.8, linewidth=2, label=r'$\omega_{0,2}[\Delta v = 0.04]$')
    ax_exact.legend(loc='center left', fontsize=LEGEND_SIZE_FIG7)
    
    # -------------------------------------------------------------------------
    # Panel (b): Multi-input ESN test predictions
    # -------------------------------------------------------------------------
    contour_multi = ax_multi.contourf(time_mesh_test, freq_mesh_test, density_multi_test,
                                     levels=density_levels, cmap='viridis', vmin=-2, vmax=2)
    ax_multi.set_xlabel(r'$t$', fontsize=LABEL_SIZE_FIG7, labelpad=0.1)
    ax_multi.set_ylabel(r'$\omega_{\mathrm{dr}}$', fontsize=LABEL_SIZE_FIG7)
    ax_multi.set_title(r'$\Delta n^{\mathrm{ESN}}_{\omega} (t)$', fontsize=TITLE_SIZE_FIG7)
    ax_multi.tick_params(labelsize=TICK_SIZE_FIG7)
    
    # Add reference lines
    ax_multi.axvline(x=warmup_time, color='white', linestyle='--', alpha=0.8, linewidth=2)
    ax_multi.axhline(y=omega_01_ref, color='red', linestyle='-', alpha=0.8, linewidth=2)
    ax_multi.axhline(y=omega_02_ref, color='orange', linestyle='-', alpha=0.8, linewidth=2)
    
    # -------------------------------------------------------------------------
    # Panel (c): Single-input ESN test predictions
    # -------------------------------------------------------------------------
    contour_single_test = ax_single_test.contourf(time_mesh_test, freq_mesh_test, density_single_test,
                                                  levels=density_levels, cmap='viridis', vmin=-2, vmax=2)
    ax_single_test.set_xlabel(r'$t$', fontsize=LABEL_SIZE_FIG7, labelpad=0.1)
    ax_single_test.set_ylabel(r'$\omega_{\mathrm{dr}}$', fontsize=LABEL_SIZE_FIG7)
    ax_single_test.set_title(r'$\Delta n^{\mathrm{ESN}} (t)$', fontsize=TITLE_SIZE_FIG7)
    ax_single_test.tick_params(labelsize=TICK_SIZE_FIG7)
    
    # Add reference lines
    ax_single_test.axvline(x=warmup_time, color='white', linestyle='--', alpha=0.8, linewidth=2)
    ax_single_test.axhline(y=omega_01_ref, color='red', linestyle='-', alpha=0.8, linewidth=2)
    ax_single_test.axhline(y=omega_02_ref, color='orange', linestyle='-', alpha=0.8, linewidth=2)
    
    # -------------------------------------------------------------------------
    # Panel (d): Single-input ESN train predictions
    # -------------------------------------------------------------------------
    '''contour_single_train = ax_single_train.contourf(time_mesh_train, freq_mesh_train, density_single_train,
                                                    levels=density_levels, cmap='viridis', vmin=-2, vmax=2)
    ax_single_train.set_xlabel(r'$t$', fontsize=LABEL_SIZE_FIG7, labelpad=0.1)
    ax_single_train.set_ylabel(r'$\omega_{\mathrm{dr}}$', fontsize=LABEL_SIZE_FIG7)
    ax_single_train.set_title(r'Train data $\Delta n^{\mathrm{ESN}} (t)$', fontsize=TITLE_SIZE_FIG7)
    ax_single_train.tick_params(labelsize=TICK_SIZE_FIG7)
    
    # Add reference lines
    ax_single_train.axvline(x=warmup_time, color='white', linestyle='--', alpha=0.8, linewidth=2)
    ax_single_train.axhline(y=omega_01_ref, color='red', linestyle='-', alpha=0.8, linewidth=2)
    ax_single_train.axhline(y=omega_02_ref, color='orange', linestyle='-', alpha=0.8, linewidth=2)
    
    # Add colorbars
    for ax, contour in zip(axes, [contour_exact, contour_multi, contour_single_test, contour_single_train]):'''
    for ax, contour in zip(axes, [contour_exact, contour_multi, contour_single_test]):
        cbar = fig.colorbar(contour, ax=ax, pad=0.01, fraction=0.05, ticks=[-2, 0, 2])
        cbar.set_label(r'$\Delta n$', fontsize=LABEL_SIZE_FIG7, rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=TICK_SIZE_FIG7)
    
    # Add panel labels
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for ax, label in zip(axes, panel_labels):
        ax.text(0.0, 1.02, label, transform=ax.transAxes,
               fontsize=TITLE_SIZE_FIG7, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92,
                       wspace=0.12, hspace=0.5)
    
    # Save figure
    plt.savefig('new_plots/Fig7.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 7 saved!")


# ============================================================================
# FIGURE 8: FOUR-SITE LATTICE DYNAMICS (CONTOUR PLOTS)
# ============================================================================

'''def generate_figure_8():
    """
    Generate Figure 8 showing occupation density on a 4-site lattice as a function
    of time and potential. Creates 4x3 grid of contour plots showing:
    - Rows: Each lattice site (1-4)
    - Columns: Exact, Single-input ESN, Multi-input ESN
    """
    print("Generating Figure 8...")
    
    # Load test data
    with open('test4_test_data.json', 'r') as f:
        data = json.load(f)
    
    # Extract arrays
    density_exact = np.array(data['n_array'])  # Shape: (n_systems, n_timesteps, 4 sites)
    potential_array = np.array(data['v_array'])[:, -1, 1]  # Potential for site 2 at final time
    tmax = data['tmax']
    dt = data['dt']
    n_timesteps = int(tmax / dt)
    t_array = np.linspace(0, tmax, n_timesteps)
    
    print(f"Exact data shape: {density_exact.shape}")
    
    # Load predictions
    with open('test4_single_results.json', 'r') as f:
        results_single = json.load(f)
    density_single = np.array(results_single['test']['y_pred_all'])
    print(f"Single-input ESN shape: {density_single.shape}")
    
    with open('test4_multi_results.json', 'r') as f:
        results_multi = json.load(f)
    density_multi = np.array(results_multi['test']['y_pred_all'])
    print(f"Multi-input ESN shape: {density_multi.shape}")
    
    # Clip densities to physical range [0, 1]
    density_exact = np.clip(density_exact, 0, 1)
    density_single = np.clip(density_single, 0, 1)
    density_multi = np.clip(density_multi, 0, 1)
    
    # Create meshgrid for contour plots
    time_mesh, potential_mesh = np.meshgrid(t_array, potential_array)
    
    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=(16, 7))
    
    # Font sizes for this figure
    fontsize = 16
    title_fontsize = 16
    
    # Panel labels for each subplot
    panel_labels = [['(a)', '(b)', '(c)', '(d)'],
                   ['(e)', '(f)', '(g)', '(h)'],
                   ['(i)', '(j)', '(k)', '(l)']]
    
    # Column titles and data
    column_titles = [r'$\bar{n}^{\mathrm{Exact}}$',
                    r'$\bar{n}^{\mathrm{ESN}}$',
                    r'$\bar{n}^{\mathrm{ESN}}_{\omega}$']
    density_arrays = [density_exact, density_single, density_multi]
    
    # Create contour plots
    for site_idx in range(4):  # 4 lattice sites
        for col_idx, (density_data, col_title) in enumerate(zip(density_arrays, column_titles)):
            ax = axes[site_idx, col_idx]
            
            # Extract density for this site
            site_density = density_data[:, :, site_idx]
            
            # Create contour plot
            contour = ax.contourf(time_mesh, potential_mesh, site_density,
                                 levels=200, cmap='viridis', vmin=0, vmax=1)
            
            # Add panel label
            ax.text(0.05, 0.8, panel_labels[col_idx][site_idx],
                   transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Add warmup line (only for predictions, not exact)
            if col_idx > 0:
                ax.axvline(x=warmup_time, color='white', linestyle='--',
                          alpha=0.8, linewidth=2, label=r'$t_{\mathrm{warm-up}}$')
            
            # Add xlabel for bottom row
            if site_idx == 3:
                ax.set_xlabel('$t$', fontsize=fontsize)
                ax.set_xticks([0, 50, 100, 150, 200, 250])
            
            # Add title for top row
            if site_idx == 0:
                ax.set_title(col_title, fontsize=title_fontsize, fontweight='bold')
                if col_idx == 1:
                    ax.legend(loc='lower left', fontsize=fontsize)
            
            # Add site label and ylabel for first column
            if col_idx == 0:
                ax.text(-0.3, 0.5, f'Site {site_idx + 1}',
                       transform=ax.transAxes, rotation=90,
                       verticalalignment='center', fontsize=fontsize)
                ax.set_ylabel(r'$v_2 (t>t_{\mathrm{on}})$', fontsize=fontsize)
            
            # Set axis limits
            ax.set_xlim(0, tmax)
            ax.set_ylim(potential_mesh.min(), potential_mesh.max())
            
            # Remove tick labels except for edges
            if site_idx != 3:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='x', labelsize=fontsize-2)
            
            if col_idx != 0:
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis='y', labelsize=fontsize-2)
    
    # Add single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    cbar.set_label('$n_i$', rotation=270, labelpad=20, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize-2)
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.9, bottom=0.08, top=0.92,
                       wspace=0.05, hspace=0.05)
    
    # Save figure
    plt.savefig('new_plots/Fig8.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 8 saved!")'''

def generate_figure_8():
    """
    Generate Figure 8 showing occupation density on a 4-site lattice as a function
    of time and potential. Creates 4x2 grid of contour plots showing:
    - Rows: Each lattice site (1-4)
    - Columns: Exact, Single-input ESN
    """
    print("Generating Figure 8...")
    
    # Load test data
    with open('test4_test_data_1200.json', 'r') as f:
        data = json.load(f)
    
    # Extract arrays
    density_exact = np.array(data['n_array'])  # Shape: (n_systems, n_timesteps, 4 sites)
    potential_array = np.array(data['v_array'])[:, -1, 1]  # Potential for site 2 at final time
    tmax = data['tmax']
    dt = data['dt']
    n_timesteps = int(tmax / dt)
    t_array = np.linspace(0, tmax, n_timesteps)
    
    print(f"Exact data shape: {density_exact.shape}")
    
    # Load predictions
    with open('test4_direct_1200.json', 'r') as f:
        results_single = json.load(f)
    density_single = np.array(results_single['test']['y_pred_all'])
    print(f"Single-input ESN shape: {density_single.shape}")
    
    idx = 40
    exact_ex = density_exact[idx,:,:]
    single_ex = density_single[idx,:,:]
    
    # Clip densities to physical range [0, 1]
    density_exact = np.clip(density_exact, 0, 1)
    density_single = np.clip(density_single, 0, 1)
    
    # Create meshgrid for contour plots
    time_mesh, potential_mesh = np.meshgrid(t_array, potential_array)
    
    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(11, 5))
    
    # Font sizes for this figure
    fontsize = 16
    title_fontsize = 16
    
    # Panel labels for each subplot
    panel_labels = [['(a)', '(b)', '(c)', '(d)'],
                   ['(e)', '(f)', '(g)', '(h)']]
    
    # Column titles and data
    column_titles = [r'$\bar{n}^{\mathrm{Exact}}$',
                    r'$\bar{n}^{\mathrm{ESN}}$']
    density_arrays = [density_exact, density_single]
    
    # Create contour plots
    for site_idx in range(4):  # 4 lattice sites
        for col_idx, (density_data, col_title) in enumerate(zip(density_arrays, column_titles)):
            ax = axes[site_idx, col_idx]
            
            # Extract density for this site
            site_density = density_data[:, :, site_idx]
            
            # Create contour plot
            contour = ax.contourf(time_mesh, potential_mesh, site_density,
                                 levels=200, cmap='viridis', vmin=0, vmax=1)
            
            # Add panel label
            ax.text(0.05, 0.8, panel_labels[col_idx][site_idx],
                   transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Add warmup line (only for predictions, not exact)
            if col_idx > 0:
                ax.axvline(x=warmup_time, color='white', linestyle='--',
                          alpha=0.8, linewidth=2, label=r'$t_{\mathrm{warm-up}}$')
            
            # Add xlabel for bottom row
            if site_idx == 3:
                ax.set_xlabel('$t$', fontsize=fontsize)
                ax.set_xticks([0, 300,600,900,1200])
            
            # Add title for top row
            if site_idx == 0:
                ax.set_title(col_title, fontsize=title_fontsize, fontweight='bold')
                if col_idx == 1:
                    ax.legend(loc='lower left', fontsize=fontsize)
            
            # Add site label and ylabel for first column
            if col_idx == 0:
                ax.text(-0.3, 0.5, f'Site {site_idx + 1}',
                       transform=ax.transAxes, rotation=90,
                       verticalalignment='center', fontsize=fontsize)
                ax.set_ylabel(r'$v_2 (t>t_{\mathrm{on}})$', fontsize=fontsize)
            
            # Set axis limits
            ax.set_xlim(0, tmax)
            ax.set_ylim(potential_mesh.min(), potential_mesh.max())
            
            # Remove tick labels except for edges
            if site_idx != 3:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='x', labelsize=fontsize-2)
            
            if col_idx != 0:
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis='y', labelsize=fontsize-2)
    
    # Add single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    cbar.set_label('$n_i$', rotation=270, labelpad=20, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize-2)
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.9, bottom=0.08, top=0.92,
                       wspace=0.05, hspace=0.05)
    
    # Save figure
    plt.savefig('new_plots/Fig8.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 8 saved!")
    
def generate_figure_8_2():
    """
    Generate Figure 8 showing occupation density on a 4-site lattice as a function
    of time and potential. Creates 6x2 grid of contour plots showing:
    - Rows 1-4: Contour plots for each lattice site (1-4)
      - Columns: Exact, Single-input ESN
    - Rows 5-6: Line plots comparing exact vs ESN for each site
    """
    print("Generating Figure 8...")
    
    # Load test data
    with open('test4_test_data_1200.json', 'r') as f:
        data = json.load(f)
    
    # Extract arrays
    density_exact = np.array(data['n_array'])  # Shape: (n_systems, n_timesteps, 4 sites)
    potential_array = np.array(data['v_array'])[:, -1, 1]  # Potential for site 2 at final time
    tmax = data['tmax']
    dt = data['dt']
    n_timesteps = int(tmax / dt)
    t_array = np.linspace(0, tmax, n_timesteps)
    
    print(f"Exact data shape: {density_exact.shape}")
    
    # Load predictions
    with open('test4_direct_1200.json', 'r') as f:
        results_single = json.load(f)
    density_single = np.array(results_single['test']['y_pred_all'])
    print(f"Single-input ESN shape: {density_single.shape}")
    
    idx = 40
    exact_ex = density_exact[idx,:,:]
    single_ex = density_single[idx,:,:]
    
    # Clip densities to physical range [0, 1]
    density_exact = np.clip(density_exact, 0, 1)
    density_single = np.clip(density_single, 0, 1)
    
    # Create meshgrid for contour plots
    time_mesh, potential_mesh = np.meshgrid(t_array, potential_array)
    
    # Create figure with 6x2 grid
    fig, axes = plt.subplots(6, 2, figsize=(11, 8))
    
    # Font sizes for this figure
    fontsize = 16
    title_fontsize = 16
    
    # Panel labels for contour subplots
    panel_labels = [['(a)', '(b)', '(c)', '(d)'],
                   ['(e)', '(f)', '(g)', '(h)']]
    
    # Panel labels for line plot subplots
    line_panel_labels = ['(i)', '(j)', '(k)', '(l)']
    
    # Column titles and data
    column_titles = [r'$\bar{n}^{\mathrm{Exact}}$',
                    r'$\bar{n}^{\mathrm{ESN}}$']
    density_arrays = [density_exact, density_single]
    
    # Create contour plots (rows 0-3)
    for site_idx in range(4):  # 4 lattice sites
        for col_idx, (density_data, col_title) in enumerate(zip(density_arrays, column_titles)):
            ax = axes[site_idx, col_idx]
            
            # Extract density for this site
            site_density = density_data[:, :, site_idx]
            
            # Create contour plot
            contour = ax.contourf(time_mesh, potential_mesh, site_density,
                                 levels=200, cmap='viridis', vmin=0, vmax=1)
            
            # Add panel label
            ax.text(0.05, 0.8, panel_labels[col_idx][site_idx],
                   transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Add warmup line (only for predictions, not exact)
            if col_idx > 0:
                ax.axvline(x=warmup_time, color='white', linestyle='--',
                          alpha=0.8, linewidth=2, label=r'$t_{\mathrm{warm-up}}$')
            
            # Add title for top row
            if site_idx == 0:
                ax.set_title(col_title, fontsize=title_fontsize, fontweight='bold')
                if col_idx == 1:
                    ax.legend(loc='lower left', fontsize=fontsize)
            
            # Add site label and ylabel for first column
            if col_idx == 0:
                ax.text(-0.3, 0.5, f'Site {site_idx + 1}',
                       transform=ax.transAxes, rotation=90,
                       verticalalignment='center', fontsize=fontsize)
                ax.set_ylabel(r'$v_2 (t>t_{\mathrm{on}})$', fontsize=fontsize)
            
            # Set axis limits
            ax.set_xlim(0, tmax)
            ax.set_ylim(potential_mesh.min(), potential_mesh.max())
            
            # Remove tick labels for contour plots (no x labels since line plots below)
            ax.set_xticklabels([])
            
            if col_idx != 0:
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis='y', labelsize=fontsize-2)
    
    # Create line plots (rows 4-5, 2x2 grid for 4 sites)
    for site_idx in range(4):
        row_idx = 4 + site_idx // 2  # Row 4 for sites 0,1; Row 5 for sites 2,3
        col_idx = site_idx % 2        # Column 0 for sites 0,2; Column 1 for sites 1,3
        
        ax = axes[row_idx, col_idx]
        
        # Plot exact and ESN predictions
        ax.plot(t_array, exact_ex[:, site_idx], 'b-', linewidth=1.5, label='Exact')
        ax.plot(t_array, single_ex[:, site_idx], 'r--', linewidth=1.5, label='ESN')
        
        # Add warmup line
        ax.axvline(x=warmup_time, color='gray', linestyle='--',
                  alpha=0.8, linewidth=1.5, label=r'$t_{\mathrm{warm-up}}$')
        
        # Add panel label
        ax.text(0.05, 0.85, line_panel_labels[site_idx],
               transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Add site label
        ax.text(0.95, 0.85, f'Site {site_idx + 1}',
               transform=ax.transAxes, fontsize=fontsize-2,
               ha='right', bbox=dict(facecolor='white', alpha=0.7))
        
        # Add legend for first line plot only
        if site_idx == 0:
            ax.legend(loc='lower right', fontsize=fontsize-4)
        
        # Labels
        if row_idx == 5:  # Bottom row
            ax.set_xlabel('$t$', fontsize=fontsize)
            ax.set_xticks([0, 300, 600, 900, 1200])
            ax.tick_params(axis='x', labelsize=fontsize-2)
        else:
            ax.set_xticklabels([])
        
        if col_idx == 0:  # Left column
            ax.set_ylabel('$n_i$', fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize-2)
        else:
            ax.set_yticklabels([])
        
        # Set axis limits
        ax.set_xlim(0, tmax)
        ax.set_ylim(0.2, 0.8)
    
    # Add single colorbar for contour subplots
    cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.55])
    cbar = fig.colorbar(contour, cax=cbar_ax, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    cbar.set_label('$n_i$', rotation=270, labelpad=20, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize-2)
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.9, bottom=0.06, top=0.94,
                       wspace=0.05, hspace=0.08)
    

    
    # Save figure
    plt.savefig('new_plots/Fig8.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 8 saved!")
    
def generate_figure_8_3():
    """
    Generate Figure 8 showing occupation density on a 4-site lattice as a function
    of time and potential. Creates 8x2 grid of plots showing:
    - Rows 1-4: Contour plots for each lattice site (1-4)
      - Columns: Exact, Single-input ESN
    - Rows 5-6: Line plots comparing exact vs ESN for each site
    - Rows 7-8: FFT plots comparing exact vs ESN for each site
    """
    print("Generating Figure 8...")
    
    # Load test data
    with open('test4_test_data_1200.json', 'r') as f:
        data = json.load(f)
    
    # Extract arrays
    density_exact = np.array(data['n_array'])  # Shape: (n_systems, n_timesteps, 4 sites)
    potential_array = np.array(data['v_array'])[:, -1, 1]  # Potential for site 2 at final time
    tmax = data['tmax']
    dt = data['dt']
    n_timesteps = int(tmax / dt)
    t_array = np.linspace(0, tmax, n_timesteps)
    
    print(f"Exact data shape: {density_exact.shape}")
    
    # Load predictions
    with open('test4_direct_1200.json', 'r') as f:
        results_single = json.load(f)
    density_single = np.array(results_single['test']['y_pred_all'])
    print(f"Single-input ESN shape: {density_single.shape}")
    
    idx = 40
    exact_ex = density_exact[idx,:,:]
    single_ex = density_single[idx,:,:]
    
    # Clip densities to physical range [0, 1]
    density_exact = np.clip(density_exact, 0, 1)
    density_single = np.clip(density_single, 0, 1)
    
    # Create meshgrid for contour plots
    time_mesh, potential_mesh = np.meshgrid(t_array, potential_array)
    
    # Compute FFTs
    fft_start_idx = int(200 / dt)
    fft_exact = np.fft.rfft(exact_ex[fft_start_idx:, :], axis=0, norm="forward")
    fft_single = np.fft.rfft(single_ex[fft_start_idx:, :], axis=0, norm="forward")
    fft_freq = np.fft.rfftfreq(len(exact_ex[fft_start_idx:, 0]), dt)
    omega = 2 * np.pi * fft_freq
    
    # Create figure with 8x2 grid
    fig, axes = plt.subplots(8, 2, figsize=(6.25, 11))
    
    # Font sizes for this figure
    fontsize = 16
    title_fontsize = 16
    
    # Panel labels for contour subplots
    panel_labels = [['(a)', '(b)', '(c)', '(d)'],
                   ['(e)', '(f)', '(g)', '(h)']]
    
    # Panel labels for line plot subplots
    line_panel_labels = ['(i)', '(j)', '(k)', '(l)']
    
    # Panel labels for FFT subplots
    fft_panel_labels = ['(m)', '(n)', '(o)', '(p)']
    
    # Column titles and data
    column_titles = [r'$\bar{n}^{\mathrm{Exact}}$',
                    r'$\bar{n}^{\mathrm{ESN}}$']
    density_arrays = [density_exact, density_single]
    
    # Create contour plots (rows 0-3)
    for site_idx in range(4):  # 4 lattice sites
        for col_idx, (density_data, col_title) in enumerate(zip(density_arrays, column_titles)):
            ax = axes[site_idx, col_idx]
            
            # Extract density for this site
            site_density = density_data[:, :, site_idx]
            
            # Create contour plot
            contour = ax.contourf(time_mesh, potential_mesh, site_density,
                                 levels=200, cmap='viridis', vmin=0, vmax=1)
            
            # Add panel label
            ax.text(0.05, 0.8, panel_labels[col_idx][site_idx],
                   transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Add warmup line (only for predictions, not exact)
            if col_idx > 0:
                ax.axvline(x=warmup_time, color='white', linestyle='--',
                          alpha=0.8, linewidth=2, label=r'$t_{\mathrm{warm-up}}$')
            
            # Add title for top row
            if site_idx == 0:
                ax.set_title(col_title, fontsize=title_fontsize, fontweight='bold')
                if col_idx == 1:
                    ax.legend(loc='lower left', fontsize=fontsize)
            
            # Add site label and ylabel for first column
            if col_idx == 0:
                ax.text(-0.3, 0.5, f'Site {site_idx + 1}',
                       transform=ax.transAxes, rotation=90,
                       verticalalignment='center', fontsize=fontsize)
                ax.set_ylabel(r'$v_2 (t>t_{\mathrm{on}})$', fontsize=fontsize)
            
            # Set axis limits
            ax.set_xlim(0, tmax)
            ax.set_ylim(potential_mesh.min(), potential_mesh.max())
            
            # Remove tick labels for contour plots
            ax.set_xticklabels([])
            ax.set_yticks([0.01,0.05,0.09])
            
            if col_idx != 0:
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis='y', labelsize=fontsize-2)
    
    # Create line plots (rows 4-5, 2x2 grid for 4 sites)
    for site_idx in range(4):
        row_idx = 4 + site_idx // 2  # Row 4 for sites 0,1; Row 5 for sites 2,3
        col_idx = site_idx % 2        # Column 0 for sites 0,2; Column 1 for sites 1,3
        
        ax = axes[row_idx, col_idx]
        
        # Plot exact and ESN predictions
        ax.plot(t_array, exact_ex[:, site_idx], 'b-', linewidth=1.5, label='Exact')
        ax.plot(t_array, single_ex[:, site_idx], 'r--', linewidth=1.5, label='ESN')
        
        # Add warmup line
        ax.axvline(x=warmup_time, color='gray', linestyle='--',
                  alpha=0.8, linewidth=1.5, label=r'$t_{\mathrm{warm-up}}$')
        
        # Add panel label
        ax.text(0.05, 0.85, line_panel_labels[site_idx],
               transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Add site label
        ax.text(0.95, 0.85, f'Site {site_idx + 1}',
               transform=ax.transAxes, fontsize=fontsize-2,
               ha='right', bbox=dict(facecolor='white', alpha=0.7))
        
        # Add legend for first line plot only
        if site_idx == 0:
            ax.legend(loc='lower right', ncols=2, labelspacing=0.05, 
                             framealpha=0.6, markerscale=0.1, handlelength=1.5,
                             columnspacing=1.0, handletextpad=0.2, 
                             borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
        
        # Labels - no x labels since FFT plots are below
        ax.set_xticklabels([])
        
        if col_idx == 0:  # Left column
            ax.set_ylabel('$n_i$', fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize-2)
        else:
            ax.set_yticklabels([])
        
        # Set axis limits
        ax.set_xlim(0, tmax)
        ax.set_ylim(0.2, 0.8)
    
    # Create FFT plots (rows 6-7, 2x2 grid for 4 sites)
    v_sites = np.array(data['v_array'])[idx, -1, :]
    T_hop=0.05
    L=4
    H0 = H__(v_sites, T_hop, L)
    vals, vecs = np.linalg.eigh(H0)
    transitions = np.array([vals[3] - vals[0],
                            vals[3] - vals[1],
                            vals[3] - vals[2],
                            vals[2] - vals[0],
                            vals[2] - vals[1],
                            vals[1] - vals[0]])
    transition_labels = [r'$\omega_{0,3}$',
                         r'$\omega_{1,3}$',
                         r'$\omega_{2,3}$',
                         r'$\omega_{0,2}$',
                         r'$\omega_{1,2}$',
                         r'$\omega_{0,1}$',]
    # Define colors for transition lines
    transition_colors = ['green', 'orange', 'purple', 'cyan', 'magenta', 'brown']
    
    for site_idx in range(4):
        row_idx = 6 + site_idx // 2  # Row 6 for sites 0,1; Row 7 for sites 2,3
        col_idx = site_idx % 2        # Column 0 for sites 0,2; Column 1 for sites 1,3
        
        ax = axes[row_idx, col_idx]
        
        # Plot FFT of exact and ESN predictions
        ax.plot(omega, np.abs(fft_exact[:, site_idx]), 'b-', linewidth=1.5, label='Exact')
        ax.plot(omega, np.abs(fft_single[:, site_idx]), 'r--', linewidth=1.5, label='ESN')
        
        # Add transition frequency lines
        for i in range(6):
            ax.axvline(transitions[i], color=transition_colors[i], linestyle='--', 
                      linewidth=1.2, alpha=0.8, label=transition_labels[i])
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1E-2, 0.6)
        
        # Add panel label
        ax.text(0.05, 0.85, fft_panel_labels[site_idx],
               transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Add site label
        ax.text(0.95, 0.85, f'Site {site_idx + 1}',
               transform=ax.transAxes, fontsize=fontsize-2,
               ha='right', bbox=dict(facecolor='white', alpha=0.7))
        
        # Add legend for first FFT plot only (includes all lines)
        if site_idx == 0:
            ax.legend(loc='lower left', ncols=2, labelspacing=0.05, 
                             framealpha=0.6, markerscale=0.1, handlelength=1.5,
                             columnspacing=1.0, handletextpad=0.2, 
                             borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
        
        # Labels
        if row_idx == 7:  # Bottom row
            ax.set_xlabel(r'$\omega$', fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize-2)
        else:
            ax.set_xticklabels([])
        
        if col_idx == 0:  # Left column
            ax.set_ylabel(r'$|\tilde{n}_i(\omega)|$', fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize-2)
        else:
            ax.set_yticklabels([])
            
    
    # Add single colorbar for contour subplots
    cbar_ax = fig.add_axes([0.92, 0.52, 0.02, 0.4])
    cbar = fig.colorbar(contour, cax=cbar_ax, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    cbar.set_label('$n_i$', rotation=270, labelpad=20, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize-2)
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.9, bottom=0.05, top=0.95,
                       wspace=0.05, hspace=0.08)
    
    # Save figure
    plt.savefig('new_plots/Fig8.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 8 saved!")
    
def generate_figure_8_4():
    """
    Generate Figure 8 showing occupation density on a 4-site lattice as a function
    of time and potential. Creates 8x2 grid of plots showing:
    - Rows 1-4: Contour plots for each lattice site (1-4)
      - Columns: Exact, Single-input ESN
    - Rows 5-6: Line plots comparing exact vs ESN for each site
    - Rows 7-8: FFT plots comparing exact vs ESN for each site
    """
    print("Generating Figure 8...")
    
    # Load test data
    with open('test4_test_data_1200.json', 'r') as f:
        data = json.load(f)
    
    # Extract arrays
    density_exact = np.array(data['n_array'])  # Shape: (n_systems, n_timesteps, 4 sites)
    potential_array = np.array(data['v_array'])[:, -1, 1]  # Potential for site 2 at final time
    tmax = data['tmax']
    dt = data['dt']
    n_timesteps = int(tmax / dt)
    t_array = np.linspace(0, tmax, n_timesteps)
    
    print(f"Exact data shape: {density_exact.shape}")
    
    # Load predictions
    with open('test4_direct_1200.json', 'r') as f:
        results_single = json.load(f)
    density_single = np.array(results_single['test']['y_pred_all'])
    print(f"Single-input ESN shape: {density_single.shape}")
    
    idx = 40
    exact_ex = density_exact[idx,:,:]
    single_ex = density_single[idx,:,:]
    
    # Clip densities to physical range [0, 1]
    density_exact = np.clip(density_exact, 0, 1)
    density_single = np.clip(density_single, 0, 1)
    
    # Create meshgrid for contour plots
    time_mesh, potential_mesh = np.meshgrid(t_array, potential_array)
    
    # Compute FFTs
    fft_start_idx = int(200 / dt)
    fft_exact = np.fft.rfft(exact_ex[fft_start_idx:, :], axis=0, norm="forward")
    fft_single = np.fft.rfft(single_ex[fft_start_idx:, :], axis=0, norm="forward")
    fft_freq = np.fft.rfftfreq(len(exact_ex[fft_start_idx:, 0]), dt)
    omega = 2 * np.pi * fft_freq
    
    # Compute transition frequencies
    v_sites = np.array(data['v_array'])[idx, -1, :]
    T_hop = 0.05
    L = 4
    H0 = H__(v_sites, T_hop, L)
    vals, vecs = np.linalg.eigh(H0)
    transitions = np.array([vals[3] - vals[0],
                            vals[3] - vals[1],
                            vals[3] - vals[2],
                            vals[2] - vals[0],
                            vals[2] - vals[1],
                            vals[1] - vals[0]])
    
    transition_labels = [r'$\omega_{0,3}$',
                         r'$\omega_{1,3}$',
                         r'$\omega_{2,3}$',
                         r'$\omega_{0,2}$',
                         r'$\omega_{1,2}$',
                         r'$\omega_{0,1}$']
    transition_colors = ['green', 'orange', 'purple', 'cyan', 'magenta', 'brown']
    
    for i in range(6):
        print(transition_labels[i], transitions[i])
    
    # Create figure with 8x2 grid - increased height for spacing
    fig, axes = plt.subplots(8, 2, figsize=(8, 16))
    
    # Font sizes for this figure
    fontsize = 16
    title_fontsize = 16
    
    # Panel labels for contour subplots
    panel_labels = [['(a)', '(b)', '(c)', '(d)'],
                   ['(e)', '(f)', '(g)', '(h)']]
    
    # Panel labels for line plot subplots
    line_panel_labels = ['(i)', '(j)', '(k)', '(l)']
    
    # Panel labels for FFT subplots
    fft_panel_labels = ['(m)', '(n)', '(o)', '(p)']
    
    # Column titles and data
    column_titles = [r'$\bar{n}^{\mathrm{Exact}}$',
                    r'$\bar{n}^{\mathrm{ESN}}$']
    density_arrays = [density_exact, density_single]
    
    # Create contour plots (rows 0-3)
    for site_idx in range(4):  # 4 lattice sites
        for col_idx, (density_data, col_title) in enumerate(zip(density_arrays, column_titles)):
            ax = axes[site_idx, col_idx]
            
            # Extract density for this site
            site_density = density_data[:, :, site_idx]
            
            # Create contour plot
            contour = ax.contourf(time_mesh, potential_mesh, site_density,
                                 levels=200, cmap='viridis', vmin=0, vmax=1)
            
            # Add panel label
            ax.text(0.05, 0.8, panel_labels[col_idx][site_idx],
                   transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Add warmup line (only for predictions, not exact)
            if col_idx > 0:
                ax.axvline(x=warmup_time, color='white', linestyle='--',
                          alpha=0.8, linewidth=2, label=r'$t_{\mathrm{warm-up}}$')
            
            # Add title for top row
            if site_idx == 0:
                ax.set_title(col_title, fontsize=title_fontsize, fontweight='bold')
                if col_idx == 1:
                    ax.legend(loc='lower left', ncols=1, labelspacing=0.05, 
                                     framealpha=0.6, markerscale=0.1, handlelength=1.5,
                                     columnspacing=0.5, handletextpad=0.2, 
                                     borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
            
            # Add site label and ylabel for first column
            if col_idx == 0:
                ax.text(-0.3, 0.5, f'Site {site_idx + 1}',
                       transform=ax.transAxes, rotation=90,
                       verticalalignment='center', fontsize=fontsize)
                #ax.set_ylabel(r'$v_2 (t>t_{\mathrm{on}})$', fontsize=fontsize)
                ax.set_ylabel(r'$\epsilon$', fontsize=fontsize)
            
            # Set axis limits
            ax.set_xlim(0, tmax)
            ax.set_ylim(potential_mesh.min(), potential_mesh.max())
            
            # Add xticks and xlabel to all contour plots
            ax.set_xticks([0, 300, 600, 900])
            ax.set_xlabel('$t$', fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize-2)
            ax.set_yticks([0.01,0.05,0.09])
            
            if col_idx != 0:
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis='y', labelsize=fontsize-2)
    
    # Create line plots (rows 4-5, 2x2 grid for 4 sites)
    for site_idx in range(4):
        row_idx = 4 + site_idx // 2  # Row 4 for sites 0,1; Row 5 for sites 2,3
        col_idx = site_idx % 2        # Column 0 for sites 0,2; Column 1 for sites 1,3
        
        ax = axes[row_idx, col_idx]
        
        # Plot exact and ESN predictions
        ax.plot(t_array, exact_ex[:, site_idx], 'b-', linewidth=1.5, label='Exact')
        ax.plot(t_array, single_ex[:, site_idx], 'r--', linewidth=1.5, label='ESN')
        
        # Add warmup line
        ax.axvline(x=warmup_time, color='gray', linestyle='--',
                  alpha=0.8, linewidth=1.5, label=r'$t_{\mathrm{warm-up}}$')
        
        # Add panel label
        ax.text(0.05, 0.85, line_panel_labels[site_idx],
               transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Add site label
        ax.text(0.95, 0.85, f'Site {site_idx + 1}',
               transform=ax.transAxes, fontsize=fontsize-2,
               ha='right', bbox=dict(facecolor='white', alpha=0.7))
        
        # Add legend for first line plot only
        if site_idx == 0:
            ax.legend(loc='lower left', ncols=1, labelspacing=0.05, 
                             framealpha=0.6, markerscale=0.1, handlelength=1.5,
                             columnspacing=0.5, handletextpad=0.2, 
                             borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
        
        # Add xticks and xlabel to all line plots
        ax.set_xticks([0, 300, 600, 900])
        ax.set_xlabel('$t$', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize-2)
        
        if col_idx == 0:  # Left column
            ax.set_ylabel('$n_i$', fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize-2)
        else:
            ax.set_yticklabels([])
        
        # Set axis limits
        ax.set_xlim(0, tmax)
        ax.set_ylim(0.2, 0.8)
    
    # Create FFT plots (rows 6-7, 2x2 grid for 4 sites)
    for site_idx in range(4):
        row_idx = 6 + site_idx // 2  # Row 6 for sites 0,1; Row 7 for sites 2,3
        col_idx = site_idx % 2        # Column 0 for sites 0,2; Column 1 for sites 1,3
        
        ax = axes[row_idx, col_idx]
        
        # Plot FFT of exact and ESN predictions
        ax.plot(omega, np.abs(fft_exact[:, site_idx]), 'b-', linewidth=1.5, label='Exact')
        ax.plot(omega, np.abs(fft_single[:, site_idx]), 'r--', linewidth=1.5, label='ESN')
        
        # Add transition frequency lines
        for i in range(6):
            ax.axvline(transitions[i], color=transition_colors[i], linestyle='--',
                      linewidth=1.2, alpha=0.8, label=transition_labels[i])
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1E-2, 0.6)
        
        # Add panel label
        ax.text(0.05, 0.85, fft_panel_labels[site_idx],
               transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Add site label
        ax.text(0.95, 0.85, f'Site {site_idx + 1}',
               transform=ax.transAxes, fontsize=fontsize-2,
               ha='right', bbox=dict(facecolor='white', alpha=0.7))
        
        # Add legend for first FFT plot only
        if site_idx == 0:
            ax.legend(loc='lower left', ncols=3, labelspacing=0.05, 
                             framealpha=0.6, markerscale=0.1, handlelength=1.5,
                             columnspacing=0.5, handletextpad=0.2, 
                             borderaxespad=0.0, borderpad=0.0, fontsize=LEGEND_SIZE)
        
        # Labels
        
        ax.set_xlabel(r'$\omega$', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize-2)
        #else:
        #ax.set_xticklabels([])
        ax.set_xticks([1E-2,1E-1])        
        if col_idx == 0:  # Left column
            ax.set_ylabel(r'$|\tilde{n}_i(\omega)|$', fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize-2)
        else:
            ax.set_yticklabels([])
    
    # Add single colorbar for contour subplots
    cbar_ax = fig.add_axes([0.92, 0.58, 0.02, 0.35])
    cbar = fig.colorbar(contour, cax=cbar_ax, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    cbar.set_label('$n_i$', rotation=270, labelpad=20, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize-2)
    
    # Adjust layout with increased vertical spacing
    plt.subplots_adjust(left=0.08, right=0.9, bottom=0.04, top=0.96,
                       wspace=0.05, hspace=0.4)
    
    # Save figure
    plt.savefig('new_plots/Fig8.png', dpi=400, bbox_inches='tight')
    plt.close()
    print("Figure 8 saved!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution: Generate all figures for the manuscript.
    """
    print("\n" + "="*70)
    print("GENERATING ALL FIGURES FOR MANUSCRIPT")
    print("="*70 + "\n")
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('new_plots', exist_ok=True)
    
    # Generate each figure
    try:
        #generate_figure_3()
        #generate_figure_4()
        #generate_figure_5()
        #generate_figure_6()
        #generate_figure_7()
        generate_figure_8_4()
        
        print("\n" + "="*70)
        print("ALL FIGURES GENERATED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        raise