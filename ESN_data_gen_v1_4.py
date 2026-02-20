import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time
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

def create_frequency_interpolators(T, v_range=(-2, 2), n_points=10001):
    """
    Create interpolation functions to map potential to natural frequencies.
    
    Parameters
    ----------
    T : float
        Hopping amplitude
    v_range : tuple, optional
        Range of static potentials to sample (default: (-3, 3))
    n_points : int, optional
        Number of points for interpolation (default: 1000)
    
    Returns
    -------
    tuple
        (omega1_interp, omega2_interp, omega3_interp) interpolation functions
        where omega1 = ε₁ - ε₀, omega2 = ε₂ - ε₀, omega3 = ε₂ - ε₁
    """
    v_static_array = np.linspace(v_range[0], v_range[1], n_points)
    omega1_array = np.zeros(n_points)
    omega2_array = np.zeros(n_points)
    omega3_array = np.zeros(n_points)
    
    for i in range(n_points):
        dv = v_static_array[i]
        eigs = np.linalg.eigvalsh(H_(dv, T))
        omega1_array[i] = eigs[1] - eigs[0]
        omega2_array[i] = eigs[2] - eigs[0]
        omega3_array[i] = eigs[2] - eigs[1]
    kind = 'linear'
    # Create interpolation functions
    omega1_interp = interp1d(v_static_array, omega1_array, kind=kind, 
                            bounds_error=False, fill_value='extrapolate')
    omega2_interp = interp1d(v_static_array, omega2_array, kind=kind, 
                            bounds_error=False, fill_value='extrapolate')
    omega3_interp = interp1d(v_static_array, omega3_array, kind=kind, 
                            bounds_error=False, fill_value='extrapolate')
    
    return omega1_interp, omega2_interp, omega3_interp

def wf2d_(wf):
    dm = np.outer(np.conjugate(wf),wf)
    return 2*(dm[0,0] - dm[2,2]).real

def dictionary_data_to_arrays(dictionary, observables=False):
    if observables == False:
        num_sys = len(dictionary.keys())
        timesteps = len(dictionary['system_0']['density'])
        v_array = np.zeros((num_sys,timesteps))
        n_array = np.zeros((num_sys,timesteps))
        for i in range(num_sys):
            v_array[i] = np.array(dictionary['system_'+str(i)]['ext_potential'])
            n_array[i] = np.array(dictionary['system_'+str(i)]['density'])
        return v_array, n_array
    elif observables == True:
        num_sys = len(dictionary.keys())
        timesteps = len(dictionary['system_0']['density'])
        TplusU = np.zeros((num_sys,timesteps))
        TplusUplusV = np.zeros((num_sys,timesteps))
        psi0_overlap = np.zeros((num_sys,timesteps))
        psi1_overlap = np.zeros((num_sys,timesteps))
        psi2_overlap = np.zeros((num_sys,timesteps))
        # Changed: Now these are sin(omega*t) values, not phase components
        sin_omega1_t = np.zeros((num_sys,timesteps))
        sin_omega2_t = np.zeros((num_sys,timesteps))
        sin_omega3_t = np.zeros((num_sys,timesteps))
        v_array = np.zeros((num_sys,timesteps))
        n_array = np.zeros((num_sys,timesteps))
        for i in range(num_sys):
            v_array[i] = np.array(dictionary['system_'+str(i)]['ext_potential'])
            n_array[i] = np.array(dictionary['system_'+str(i)]['density'])
            TplusU[i] = np.array(dictionary['system_'+str(i)]['TplusU'])
            TplusUplusV[i] = np.array(dictionary['system_'+str(i)]['TplusUplusV'])
            psi0_overlap[i] = np.array(dictionary['system_'+str(i)]['psi0_overlap'])
            psi1_overlap[i] = np.array(dictionary['system_'+str(i)]['psi1_overlap'])
            psi2_overlap[i] = np.array(dictionary['system_'+str(i)]['psi2_overlap'])
            sin_omega1_t[i] = np.array(dictionary['system_'+str(i)]['sin_omega1_t'])
            sin_omega2_t[i] = np.array(dictionary['system_'+str(i)]['sin_omega2_t'])
            sin_omega3_t[i] = np.array(dictionary['system_'+str(i)]['sin_omega3_t'])
        return v_array, n_array, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, sin_omega1_t, sin_omega2_t, sin_omega3_t

def json_file_to_arrays(file_name, observables=False):
    """
    Load time-dependent quantum system data from JSON file and convert to NumPy arrays.
    
    Parameters
    ----------
    file_name : str
        Path to JSON file containing dataset
    observables : bool, optional
        Whether to load additional observables (default: False)
        If True, returns:
        - TplusU: Kinetic + on-site interaction energy time series
        - TplusUplusV: Total energy including potential time series
        - ψ₀/ψ₁/ψ₂ state overlaps time series
        - sin_omega1_t, sin_omega2_t, sin_omega3_t: sin(omega*t) values
    
    Returns
    -------
    tuple
        If observables=False:
        - (v_array, n_array, vhxc_array) where:
          * v_array: Potential time series [n_systems × n_timesteps]
          * n_array: Density time series [n_systems × n_timesteps]
          * vhxc_array: Hartree-exchange-correlation potential [n_systems × n_timesteps]
        
        If observables=True:
        - (v_array, n_array, vhxc_array, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, 
           sin_omega1_t, sin_omega2_t, sin_omega3_t)
          All arrays shaped [n_systems × n_timesteps]
    """
    if observables == False:
        with open(file_name, mode='r') as f:
            dictionary = json.load(f)
        num_sys = len(dictionary.keys())
        timesteps = len(dictionary['system_0']['density'])
        v_array = np.zeros((num_sys,timesteps))
        n_array = np.zeros((num_sys,timesteps))
        #vhxc_array = np.zeros((num_sys,timesteps))
        for i in range(num_sys):
            v_array[i] = np.array(dictionary['system_'+str(i)]['ext_potential'])
            n_array[i] = np.array(dictionary['system_'+str(i)]['density'])
            #vhxc_array[i] = np.array(dictionary['system_'+str(i)]['hxc_potential'])
        return v_array, n_array#, vhxc_array
    elif observables==True:
        with open(file_name, mode='r') as f:
            dictionary = json.load(f)
        num_sys = len(dictionary.keys())
        timesteps = len(dictionary['system_0']['density'])
        TplusU = np.zeros((num_sys,timesteps))
        TplusUplusV = np.zeros((num_sys,timesteps))
        psi0_overlap = np.zeros((num_sys,timesteps))
        psi1_overlap = np.zeros((num_sys,timesteps))
        psi2_overlap = np.zeros((num_sys,timesteps))
        sin_omega1_t = np.zeros((num_sys,timesteps))
        sin_omega2_t = np.zeros((num_sys,timesteps))
        sin_omega3_t = np.zeros((num_sys,timesteps))
        v_array = np.zeros((num_sys,timesteps))
        n_array = np.zeros((num_sys,timesteps))
        vhxc_array = np.zeros((num_sys,timesteps))
        for i in range(num_sys):
            v_array[i] = np.array(dictionary['system_'+str(i)]['ext_potential'])
            n_array[i] = np.array(dictionary['system_'+str(i)]['density'])
            vhxc_array[i] = np.array(dictionary['system_'+str(i)]['hxc_potential'])
            TplusU[i] = np.array(dictionary['system_'+str(i)]['TplusU'])
            TplusUplusV[i] = np.array(dictionary['system_'+str(i)]['TplusUplusV'])
            psi0_overlap[i] = np.array(dictionary['system_'+str(i)]['psi0_overlap'])
            psi1_overlap[i] = np.array(dictionary['system_'+str(i)]['psi1_overlap'])
            psi2_overlap[i] = np.array(dictionary['system_'+str(i)]['psi2_overlap'])
            sin_omega1_t[i] = np.array(dictionary['system_'+str(i)]['sin_omega1_t'])
            sin_omega2_t[i] = np.array(dictionary['system_'+str(i)]['sin_omega2_t'])
            sin_omega3_t[i] = np.array(dictionary['system_'+str(i)]['sin_omega3_t'])
        return v_array, n_array, vhxc_array, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, sin_omega1_t, sin_omega2_t, sin_omega3_t


def dataset(T, v_func_list, t_array, mode='to_array', file_name='dataset.json', observables=False):
    """
    Generate time-dependent quantum simulation datasets for a Hubbard dimer system.
    
    Parameters
    ----------
    T : array_like
        Hopping amplitudes (one value per system)
    v_func_list : list of callable
        List of potential functions v(t) for each system
    t_array : array_like
        Time points for the simulation
    mode : str, optional
        Output mode: 
        - 'to_array' returns numpy arrays (default)
        - 'to_file' saves to JSON file
    file_name : str, optional
        Name of output JSON file when mode='to_file' (default: 'dataset.json')
    observables : bool, optional
        Whether to compute additional observables (default: False)
        If True, returns/saves:
        - TplusU: Kinetic + on-site interaction energy
        - TplusUplusV: Total energy including potential
        - ψ₀/ψ₁/ψ₂ overlaps with instantaneous states
        - sin_omega1_t, sin_omega2_t, sin_omega3_t: sin(omega*t) values
    
    Returns
    -------
    tuple or None
        If mode='to_array' and observables=False:
        - (v_array, n_array) tuple of numpy arrays
        If mode='to_array' and observables=True:
        - (v_array, n_array, TplusU, TplusUplusV, psi0_overlap, psi1_overlap, psi2_overlap, 
           sin_omega1_t, sin_omega2_t, sin_omega3_t)
        If mode='to_file':
        - None (saves data to specified JSON file)
    """
    dataset_dict = {}
    num_sys = len(v_func_list)
    steps = len(t_array)
    dt = t_array[1] - t_array[0] if len(t_array) > 1 else 0.1
    tmax = t_array[-1] + dt
    
    for i in range(num_sys):
        t1 = time.time()
        
        # Create frequency interpolators for this system's hopping amplitude
        omega1_interp, omega2_interp, omega3_interp = create_frequency_interpolators(T[i])
        
        # Get the potential function for this system
        v_t = v_func_list[i]
        
        # Get initial state from ground state at t=0
        v_initial = v_t(0)
        vals, vecs = np.linalg.eigh(H_(v_initial, T[i]))
        psi0 = vecs[:,0]
        
        # Define the ODE for time evolution
        def f(t, y):
            return -1.j * np.array(H_(v_t(t), T[i]) @ y).T
        
        # Solve the time-dependent Schrödinger equation
        sln2 = solve_ivp(f, [t_array[0], t_array[-1]], np.complex128(psi0),
                        method='DOP853', t_eval=t_array, rtol=1e-12, atol=1e-12)
        
        delta_n = np.zeros(steps)
        n1_dot = np.zeros(steps)
        n1_ddot = np.zeros(steps)
        
        # Compute time-dependent potential array
        v_array = np.array([v_t(t) for t in t_array])
        
        # Map potential to time-dependent natural frequencies
        omega1_t = omega1_interp(v_array)
        omega2_t = omega2_interp(v_array)
        omega3_t = omega3_interp(v_array)
        
        # Compute sin(omega*t) values
        sin_omega1_t = np.sin(np.multiply(omega1_t, t_array))
        sin_omega2_t = np.sin(np.multiply(omega2_t, t_array))
        sin_omega3_t = np.sin(np.multiply(omega3_t, t_array))
        
        # Compute density and its derivatives
        for k in range(sln2.y.shape[-1]):
            wf = sln2.y[:,k]
            delta_n[k] = wf2d_(wf)
            q = (4*T[i]**2)*(np.abs(wf[2])**2 - np.abs(wf[0])**2) - (4/np.sqrt(2))*T[i]*(np.conjugate(wf[0])*wf[1] - np.conjugate(wf[2])*wf[1]).real
            v1 = v_t(t_array[k])/2
            n1_ddot[k] = -(8/np.sqrt(2))*T[i]*v1*(np.conjugate(wf[0])*wf[1] + np.conjugate(wf[1])*wf[2]).real + q
            n1_dot[k] = -(4/np.sqrt(2))*T[i]*(np.conjugate(wf[0])*wf[1] + np.conjugate(wf[1])*wf[2]).imag
        
        # Compute density and HXC potential
        n1 = (1 + (delta_n/2))
        n2 = (1-(delta_n/2))/2
        n1_dot = n1_dot/2
        n1_ddot = n1_ddot/2
        numer = n1_ddot + (2*T[i]**2)*(n1 - n2)
        denom = 2*np.sqrt(4*T[i]**2*n1*n2 - n1_dot**2)
        vp = -numer/denom
        vt = np.array([v_t(t) for t in t_array])/2
        vhxc = vp - vt
        
        # Store basic data
        dataset_dict[f'system_{i}'] = {
            'density': n1.tolist(),
            'ext_potential': v_array.tolist(),
            'hxc_potential': vhxc.tolist(),
            'hopping_amplitude': float(T[i]),
            'tmax': float(tmax),
            'timestep': float(dt)
        }
        
        if observables == True:
            # Get instantaneous eigenstates at t=0 for overlap calculations
            vals_0, vecs_0 = np.linalg.eigh(H_(v_initial, T[i]))
            rho0 = np.outer(vecs_0[:,0][:,None], np.conjugate(vecs_0[:,0])[None,:])
            rho1 = np.outer(vecs_0[:,1][:,None], np.conjugate(vecs_0[:,1])[None,:])
            rho2 = np.outer(vecs_0[:,2][:,None], np.conjugate(vecs_0[:,2])[None,:])
            
            # Compute additional observables
            TplusU = np.array([np.conjugate(sln2.y[:,k]) @ H_(v_array[k], T[i]) @ sln2.y[:,k] for k in range(steps)]).real
            TplusUplusV = np.array([np.conjugate(sln2.y[:,k]) @ H_(0, T[i]) @ sln2.y[:,k] for k in range(steps)]).real
            rho0_t = np.array([np.conjugate(sln2.y[:,k]) @ rho0 @ sln2.y[:,k] for k in range(steps)]).real
            rho1_t = np.array([np.conjugate(sln2.y[:,k]) @ rho1 @ sln2.y[:,k] for k in range(steps)]).real
            rho2_t = np.array([np.conjugate(sln2.y[:,k]) @ rho2 @ sln2.y[:,k] for k in range(steps)]).real
            
            dataset_dict[f'system_{i}']['TplusU'] = TplusU.tolist()
            dataset_dict[f'system_{i}']['TplusUplusV'] = TplusUplusV.tolist()
            dataset_dict[f'system_{i}']['psi0_overlap'] = rho0_t.tolist()
            dataset_dict[f'system_{i}']['psi1_overlap'] = rho1_t.tolist()
            dataset_dict[f'system_{i}']['psi2_overlap'] = rho2_t.tolist()
            
            # Store sin(omega*t) values
            dataset_dict[f'system_{i}']['sin_omega1_t'] = sin_omega1_t.tolist()
            dataset_dict[f'system_{i}']['sin_omega2_t'] = sin_omega2_t.tolist()
            dataset_dict[f'system_{i}']['sin_omega3_t'] = sin_omega3_t.tolist()
        
        t2 = time.time()
        print(f'System {i+1}/{num_sys} completed in {t2-t1:.4f}s')
    
    if mode == 'to_array':
        return dictionary_data_to_arrays(dataset_dict, observables)
    elif mode == 'to_file':
        with open(file_name, mode='w') as f:
            json.dump(dataset_dict, f)