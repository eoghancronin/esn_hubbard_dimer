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

def sigmoid(t, t_on, t_off, gamma_on, gamma_off):
    return (1/(1+np.exp(-gamma_on*(t-t_on))))*(1/(1+np.exp(gamma_off*(t-t_off))))

def v_sin(v0, v1, omega, t_on, t_off, gamma_on, gamma_off):
    def v(t):
        eps = v0 + sigmoid(t, t_on, t_off, gamma_on, gamma_off)*2*v1*np.sin(omega*t)
        return eps
    return v

def v_sig(dv0, gamma, t_on):
    def v(t):
        return dv0 * (1 / (1 + np.exp(gamma * (t - t_on))))
    return v

def sig(gamma, t_on):
    def sig_(t):
        return (1/(1+np.exp(-gamma*(t-t_on))))
    return sig_

def generate_test3_data():
    num_sys_train = 201
    T_train = 0.05*np.ones(num_sys_train)
    delta_v0_train = 0.04 + 0.0*np.linspace(0, 1, num_sys_train)
    amplitude_train = 0.1*np.ones(num_sys_train)
    dt = 0.2
    tmax = 650
    t_on_train = 150*np.ones(num_sys_train)
    t_off_train = 1000*np.ones(num_sys_train)
    gamma_on_train = 0.2*np.ones(num_sys_train)
    gamma_off_train = 0.2*np.ones(num_sys_train)
    omega_dr_train = np.linspace(0.965, 1.065, num_sys_train)
    
    steps = int(tmax/dt)
    t_array = np.linspace(0, tmax-dt, steps)
    
    
    v_func_list_train = []
    for i in range(num_sys_train):
        v_func_list_train.append(v_sin(delta_v0_train[i], amplitude_train[i], omega_dr_train[i],
                                       t_on_train[i], t_off_train[i], gamma_on_train[i], gamma_off_train[i]))
    
    # Generate training dataset
    file_name_train = 'test3_train_data.json'
    dataset(T_train, v_func_list_train, t_array, mode='to_file', file_name=file_name_train, observables=True)
    
    # Generate TESTING dataset with midpoint values
    num_sys_test = 200
    T_test = 0.05*np.ones(num_sys_test)
    delta_v0_test = 0.04 + 0.0*np.linspace(0, 1, num_sys_test)
    amplitude_test = 0.1*np.ones(num_sys_test)
    t_on_test = 150*np.ones(num_sys_test)
    t_off_test = 1000*np.ones(num_sys_test)
    gamma_on_test= 0.2*np.ones(num_sys_test)
    gamma_off_test = 0.2*np.ones(num_sys_test)
    
    omega_dr_test = np.zeros(num_sys_test)
    for i in range(num_sys_test):
        omega_dr_test[i] = (omega_dr_train[i] + 2*omega_dr_train[i+1])/3
    
    v_func_list_test = []
    for i in range(num_sys_test):
        v_func_list_test.append(v_sin(delta_v0_test[i], amplitude_test[i], omega_dr_test[i],
                                      t_on_test[i], t_off_test[i], gamma_on_test[i], gamma_off_test[i]))
    
    # Generate testing dataset
    file_name_test = 'test3_test_data.json'
    dataset(T_test, v_func_list_test, t_array, mode='to_file', file_name=file_name_test, observables=True)
    
    num_sys_validation = 200
    T_validation = 0.05*np.ones(num_sys_validation)
    delta_v0_validation = 0.04 + 0.0*np.linspace(0, 1, num_sys_validation)
    amplitude_validation = 0.1*np.ones(num_sys_validation)
    t_on_validation = 150*np.ones(num_sys_validation)
    t_off_validation = 1000*np.ones(num_sys_validation)
    gamma_on_validation= 0.2*np.ones(num_sys_validation)
    gamma_off_validation = 0.2*np.ones(num_sys_validation)
    
    omega_dr_validation = np.zeros(num_sys_validation)
    for i in range(num_sys_test):
        omega_dr_validation[i] = (2*omega_dr_train[i] + omega_dr_train[i+1])/3
    
    v_func_list_validation = []
    for i in range(num_sys_validation):
        v_func_list_validation.append(v_sin(delta_v0_validation[i], amplitude_validation[i], omega_dr_validation[i],
                                      t_on_validation[i], t_off_validation[i], gamma_on_validation[i], gamma_off_validation[i]))
    
    # Generate testing dataset
    file_name_validation = 'test3_validation_data.json'
    dataset(T_validation, v_func_list_validation, t_array, mode='to_file', file_name=file_name_validation, observables=True)




generate_test3_data()