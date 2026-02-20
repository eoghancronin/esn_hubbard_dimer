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

def generate_test1_data():
    num_sys_train = 201
    T_train = 0.05 * np.ones(num_sys_train)
    delta_v0_train = -1 - np.linspace(0, 0.4, num_sys_train)
    dt = 0.2
    t_on_train = 150 * np.ones(num_sys_train)
    tmax = 450
    gamma_off_train = -1.0 * np.ones(num_sys_train)
    steps = int(tmax / dt)
    t_array = np.linspace(0, tmax - dt, steps)
    
    v_func_list_train = []
    for i in range(num_sys_train):
        v_func_list_train.append(v_sig(delta_v0_train[i], gamma_off_train[i], t_on_train[i]))
    
    # Generate training dataset
    file_name_train = 'test1_train_data.json'
    dataset(T_train, v_func_list_train, t_array, mode='to_file', file_name=file_name_train, observables=True)
    
    # Generate TESTING dataset with 1/3 - 2/3 interpolation
    num_sys_test = 200
    T_test = 0.05 * np.ones(num_sys_test)
    delta_v0_test = np.zeros(num_sys_test)
    for i in range(num_sys_test):
        delta_v0_test[i] = (delta_v0_train[i] + delta_v0_train[i + 1]) / 2
    
    t_on_test = 150 * np.ones(num_sys_test)
    gamma_off_test = -1.0 * np.ones(num_sys_test)
    
    v_func_list_test = []
    for i in range(num_sys_test):
        v_func_list_test.append(v_sig(delta_v0_test[i], gamma_off_test[i], t_on_test[i]))
    
    # Generate testing dataset
    file_name_test = 'test1_test_data.json'
    dataset(T_test, v_func_list_test, t_array, mode='to_file', file_name=file_name_test, observables=True)
    



generate_test1_data()
