# -*- coding: utf-8 -*-
""" usefull base
"""

#__all__ = ['def_cost_func',
#           'grad_cost']


import qiskit
import vqa.constants
import vqa.bounds

import numpy as np
from numpy.linalg import inv
import copy

def trace_f_invq(qc, qcirs):                        
    """Return the cost function
        C = 1 - tr[cfim*qfim.inv]/d
    
    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: set of circuits 

    Returns:
        - Numpy array: The cost function
    """
        
    #calculate F and Q here
    qfim = vqa.bounds.sld_qfim(qc.copy(),qcirs)          
    cfim = vqa.bounds.cfim(qc.copy(),qcirs) 
    inv_qfim = inv(qfim + np.eye(len(qfim)) * 10e-10)
    
    return 1 - np.trace(cfim @ inv_qfim)/len(qcirs[1][2]) #[1]:u_phase,[1][2]:phases


def bound_sld_cls(qc, qcirs):
    """Return the cost function
        C = 1 - sld_bound/cls_bound
    
    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: set of circuits 

    Returns:
        - Numpy array: The cost function
    """
    
    sld_bound = vqa.bounds.sld_bound(qc.copy(),qcirs) 
    rld_bound = vqa.bounds.rld_bound(qc.copy(),qcirs) 
    cls_bound = vqa.bounds.cls_bound(qc.copy(),qcirs)

    return 1 - sld_bound/cls_bound


def bound_rld_cls(qc, qcirs):           
    """Return the cost function
        C = 1 - sld_bound/cls_bound
    
    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: set of circuits 

    Returns:
        - Numpy array: The cost function
    """
       
    rld_bound = vqa.bounds.rld_bound(qc.copy(),qcirs)  
    cls_bound = vqa.bounds.cls_bound(qc.copy(),qcirs)
    
    return 1 - rld_bound/cls_bound


def norm2(qc, qcirs):
    """Return the cost function
        C = trace(f) - trace(q)
    
    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: set of circuits 

    Returns:
        - Numpy array: The cost function
    """
    #calculate F and Q here
    qfim = vqa.bounds.sld_qfim(qc.copy(),qcirs)          
    cfim = vqa.bounds.cfim(qc.copy(),qcirs)
    inv_qfim = inv(qfim + np.eye(len(qfim)) * 10e-10)
    
    return np.linalg.norm(cfim - qfim, ord = 2) #np.trace(inv_qfim) 


def cost_func_sld(qc, qcirs):
    """Return the cost function sld
        C = sld_qfim
    
    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: set of circuits 

    Returns:
        - Numpy array: The cost function
    """
    sld_qfim = vqa.bounds.sld_bound(qc.copy(),qcirs)     
        
    return sld_qfim


def cost_func_cls(qc, qcirs):
    """Return the cost function cls
        C = cls_qfim
    
    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: set of circuits 

    Returns:
        - Numpy array: The cost function
    """
    cfim = vqa.bounds.cls_bound(qc.copy(),qcirs)     
        
    return cfim


def grad_cost(qc, qcirs, cost_func, which_train):
    
    """Return the gradient of the loss function
        C = 1 - tr[cfim*qfim.inv]/d
        => nabla_C

    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: set of circuits 
        - cost_func: defined cost function
        - which_train (Numpy array): setting which train

    Returns:
        - Numpy array: The vector of gradient
    """
    
    
    s = vqa.constants.step_size
    grad_cost = [] #to reset index
        
    for i in which_train:
        for j in range(len(qcirs[i][2])):  #[i]:vi tri cirquit, [2]:param in circuit[i]
            qcirs1, qcirs2 = copy.deepcopy(qcirs), copy.deepcopy(qcirs)
            qcirs1[i][2][j] += s
            qcirs2[i][2][j] -= s

            cost1 = cost_func(qc.copy(),qcirs1)
            cost2 = cost_func(qc.copy(),qcirs2)
            
            grad_cost.append((cost1 - cost2)/(2*s))
            
    return grad_cost

def fit(qc: qiskit.QuantumCircuit,
        qcirs,
        which_train,
        cost_func,
        grad_func,
        opt_func,
        num_steps):
    """Return the new thetas that fit with the circuit from create_circuit_func function#

    Args:
        - qc (QuantumCircuit): Fitting circuit
        - qcirs: set of circuits functions
        - which_train: training option
        - coss_func (FunctionType): coss function
        - grad_func (FunctionType): Gradient function
        - opt_func (FunctionType): Otimizer function
        - num_steps (Int): number of iterations

    Returns:
        - thetas (Numpy array): the optimized parameters
        - loss_values (Numpy array): the list of loss_value
    """
    paramss = []
    costs = []
        
    # get params_train
    params_train = []
    for i in which_train:
        params_train.extend(qcirs[i][2]) #[i]:vi tri cirquit, [2]:param in circuit[i]
    
    for s in range(0, num_steps):
        dev_cost = grad_func(qc.copy(),qcirs,cost_func,which_train)
            
        # calculate params_train
        params_train = opt_func(params_train,dev_cost,s)
        
                
        # update full params
        pre_ind = 0
        for i in which_train:              
            qcirs[i][2] = params_train[pre_ind:pre_ind+len(qcirs[i][2])]
            pre_ind += len(qcirs[i][2])           

        # compute cost   
        cost = cost_func(qc.copy(),qcirs)
        
        costs.append(cost)
        print(s,cost)
        
    return qcirs, costs


def sgd(params, dev_cost, i):
    """optimizer standard gradient descene
    
    Args:
        - params: parameters
        - dev_cost: gradient of the cost function
        - i: use for adam (dont use here)
    
    Returns:
        - params
    """
    learning_rate = vqa.constants.learning_rate
    params -= learning_rate*dev_cost
    return params
    
def adam(params, dev_cost, iteration):
    """optimizer adam
    
    Args:
        - params: parameters
        - dev_cost: gradient of the cost function
    
    Returns:
        - params
    """    
    num_params = len(params) #.shape[0]
    beta1, beta2,epsilon = 0.8, 0.999, 10**(-8)
    
    m, v = list(np.zeros(num_params)), list(
                    np.zeros(num_params))
    
    for i in range(num_params):
        m[i] = beta1 * m[i] + (1 - beta1) * dev_cost[i]
        v[i] = beta2 * v[i] + (1 - beta1) * dev_cost[i]**2
        mhat = m[i] / (1 - beta1**(iteration + 1))
        vhat = v[i] / (1 - beta2**(iteration + 1))
   
        params[i] -= vqa.constants.learning_rate * mhat / (np.sqrt(vhat) + epsilon)
    
    return params
    
    
    