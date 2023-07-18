# -*- coding: utf-8 -*-
""" usefull code for barren plateau
"""

__all__ = ['plateau']


import qiskit
import vqa.constants
import vqa.circuits
import copy

import numpy as np


def plateau(qc: qiskit.QuantumCircuit, 
            cirs,
            coefs,
            params,
            cost_func,
            num_samples):
            
    """ to run plateau
    
    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - cirs (Function): circuit functions
        - coefs (Numpy array): coefs
        - params (Numpy array): Parameters
        - cost_func: name of cost function
        - num_sample = number of sample

    Returns:
        - barren plateau
    """
    
    num_qubits = qc.num_qubits
    grads = []
    
    for _ in range(num_samples):
        params = vqa.circuits.create_params(cirs,coefs,num_qubits,['random',params[1],params[2],'random'])  
        grad = gradf(qc.copy(),cirs,coefs,params,cost_func)      
        grads.append(grad) 
        
    var =  np.var(grads)       
    return var


def gradf(qc: qiskit.QuantumCircuit,cirs,coefs,params,cost_func):
    
    """Return the gradient of the loss function
        w.r.t theta_1

    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - cirs (Function): circuit functions
        - coefs (Numpy array): coefs
        - params (Numpy array): Parameters
        - cost_func: name of cost function

    Returns:
        - gradient of loss function w.r.t. theta_0 
    """
    
    s = vqa.constants.step_size    
    params1, params2 = copy.deepcopy(params), copy.deepcopy(params)
    params1[0][0] += s #[0][0]the first paramater
    params2[0][0] -= s

    cost1 = cost_func(qc.copy(),cirs,coefs,params1) #ues full params
    cost2 = cost_func(qc.copy(),cirs,coefs,params2)
    
    return (cost1 - cost2)/(2*s)
    
    
    
    