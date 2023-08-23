# -*- coding: utf-8 -*-
""" usefull code for barren plateau
"""

__all__ = ['plateau']


import qiskit
import tqix.vqa.constants
import tqix.vqa.circuits
import copy

import numpy as np


def plateau(qc: qiskit.QuantumCircuit, 
            qcirs,
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
        grad = gradf(qc.copy(),qcirs,cost_func)      
        grads.append(grad) 
        
    var =  np.var(grads)       
    return var


def gradf(qc: qiskit.QuantumCircuit,qcirs,cost_func):
    
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
    
    s = tqix.vqa.constants.step_size    
    
    qcirs1, qcirs2 = copy.deepcopy(qcirs), copy.deepcopy(qcirs)
    qcirs1[0][0][0] += s
    qcirs2[0][0][0] -= s
        
    cost1 = cost_func(qc.copy(),qcirs1)
    cost2 = cost_func(qc.copy(),qcirs2)
    
    return (cost1 - cost2)/(2*s)
    
    
    
    