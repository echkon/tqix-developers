# -*- coding: utf-8 -*-
""" usefull code for variational quantum entanglement
"""

__all__ = ['training']


import qiskit
import tqix

import numpy as np
import copy
#from autograd.numpy.linalg import inv

def training(qc: qiskit.QuantumCircuit,
            qcirs,
            optimizer,
            num_steps,
            ofset):
            
    """ to run vqent
    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: name of circuit function
        - optimizer: optimizer
        - num_steps: number of steps
        - ofset: additional parameters for changing ce

    Returns:
        - params (Numpy array): the optimized parameters
        - cost (Numpy array): the list of loss_value
        - ce: concerntrable entanglement
    """
    
    #model = vqa.circuits.create_ansatz(qc,params,**kwargs)    
    params, costs, ce = fit(qc,qcirs,optimizer,num_steps,ofset)
    
    return [params], costs, ce


def _cost_func(qc,qcirs,ofset):
    
    """Return the cost function
        C = np.abs(ofset - ce)
    
    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: name of circuit function
        - ofset: additional parameters for changing ce

    Returns:
        - Numpy array: The cost function
    """
    #calculate ce 
    fcir = tqix.vqa.vqm.qc_add(qc.copy(), qcirs)
    ces = tqix.vqa.entanglement.concentratable_entanglement(fcir)
    
    return np.abs(ofset - ces), ces


def _grad_func(qc,qcirs,ofset):
    
    """Return the gradient of the loss function
        C = np.abs(ofset - ces)
        => nabla_C

    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - qcirs: name of circuit function
        - ofset: additional parameters for changing ce

    Returns:
        - Numpy array: The vector of gradient
    """
    
    
    s = tqix.vqa.constants.step_size
    grad_cost = [] #to reset index

    indx = list(range(0, len(qcirs[0][2])))       
    for i in indx:
        qcirs1, qcirs2 = copy.deepcopy(qcirs), copy.deepcopy(qcirs)
        qcirs1[0][2][i] += s #[0] ansatz_name, [2] ansatz_params [0] first_param
        qcirs2[0][2][i] -= s

        cost1,ces1 = _cost_func(qc.copy(),qcirs1,ofset)
        cost2,ces2 = _cost_func(qc.copy(),qcirs2,ofset)
        
        grad_cost.append((cost1 - cost2)/(2*s))
    return grad_cost


def fit(qc: qiskit.QuantumCircuit,qcirs,optimizer,num_steps,ofset):
    
    """Return the new thetas that fit with the circuit from create_circuit_func function#

    Args:
        - qc (QuantumCircuit): The quantum circuit want to calculate the gradient
        - cirs: name of circuit function
        - optimizer: optimizer
        - num_steps: number of steps
        - ofset: additional parameters for changing ce

    Returns:
        - thetas (Numpy array): the optimized parameters
        - loss_values (Numpy array): the list of loss_value
    """
    
    costs = []
    
    for i in range(0, num_steps):
        dev_cost = _grad_func(qc.copy(),qcirs,ofset)
        params = qcirs[0][2]
        
        # update params
        params = optimizer(params,dev_cost,i)
        qcirs[0][2] = params
        
        # compute cost    
        cost,ces = _cost_func(qc.copy(),qcirs,ofset)
        
        costs.append(cost)
        #print(i,cost)
        
    #calculate ce
    #qc1 = create_circuit_func(qc,params,**kwargs)
    #ce = vqa.entanglement.concentratable_entanglement(qc1)
    
    return params, costs, ces


def sgd(params, dev_cost, i):
    """optimizer standard gradient descene
    
    Args:
        - params: parameters
        - dev_cost: gradient of the cost function
        - i: use for adam (dont use here)
    
    Returns:
        - params
    """
    learning_rate = tqix.vqa.constants.learning_rate
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
        
        params[i] -= tqix.vqa.constants.learning_rate * mhat / (np.sqrt(vhat) + epsilon)
    
    return params
    
    
    