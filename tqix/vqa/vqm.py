# -*- coding: utf-8 -*-
""" Variational quantum metrology
    Author: Le Bin Ho @ 2023
"""

#__all__ = []

import qiskit
import numpy as np
import tqix.vqa.fitting 

from qiskit.extensions import UnitaryGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

#model can dua params vao

def training(qc: qiskit.QuantumCircuit,
             qcirs,
             which_train,
             cost_func,
             grad_func,
             opt_func,
             num_steps):
    
    """ train a vqm model
    
    Args:
        - qc (QuantumCircuit): Fitting circuit
        - qcirs: set of circuits functions
        - which_train: training option
        - coss_func (FunctionType): coss function
        - grad_func (FunctionType): Gradient function
        - opt_func (FunctionType): Otimizer function
        - num_steps (Int): number of iterations
        
    Returns:
        - optimal parameters
    """
    

    return tqix.vqa.fitting.fit(qc,qcirs,which_train,cost_func,grad_func,opt_func,num_steps)
    

def qc_add(qc, qcirs):
    
    """ create a full circuit from qc_func
    
    Args:
        - qc: initial circuit
        - qcirs: list of circuits, num_layers, parameters, ....
        
    Returns:
        - qc: final circuit
    """
    
    cirq = qc.copy()    
    for i in range (len(qcirs)):
         cirq &= qcirs[i][0](qc.copy(), qcirs[i][1], qcirs[i][2])
        
    return cirq
 
#
# custom unitary for phase and noises
#

def u_phase(qc: qiskit.QuantumCircuit, t, params):
    
    """Add phase model to the circuit
    
    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time coefs
        - params  (number of parameters)
        
    Return
        - qc
    """
    n = qc.num_qubits
    
    for i in range(n):        
        qc.append(_u_gate(t, params),[i])
    
    return qc


def _u_gate(t, params):
    """ return arbitary gates
    
    Args:
        - t: time coefs
        - params: phases
    
    Method:
        H = x*sigma_x + y*sigma_y +  z*sigma_z
        U = exp(-i*t*H)
        t: time (use for time-dephasing)
    """
        
    x,y,z = params[0],params[1],params[2]
    p2 = np.sqrt(x*x + y*y + z*z)
    tp2 = t * p2
    
    u11 = np.cos(tp2) - 1j*z*np.sin(tp2)/p2
    u12 = (-1j*x - y)*np.sin(tp2)/p2
    u21 = (-1j*x + y)*np.sin(tp2)/p2
    u22 = np.cos(tp2) + 1j*z*np.sin(tp2)/p2
    
    u = [[u11, u12],[u21,u22]]
    gate = UnitaryGate(u, 'u_phase')

    return gate
