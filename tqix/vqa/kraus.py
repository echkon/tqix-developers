# -*- coding: utf-8 -*-
""" Variational quantum metrology
    Author: Le Bin Ho @ 2023
"""

#__all__ = []

import qiskit
import numpy as np

from qiskit.extensions import UnitaryGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Kraus
from qiskit.providers.aer.utils import approximate_quantum_error
from qiskit.providers.aer.noise import QuantumError


# various noise via Kraus operators
def dephasing(qc, t, y):
    """Add dephasing to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time coefs
        - y: gamma in params
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-t*y)

    # kraus operators
    k1 = np.array([[1, 0],[0, np.sqrt(1 - lamb)]])
    k2 = np.array([[0, 0],[0, np.sqrt(lamb)]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops) 
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def bit_flip(qc, t, y):
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-t*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(lamb) * np.array([[0, 1], [1, 0]])])
    kraus_to_error = QuantumError(noise_ops)
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def phase_flip(qc, t, y):
    """Add phase flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-t*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(lamb) * np.array([[1, 0], [0, -1]])])
    kraus_to_error = QuantumError(noise_ops)
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def depolarizing(qc, t, y):
    """Add depolarizing to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-t*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                       np.sqrt(lamb/3.) * np.array([[0, 1], [1, 0]]),
                       np.sqrt(lamb/3.) * np.array([[0, -1j], [1j, 0]]),
                       np.sqrt(lamb/3.) * np.array([[1, 0], [0, -1]])
                      ])
    kraus_to_error = QuantumError(noise_ops)
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def markovian(qc: qiskit.QuantumCircuit, t, y):
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t:  time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    qt = 1 - np.exp(-y*t) #y = gamma = 0.1
    
    k1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    k2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops)  
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def non_markovian(qc: qiskit.QuantumCircuit, t, y):
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t:  time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    tc = 20.0 #fixed in rXiv:2305.08289
    qt = 1 - np.exp(-y*t**2/(2*tc)) #gamma = 0.1
    
    k1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    k2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops)  
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc
 