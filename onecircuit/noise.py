# -*- coding: utf-8 -*-
""" Variational quantum metrology
    Author: Le Bin Ho @ 2023
"""

import qiskit
import numpy as np

from qiskit.quantum_info import Kraus
#from qiskit.providers.aer.noise import QuantumError
from qiskit_aer.noise import QuantumError


# various noise via Kraus operators
def dephasing_qc(num_qubits: int, 
                time: float = 1.0, 
                y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add dephasing to the circuit

    Args:
        - num_qubits = number of qubits
        - time: time coefs
        - y: gamma in params
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-time*y)
    
    # kraus operators
    k1 = np.array([[1, 0],[0, np.sqrt(1 - lamb)]])
    k2 = np.array([[0, 0],[0, np.sqrt(lamb)]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops) 
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def bit_flip(num_qubits: int, 
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add bit flip to the circuit

    Args:
        - num_qubit: number of qubits
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-time*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(lamb) * np.array([[0, 1], [1, 0]])])
    kraus_to_error = QuantumError(noise_ops)
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def phase_flip(num_qubits: int, 
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add phase flip to the circuit

    Args:
        - num qubits
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-time*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(lamb) * np.array([[1, 0], [0, -1]])])
    kraus_to_error = QuantumError(noise_ops)
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def depolarizing(num_qubits: int, 
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add depolarizing to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-time*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                       np.sqrt(lamb/3.) * np.array([[0, 1], [1, 0]]),
                       np.sqrt(lamb/3.) * np.array([[0, -1j], [1j, 0]]),
                       np.sqrt(lamb/3.) * np.array([[1, 0], [0, -1]])
                        ])
    kraus_to_error = QuantumError(noise_ops)
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def markovian(num_qubits: int, 
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    qt = 1 - np.exp(-y*time) 
    
    k1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    k2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops)  
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
    
    return qc


def non_markovian(num_qubits: int, 
            time: float = 1.0, 
            y: np.ndarray = []) -> qiskit.QuantumCircuit:
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    tc = 20.0 #fixed in rXiv:2305.08289
    qt = 1 - np.exp(-y*time**2/(2*tc))
    
    k1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    k2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops)  
    
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.append(kraus_to_error,[i])
    
    return qc