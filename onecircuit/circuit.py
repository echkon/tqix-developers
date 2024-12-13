# -*- coding: utf-8 -*-
""" Everything belong to circuit
"""

import qiskit
import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from tqix import *

def tensor_product(state, num_qubits: int):
    tensor_product = state
    I = eyex(2)
    for _ in range(1, num_qubits):
        tensor_product = np.kron(tensor_product, I)
    
    return tensor_product

def ghz_qc(num_qubits: int, *args, **kwargs) -> qiskit.QuantumCircuit:
    
    """Create GHZ state with a parameter

    Args:
        - num_qubits: number of qubits
        - *kwargs: for other parameter
        - **kwargs: for other parameters

    Returns:
        - QuantumCircuit: the added circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.h(0)
    
    for i in range(0, num_qubits - 1):
        qc.cnot(0, i + 1)
    return qc

def gh3_qc(num_qubits: int, *args, **kwargs) -> qiskit.QuantumCircuit:
    
    """Create GH3 state with a parameter

    Args:
        - num_qubits: number of qubits
        - *kwargs: for other parameter
        - **kwargs: for other parameters

    Returns:
        - QuantumCircuit: the added circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)

    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    tensor_x = tensor_product(sx, num_qubits)
    tensor_y = tensor_product(sy, num_qubits)
    tensor_z = tensor_product(sz, num_qubits)

    _, eigenstate_x = eigenx(tensor_x)
    _, eigenstate_y = eigenx(tensor_y)
    _, eigenstate_z = eigenx(tensor_z)

    ghx = 1/np.sqrt(2) * (eigenstate_x[0] + eigenstate_x[-1])
    ghy = 1/np.sqrt(2) * (eigenstate_y[0] + eigenstate_y[-1])
    ghz = 1/np.sqrt(2) * (eigenstate_z[0] + eigenstate_z[-1])
    gh3 = normx(ghx + ghy + ghz)

    ghz = ghz.flatten().tolist()
    gh3 = gh3.flatten().tolist()

    qc.initialize(gh3, list(range(num_qubits)))

    op = Statevector(qc).data

    return qc

def stargraph_qc(num_qubits, *args, **kwargs)-> qiskit.QuantumCircuit:
    
    """Create start graph 

    Args:
        - num_qubits: number of qubits
        - *args, **kwargs: None

    Returns:
        - QuantumCircuit: the added circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    
    for i in range(0, num_qubits):
        qc.h(i)
    for i in range(1, num_qubits):
        qc.cz(0, i)
    qc.barrier()   
    
    return qc

def ringgraph_qc(num_qubits, *args, **kwargs)-> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits)
    for i in range(0, num_qubits - 1, 2):
        qc.cz(i, i + 1)
    if n % 2 == 1:
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        qc.cz(0, num_qubits - 1)
        qc.barrier()
    return qc

def sensing_qc(num_qubits: int, 
            time: float, 
            params: np.ndarray) -> qiskit.QuantumCircuit:
    """_summary_

    Args:
        num_qubits (int): _numnber of qubits_
        time (float): _time_
        params (np.ndarray): _number of phase_

    Returns:
        qiskit.QuantumCircuit: _quantum circuit_
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    
    for i in range(0,num_qubits):     
        qc.append(phase_gate(time, params),[i])  
        
    return qc

def phase_gate(t, phis): #unitary
    x,y,z = phis
    w = np.sqrt(x*x + y*y + z*z) #omega
    wt = t * w
    
    u11 = np.cos(wt) - 1j*z*np.sin(wt)/w
    u12 = (-1j*x - y)*np.sin(wt)/w
    u21 = (-1j*x + y)*np.sin(wt)/w
    u22 = np.cos(wt) + 1j*z*np.sin(wt)/w
        
    umat =  np.array([[u11,u12],[u21, u22]])
    gate = UnitaryGate(umat, 'sensing_qc')

    return gate

def sensing_dm(num_qubits: int, 
                time: float, 
                phase: np.ndarray) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits)
    
    for i in range(0,num_qubits):     
        qc.append(phase_gate_dm(phase,0.0),[i])  
        
    return qc

def phase_gate_dm(phase, alpha):
    delta = phase[0]

    u11 = np.cos(delta)
    u12 = 1j * np.exp(-1j * alpha) * np.sin(delta)
    u21 = 1j * np.exp(1j * alpha) * np.sin(delta)
    u22 = np.cos(delta)
        
    U_DM =  np.array([[u11,u12],[u21, u22]])
    gate = UnitaryGate(U_DM, 'sensing_dm')

    return gate

def sensing_dmi(num_qubits: int, 
                index: list, 
                phase: np.ndarray) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits)
    
    for i in index:
        qc.append(phase_gate_dm(phase,0.0),[i])
        
    return qc

def sensing_dmi_alpha(num_qubits: int, 
                      alpha: list, 
                      phase: np.ndarray) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(num_qubits)
    
    for i in range(0,num_qubits):
        qc.append(phase_gate_dm(phase,alpha),[i])
        
    return qc