"""
>>> OneCircuit: One solution for all quantum circuit needs
________________________________
>>> copyright (c) 2024 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""

import qiskit
import numpy as np
from onecircuit.util import init_params

def stargraph(num_qubits: int, 
            num_layers: int, 
            params: np.ndarray) -> qiskit.QuantumCircuit:
    
    """Create star graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != num_layers*(2*num_qubits - 2):
        raise ValueError(
            f'The number of parameter must be {num_layers*(2*num_qubits - 2)}')
    
    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(2, num_qubits):
            qc.ry(params[j], 0)
            j += 1
            qc.cz(0, i)
        qc.barrier()   
        
    return qc

def stargraph_inv(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create invert star graph ansatz

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """
    return stargraph(num_qubits,num_layers,params).inverse()

def stargraph_eco(num_qubits: int, 
            num_layers: int, 
            params: np.ndarray) -> qiskit.QuantumCircuit:
    
    """Create star graph ansatz economy

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    
    if len(params) != num_layers*num_qubits:
        raise ValueError(
            f'The number of parameter must be {num_layers*num_qubits}')
    
    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            j += 1
        for i in range(1, num_qubits):
            qc.cz(0, i)
        qc.barrier()   
        
    return qc

def stargraph_eco_inv(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create invert star graph ansatz economy

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """
    return stargraph_eco(num_qubits,num_layers,params).inverse()

def ringgraph(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create ring graph ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    if len(params) != num_layers*(2*num_qubits):
        raise ValueError(
            f'The number of parameter must be {num_layers*(2*num_qubits)}')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            j += 1
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        if num_qubits % 2 == 1:
            for i in range(0, num_qubits - 1):
                qc.ry(params[j], i)
                j += 1
        else:
            for i in range(0, num_qubits):
                qc.ry(params[j], i)
                j += 1
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        if num_qubits % 2 == 1:
            qc.ry(params[j], num_qubits - 1)
            j += 1
        qc.cz(0, num_qubits - 1)
        qc.barrier()
    return qc
    
def ringgraph_inv(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create inverse ring graph ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """
    return ringgraph(num_qubits,num_layers,params).inverse()

def ringgraph_eco(num_qubits: int, 
            num_layers: int, 
            params: np.ndarray) -> qiskit.QuantumCircuit:

    """Create ring graph ansatz economy

    Args:
        num_qubits: number of qubits
        num_layers: number of layers
        params (np.ndarray): number of parameters

    Returns:
        Quantum Circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)

    if len(params) != num_layers*num_qubits:
        raise ValueError(
            f'The number of parameter must be {num_layers*num_qubits}')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            j += 1
        for i in range(0, num_qubits-1):
            qc.cz(i, i + 1)
        qc.cz(0, num_qubits - 1)
        qc.barrier()
    return qc

def ringgraph_eco_inv(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create inverse ring graph ansatz economy

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return ringgraph_eco(num_qubits,num_layers,params).inverse()

def linegraph(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create line graph ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    if len(params) != num_layers*(2*num_qubits):
        raise ValueError(
            f'The number of parameter must be {num_layers*(2*num_qubits)}')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            j += 1
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        for i in range(1, num_qubits - 1):
            qc.ry(params[j], i)
            j += 1
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()
    return qc
    
def linegraph_inv(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create inverse line graph ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """
    return linegraph(num_qubits,num_layers,params).inverse()

def linegraph_eco(num_qubits: int, 
            num_layers: int, 
            params: np.ndarray) -> qiskit.QuantumCircuit:

    """Create line graph ansatz economy

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    if len(params) != num_layers*num_qubits:
        raise ValueError(
            f'The number of parameter must be {num_layers*num_qubits}')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[j], i)
            j += 1
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()
    return qc

def linegraph_eco_inv(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create inverse line graph ansatz economy

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return linegraph_eco(num_qubits,num_layers,params).inverse()

def squeezing(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create squeezing ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """ 
    
    qc = qiskit.QuantumCircuit(num_qubits)
    if len(params) != num_layers*num_qubits*(num_qubits+1):
        raise ValueError(
            f'The number of parameter must be {num_layers*num_qubits*(num_qubits+1)}')
        
    k = 0
    for l in range(0, num_layers, 1):
        for i in range(0, num_qubits):
            qc.ry(params[k], i)
            k += 1
            
        # GMS_z gate
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                qc.rzz(params[k], i, j)
                k += 1
        # RX gate
        for i in range(0, num_qubits):
            qc.rx(params[k], i)
            k += 1

        # GMS_x gate
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                qc.rxx(params[k], i, j)   
                k += 1
        qc.barrier()    
    return qc

def squeezing_inv(
    num_qubits,
    num_layers: int,
    params: np.ndarray)-> qiskit.QuantumCircuit:
    
    """Create squeezing ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """ 
    
    return squeezing(num_qubits,num_layers,params).inverse()

def random_ansatz(num_qubits: int, 
            cnot_pairs, 
            params: np.ndarray) -> qiskit.QuantumCircuit:
    
    """Create random ansatz

    Args:
        num_qubits: number of qubits
        cnot_pairs: random set of cnot position
        params (np.ndarray): number of parameters

    Returns:
        One random quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    
    D = len(cnot_pairs)

    if len(params) != 6*D:
        raise ValueError(
            f'The number of parameter must be {6*D}')    
    
    j = 0
    for (control, target) in cnot_pairs:

        # Apply U3 gates to control and target qubits
        qc.u(params[j], params[j+1], params[j+2], control)
        qc.u(params[j+3], params[j+4], params[j+5], target)
        
        # Apply CNOT gate
        qc.cx(control, target)

        qc.barrier() 

        j += 6
        
    return qc

def create_num_params(single_sensor):
    """ to create a list of number of parameters:
    
    Args:
        - single_sensor = [qc_name, num_qubits, num_layers, params]

    Returns:
        - total number of paramaters
    """

    qc_name = single_sensor[0].__name__
    num_qubits = single_sensor[1]
    num_layers = single_sensor[2]
    cnot_pairs = single_sensor[2]
    
    if qc_name == 'stargraph' or qc_name == 'stargraph_inv':
        num_params = num_layers*(2*num_qubits - 2)
    elif qc_name == 'stargraph_eco' or qc_name == 'stargraph_eco_inv':
        num_params = num_layers*num_qubits
    elif qc_name == 'ringgraph' or qc_name == 'ringgraph_inv':
        num_params = num_layers*2*num_qubits
    elif qc_name == 'ringgraph_eco' or qc_name == 'ringgraph_eco_inv':
        num_params = num_layers*num_qubits
    elif qc_name == 'linegraph' or qc_name == 'linegraph_inv':
        num_params = num_layers*(2*num_qubits)
    elif qc_name == 'linegraph_eco' or qc_name == 'linegraph_eco_inv':
        num_params = num_layers*num_qubits
    elif qc_name == 'squeezing' or qc_name == 'squeezing_inv' :
        num_params = num_layers*num_qubits*(num_qubits+1)  
    elif qc_name == 'random_ansatz' or qc_name == 'random_ansatz_inv':
        num_params = 6*len(cnot_pairs)
    elif qc_name == 'ghz_qc' or qc_name == 'gh3_qc':
        num_params = 0
    #elif qc_name == 'u_phase':
    #    num_params = 3 # fixed
    else:
        raise ValueError(
            'No ansatz was given.') 
    
    return int(num_params)


def create_params(single_sensor):
    
    """ to create a list of parameters:
    
    Args:
        - single_sensor = [qc_name, num_qubits, num_layers, params]
        - value: initial value
        
    Returns:
        - update params
    """
    
    num_params = create_num_params(single_sensor)

    if init_params == -1: # -1 means random
        params = (np.random.uniform(0, 2 * np.pi, num_params))    
    else:
        params = [init_params] * num_params
        
    return params