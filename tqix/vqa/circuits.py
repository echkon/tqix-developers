# -*- coding: utf-8 -*-
""" Everything belong to circuit
"""

#__all__ = []

import qiskit
import numpy as np
from qiskit.circuit.library import GMS

import tqix.vqa.constants

#
# ansatz
#

def create_ansatz(qc: qiskit.QuantumCircuit, 
                  qc_func,
                  params, 
                  num_layers):
    
    """ create an ansatz with a given name
    
    Args:
        - qc: initial circuit
        - qc_func: circuit function (3 agrs)
        - params: number of params
        - num_layers: num layers
    
    Return:
        - quantum circuit
    """    
    
    return qc_func(qc.copy(), num_layers, params)


def ghz_ansatz(
    qc: qiskit.QuantumCircuit,
    theta: float = np.pi / 2,
    **kwargs):
    
    """Create GHZ state with a parameter

    Args:
        - qc (QuantumCircuit): Init circuit
        - theta (Float): Parameter
        - **kwargs: for other parameters

    Returns:
        - QuantumCircuit: the added circuit
    """
    
    if isinstance(theta, float) != True:
        theta = (theta['theta'])
    qc.ry(theta, 0)
    
    qc.h(0)
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(0, i + 1)
    return qc

    
def star_ansatz(
    qc: qiskit.QuantumCircuit,
    num_layers: int,
    params: np.ndarray):
    
    """Create star graph ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        params (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """
    
    n = qc.num_qubits
    
    if len(params) != num_layers*(2*n - 2):
        raise ValueError(
            'The number of parameter must be num_layers*(2*n - 2)')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, n):
            qc.ry(params[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(2, n):
            qc.ry(params[j], 0)
            j += 1
            qc.cz(0, i)
        qc.barrier()    
    return qc


def star_ansatz_inv(
    qc: qiskit.QuantumCircuit,
    num_layers: int,
    params: np.ndarray):
    
    """Create invert star graph ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        params (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """
      
    return star_ansatz(qc,num_layers,params).inverse()

    
def ring_ansatz(
    qc: qiskit.QuantumCircuit,
    num_layers: int,
    params: np.ndarray):
    
    """Create ring graph ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    n = qc.num_qubits
    if len(params) != num_layers*(2*n):
        raise ValueError(
            'The number of parameter must be num_layers*(2*n)')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, n):
            qc.ry(params[j], i)
            j += 1
        for i in range(0, n - 1, 2):
            qc.cz(i, i + 1)
        if n % 2 == 1:
            for i in range(0, n - 1):
                qc.ry(params[j], i)
                j += 1
        else:
            for i in range(0, n):
                qc.ry(params[j], i)
                j += 1
        for i in range(1, n - 1, 2):
            qc.cz(i, i + 1)
        if n % 2 == 1:
            qc.ry(params[j], n - 1)
            j += 1
        qc.cz(0, n - 1)
        qc.barrier()
    return qc
    
def ring_ansatz_inv(
    qc: qiskit.QuantumCircuit,
    num_layers: int,
    params: np.ndarray):
    
    """Create inverse ring graph ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """

    return ring_ansatz(qc,num_layers,params).inverse()

    
def squeezing_ansatz(
    qc: qiskit.QuantumCircuit,
    num_layers: int,
    params: np.ndarray):
    
    """Create squeezing ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """ 
    
    n = qc.num_qubits
    if len(params) != num_layers*n*(n+1):
        raise ValueError(
            'The number of parameter must be num_layers*n*(n+1)')
        
    k = 0
    for l in range(0, num_layers, 1):
        for i in range(0, n):
            qc.ry(params[k], i)
            k += 1
            
        # GMS_z gate
        for i in range(n):
            for j in range(i+1, n):
                qc.rzz(params[k], i, j)
                k += 1
        # RX gate
        for i in range(0, n):
            qc.rx(params[k], i)
            k += 1

        # GMS_x gate
        for i in range(n):
            for j in range(i+1, n):
                qc.rxx(params[k], i, j)   
                k += 1
        qc.barrier()    
    return qc


def squeezing_ansatz_inv(
    qc: qiskit.QuantumCircuit,
    num_layers: int,
    params: np.ndarray):
    
    """Create squeezing ansatz

    Args:
        qc (qiskit.QuantumCircuit): init circuit
        thetas (np.ndarray): params

    Returns:
        (qiskit.QuantumCircuit): init circuit
    """ 
    
    return squeezing_ansatz(qc,num_layers,params).inverse()

    
#@np.vectorize
def create_num_params(qc_name, num_qubits, num_layers):
    """ to create a list of number of parameters:
                    
    Args:
        - qc_name: qc name
        - num_qubtis: number of qubits
        - num_layers: list of layers for these ansatzes
  
    Returns:
        - total number of paramaters
    """
          
    if qc_name == 'star':
        num_params = num_layers*(2*num_qubits - 2)
    elif qc_name == 'ring':
        num_params = num_layers*2*num_qubits
    elif qc_name == 'sque':
        num_params = num_layers*num_qubits*(num_qubits+1)  
    #elif qc_name == 'u_phase':
    #    num_params = 3 # fixed
    else:
        #num_params = 1 # one params for y (noise)
        print('add by yourself')  
    
    return int(num_params)
    

def create_params(cirs,coefs,num_qubits,values):
    
    """ to create a list of parameters:
                    
    Args:
        - qc_funcs: list of qc_func, i.e., [create_stargraph_ansatz,create_ringgraph_ansatz]
        - num_layers: list of layers for these ansatzes
        - num_qubtis: number of qubits
        - value: initial value
        
    Returns:
        - list of paramaters [[],[],...]
    """
    
    num_params = create_num_params(cirs, coefs, num_qubits)
    params = []
    
    for i in range(len(values)):
        if values[i] == -1: #quy uoc -1 means random
            params.append(np.random.uniform(0, 2 * np.pi, num_params[i]))    
        else:
            params.append(values[i] * num_params[i])
                
    #add phase to params
    list2str = list(map(lambda f: f.__name__, cirs)) 
    if 'u_phase' in list2str:
        idx = list2str.index('u_phase')  
        params[idx] = values[idx]
    
    return params 


#
# quantum states
#

def state_vector(qc: qiskit.QuantumCircuit):
    # to get the state vector from qc  
    return qiskit.quantum_info.Statevector.from_instruction(qc.copy()).data


def state_density(qc: qiskit.QuantumCircuit):
    # to get the state densiry from qc
    return qiskit.quantum_info.DensityMatrix.from_instruction(qc.copy()).data


def ghz(qc: qiskit.QuantumCircuit, *args, **kwargs):
    
    """Create GHZ state 

    Args:
        - qc (QuantumCircuit): Init circuit
        - *args, **kwargs: nothing

    Returns:
        - QuantumCircuit: the added circuit
    """

    qc.h(0)
    
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(0, i + 1)
        
    return qc


def star_graph(qc: qiskit.QuantumCircuit, *args, **kwargs):
    
    """Create start graph 

    Args:
        - qc (QuantumCircuit): Init circuit
        - *args, **kwargs: None

    Returns:
        - QuantumCircuit: the added circuit
    """
    
    num_qubits = qc.num_qubits
    for i in range(0, num_qubits):
        qc.h(i)
    for i in range(1, num_qubits):
        qc.cz(0, i)
    qc.barrier()   
    
    return qc

def ring_graph(qc: qiskit.QuantumCircuit, *args, **kwargs):
    
    n = qc.num_qubits

    for i in range(0, n - 1, 2):
        qc.cz(i, i + 1)
    if n % 2 == 1:
        for i in range(1, n - 1, 2):
            qc.cz(i, i + 1)
        qc.cz(0, n - 1)
        qc.barrier()
    return qc

def w(qc, *args, **kwargs):
    """Create n-qubit W state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    
    qc.x(0)
    qc = w_sub(qc, qc.num_qubits)
    return qc


def w3(circuit: qiskit.QuantumCircuit, qubit: int):
    """Create W state for 3 qubits

    Args:
        - circuit (qiskit.QuantumCircuit): added circuit
        - qubit (int): the index that w3 circuit acts on

    Returns:
        - qiskit.QuantumCircuit: added circuit
    """
    qc = qiskit.QuantumCircuit(3)
    theta = np.arccos(1 / np.sqrt(3))
    qc.cf(theta, 0, 1)
    qc.cx(1, 0)
    qc.ch(1, 2)
    qc.cx(2, 1)
    w3 = qc.to_gate(label='w3')
    # Add the gate to your circuit which is passed as the first argument to cf function:
    circuit.append(w3, [qubit, qubit + 1, qubit + 2])
    return circuit

def w_sub(qc: qiskit.QuantumCircuit, num_qubits: int, shift: int = 0):
    """The below codes is implemented from [this paper](https://arxiv.org/abs/1606.09290)
    \n Simplest case: 3 qubits. <img src='../images/general_w.png' width = 500px/>
    \n General case: more qubits. <img src='../images/general_w2.png' width = 500px/>

    Args:
        - num_qubits (int): number of qubits
        - shift (int, optional): begin wire. Defaults to 0.

    Raises:
        - ValueError: When the number of qubits is not valid

    Returns:
        - qiskit.QuantumCircuit
    """
    #num_qubits = qc.num_qubits
    
    if num_qubits < 2:
        raise ValueError('W state must has at least 2-qubit')
    if num_qubits == 2:
        # |W> state ~ |+> state
        qc.h(0)
        return qc
    if num_qubits == 3:
        # Return the base function
        qc.w3(shift)
        return qc
    else:
        # Theta value of F gate base on the circuit that it acts on
        theta = np.arccos(1 / np.sqrt(qc.num_qubits - shift))
        qc.cf(theta, shift, shift + 1)
        # Recursion until the number of qubits equal 3
        w_sub(qc, num_qubits - 1, qc.num_qubits - (num_qubits - 1))
        for i in range(1, num_qubits):
            qc.cnot(i + shift, shift)
    return qc

def w_inv(qc: qiskit.QuantumCircuit):
    """Create n-qubit W state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc1.x(0)
    qc1 = w_sub(qc1, qc.num_qubits)
    qc = qc.combine(qc1.inverse())
    return qc


def cf(qc: qiskit.QuantumCircuit, theta: float, qubit1: int, qubit2: int):
    """Add Controlled-F gate to quantum circuit

    Args:
        - qc (qiskit.QuantumCircuit): ddded circuit
        - theta (float): arccos(1/sqrt(num_qubits), base on number of qubit
        - qubit1 (int): control qubit
        - qubit2 (int): target qubit

    Returns:
        - qiskit.QuantumCircuit: Added circuit
    """
    cf = qiskit.QuantumCircuit(2)
    u = np.array([[1, 0, 0, 0], [0, np.cos(theta), 0,
                            np.sin(theta)], [0, 0, 1, 0],
                [0, np.sin(theta), 0, -np.cos(theta)]])
    cf.unitary(u, [0, 1])
    cf_gate = cf.to_gate(label='CF')
    qc.append(cf_gate, [qubit1, qubit2])
    return qc

qiskit.QuantumCircuit.w3 = w3
qiskit.QuantumCircuit.cf = cf


#
# measurements
#

def measure(qc: qiskit.QuantumCircuit, qubits, cbits = []):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: Frequency of 00..0 cbit
    """    
    n = len(qubits)
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, n):
        qc.measure(qubits[i], cbits[i])
    
    counts = qiskit.execute(
            qc, backend = vqa.constants.backend,
            shots = vqa.constants.num_shots).result().get_counts()

    return counts.get("0" * len(qubits), 0) / vqa.constants.num_shots


def measure_theor(qc: qiskit.QuantumCircuit, qubits, cbits = []):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: Frequency of 00..0 cbit by Born rule
    """    
    
    nshorts = 10000000
    n = len(qubits)
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, n):
        qc.measure(qubits[i], cbits[i])
    
    counts = qiskit.execute(
            qc, backend = vqa.constants.backend,
            shots = nshorts).result().get_counts()

    return counts.get("0" * len(qubits), 0) / nshorts


def measure_all(qc: qiskit.QuantumCircuit):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: all cbits
    """
    
    n = qc.num_qubits
    qubits = qc.qubits
    cbits = qc.clbits
    for i in range(0, n):
        qc.measure(qubits[i], cbits[i])
    #qc.measure_all()

    shots = vqa.constants.num_shots
    counts = qiskit.execute(
            qc, backend=vqa.constants.backend,
            shots=shots).result().get_counts()

    new_counts = dict()
    for i in range(2**n):
        bin_str = bin(i)[2:]
        if len(bin_str) < n:
            bin_str = "0" * (n - len(bin_str)) + bin_str
        new_counts[bin_str] = counts.get(bin_str, 0)
    
    # calculate the probabilities for each bit value
    probs = {}
    for output in new_counts:
        probs[output] = new_counts[output]/(1.0*shots)
    return np.array(list(probs.values()))


def measure_born(qc: qiskit.QuantumCircuit):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: all cbits
    """
    
    n = qc.num_qubits
    d = 2**n
    probs = []
   
    rho = qiskit.quantum_info.DensityMatrix.from_instruction(qc.copy()).data

    for i in range(0, d):
        probs.append(np.real(rho[i,i]))
          
    return  np.array(probs) 

