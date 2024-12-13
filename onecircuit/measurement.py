# -*- coding: utf-8 -*-
""" Everything belong to circuit
"""

import qiskit
import tqix
import numpy as np
from onecircuit.util import meas_method

def QuantumMeasurement(sensor, method = meas_method):
    if method == 'theory':
        return measure_theor(sensor)
    elif method == 'theory0':
        return measure_theor0(sensor)
    elif method == 'experiment':
        return measure_all(sensor)
    else:
        raise ValueError(
            'No measurement method was given')
    
    
def measure(sensor):
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
            qc, backend = tqix.vqa.constants.backend,
            shots = tqix.vqa.constants.num_shots).result().get_counts()

    return counts.get("0" * len(qubits), 0) / tqix.vqa.constants.num_shots


def measure_theor(sensor):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit

    Returns:
        - float: Frequency of 00..0 cbit by Born rule
    """    
    rho = qiskit.quantum_info.DensityMatrix(sensor).data
    pros = []
    for i in range(len(rho)):
        pros.append(np.real(rho[i,i]))
    return np.array(pros)

def measure_theor0(sensor):
    """Measuring the quantum circuit which fully measurement gates

    Args:
        - qc (QuantumCircuit): Measured circuit
        - qubits (Numpy array): List of measured qubit
    
    Returns:
        - float: Frequency of 00..0 cbit by Born rule
    """
    rho = qiskit.quantum_info.DensityMatrix(sensor).data
    prob = np.real(rho[0,0])
    return prob


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

    shots = tqix.vqa.constants.num_shots
    counts = qiskit.execute(
            qc, backend= tqix.vqa.constants.backend,
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

