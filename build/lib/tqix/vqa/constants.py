import qiskit
import numpy as np

num_shots = 10000
learning_rate = 0.2
backend = qiskit.Aer.get_backend('qasm_simulator')
step_size = 0.000001

generator = {
    'cu': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'rx': -1 / 2 * np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'crx': -1 / 2 * np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'ry': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'cry': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'rz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'crz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'i': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    'id': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    '11': np.array([[0, 0], [0, 0]], dtype=np.complex128)
}

ignore_generator = [
    'barrier'
]
parameterized_generator = [
    'rx', 'ry', 'rz', 'crx', 'cry', 'crz'
]
# This information is extracted from http://dx.doi.org/10.1103/PhysRevA.83.042314
edges_graph_state = {
    2: ["0-1"],
    3: ["0-1", "0-2"],
    4: ["0-2", "1-3", "2-3"],
    5: ["0-3", "2-4", "1-4", "2-3"],
    6: ["0-4", "1-5", "2-3", "2-4", "3-5"],
    7: ["0-5", "2-3", "4-6", "1-6", "2-4", "3-5"],
    8: ["0-7", "1-6", "2-4", "3-5", "2-3", "4-6", "5-7"],
    9: ["0-8", "2-3", "4-6", "5-7", "1-7", "2-4", "3-5", "6-8"],
    10: ["0-9", "1-8", "2-3", "4-6", "5-7", "2-4", "3-5", "6-9", "7-8"]
}

def lamb2y(t,lamb):
    """calculate gamma from lambda
    Args:
        - t: time
        - lamb: lambda
    Returs:
        - gamma
    """
    return np.abs(-np.log(1.0-lamb)/t) #use abs to avoid -0.
        