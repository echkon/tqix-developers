"""
>>> OneCircuit: One solution for all quantum circuit needs
________________________________
>>> copyright (c) 2024 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""

import numpy as np
import qiskit
from onecircuit.ansatz import create_params
from onecircuit.optimizing import gradient_func, adam, adam_high, adam_newton_hybrid
import pickle


class VariationalCircuit():
    """
    create a variational quantum circuit
    """
    
    def __init__(self, circuit, unitary_names = None):
        """
        Initialize the QuantumSensing object.
        
        Args:
            circuit (list): A list of circuit unitaries, where each stage is defined as
                            [function, num_qubits, additional parameters, optional params].
            stage_names (list, optional): Names of the unitary in `circuit`. Defaults to ['u0', 'u1', ...].
        """
        self.circuit = circuit
        
        if unitary_names is None:
            unitary_names = [f"u{i}" for i in range(len(circuit))]
            
        if len(circuit) != len(unitary_names):
            raise ValueError("The number of circuit unitary must match the number of unitary names.")
        
        # Dynamically create attributes for each stage
        for name, unitary in zip(unitary_names, circuit):
            circuit_unitary = CircuitUnitary(name, *unitary)
            setattr(self, name, circuit_unitary)  # e.g., self.prep = CircuitUnitary(...)
    
    def add_params(self):
        for i in range(len(self.circuit)):
            if self.circuit[i][3] == []:
                self.circuit[i][3] = create_params(self.circuit[i])
    
    def print(self):
        """ print circuit
        
        Args:
            - circuits: list of circuits, num_qubits, num_layers, parameters, ....
        
        Returns:
            - qc: circuit information
        """
        self.add_params()
        qc = QuantumCircuit(self.circuit)
        
        return print(qc)
        
    def fit(self, 
            num_steps = 100, 
            cost_func = None, 
            optimizer = adam, 
            train_opt = [0], 
            method = 'psr', 
            step_size = None,
            learning_rate = 0.5
            ):
        
        costs = []
        self.add_params()
        
        # get params_train
        train_params = []
        for i in train_opt:
            train_params.extend(self.circuit[i][3])
        
        for s in range(0, num_steps):
            dev_cost = gradient_func(self.circuit,cost_func,train_opt,method,step_size) 
            train_params = optimizer(train_params, dev_cost, s, learning_rate)
            
            # update full params
            pre_ind = 0
            for i in train_opt:   # 0 for pre, 1 for phase,...           
                self.circuit[i][3] = train_params[pre_ind:pre_ind+len(self.circuit[i][3])]
                pre_ind += len(self.circuit[i][3])
        
            # update circuit unitary
            for i in range(len(self.circuit)):
                unitary_name = f"u{i}"
                unitary = getattr(self, unitary_name)
                unitary.params = self.circuit[i][3]
                unitary.num_qubits = self.circuit[i][1]  # update num_qubits
                unitary.coef = self.circuit[i][2]  # update coef
                
            # compute cost   
            cost = cost_func(self.circuit)
            
            # update optimizer
            if len(costs) > 2: 
                if costs[-2] < costs[-1] and optimizer == adam_high:
                    optimizer = adam_newton_hybrid
                elif costs[-2] < costs[-1] and optimizer == adam_newton_hybrid:
                    learning_rate *= 0.1 #decay_factor
        
            costs.append(cost)
            print(s, cost) if s % 5 == 0 else None

            # compare costs
            if s > 1:
                if np.abs(costs[s] - costs[s-1]) < 1e-8:
                    print(np.abs(costs[s] - costs[s-1]))
                    break
        
        return self.circuit, costs  
    
    def save(self, fname):
        """_summary_

        Args:
            fname (_type_): file name
        """
        # Save data to a file using pickle
        with open(fname, 'wb') as file:
            pickle.dump(self.circuit, file)

        print(f'Data has been written to {fname}')
    
class CircuitUnitary:
    """
    Represents a single stage in the quantum circuit (e.g., preparation, sensing, noise).
    """
    def __init__(self, name, func, num_qubits, coef, params):
        self.name = name
        self.func = func
        self.num_qubits = num_qubits
        self.coef = coef
        self.params = params or []

    def __repr__(self):
        return (f"QuantumCircuit(name={self.name}, func={self.func.__name__}, "
                f"num_qubits={self.num_qubits}, coef={self.coef}, "
                f"params={self.params})")
        
    def circuit(self):
        """Return the quantum circuit based on the function in the stage."""
        return self.func(self.num_qubits, self.coef, self.params)

def QuantumCircuit(circuit):
        """ create a full circuit from circuit(func)
    
        Args:
            - circuit: 
                - circuit[0] = prep, 
                    - prep[0] = name_qc
                    - prep[1] = num_qubits,
                    - prep[2] = num_payers
                    - prep[3] = params
                - circuit[1] = sensing_qc    
        
        Returns:
            - qc: final circuit
        """
        
        num_qubits = circuit[0][1] 
        qc = qiskit.QuantumCircuit(num_qubits)
        
        #combine all
        for i in range (len(circuit)):
            qc &= circuit[i][0](circuit[i][1], circuit[i][2], circuit[i][3])
        return qc