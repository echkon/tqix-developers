"""
>>> OneCircuit: One solution for all quantum circuit needs
________________________________
>>> copyright (c) 2024 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""

import numpy as np
import copy
from typing import List, Any

def gradient_func(circuit, 
                cost_func, 
                train_opt, 
                method = 'psr', 
                step_size = None
                )-> List[float]:
    
    """Return the gradient of the cost_func

    Args:
        - circuit (QuantumCircuit): The quantum circuit want to calculate the gradient
        - cost_func: defined cost function
        - train_opt (Numpy array): setting which train

    Returns:
        - Numpy array: The vector of gradient
    """
    
    grad_cost = [] #to reset index
    
    for i in train_opt:
        for j in range(len(circuit[i][3])): 
            if method == 'psr':
                if step_size == None:
                    step_size = np.pi/2
                
                circuit1, circuit2 = copy.deepcopy(circuit), copy.deepcopy(circuit)
                circuit1[i][3][j] += step_size
                circuit2[i][3][j] -= step_size
                
                cost1 = cost_func(circuit1)
                cost2 = cost_func(circuit2)
                
                grad = (cost1 - cost2)/(2*np.sin(step_size))

            elif method == 'theory':
                '''
                This only works for the next case:
                    qc = QuantumCircuit(2)
                    qc.ry(0)
                    qc.rx(y)
                    qc.cx(0,1)
                '''
                return [1,1]
                
            
            elif method == 'two_point':
                # Standard 2-point central difference (O(h²) accuracy)
                if step_size == None:
                    step_size = 10e-3
                    
                circuit1, circuit2 = copy.deepcopy(circuit), copy.deepcopy(circuit)
                circuit1[i][3][j] += step_size
                circuit2[i][3][j] -= step_size
                
                cost1 = cost_func(circuit1)
                cost2 = cost_func(circuit2)
                
                grad = (cost1 - cost2) / (2 * step_size)
                
            elif method == 'four_point':
                # 4-point central difference (O(h⁴) accuracy)
                if step_size == None:
                    step_size = 10e-3
                    
                circuit1, circuit2, circuit3, circuit4 = (copy.deepcopy(circuit) for _ in range(4))
                circuit1[i][3][j] += 2 * step_size
                circuit2[i][3][j] += step_size
                circuit3[i][3][j] -= step_size
                circuit4[i][3][j] -= 2 * step_size
                
                cost1 = cost_func(circuit1)
                cost2 = cost_func(circuit2)
                cost3 = cost_func(circuit3)
                cost4 = cost_func(circuit4)
                
                grad = (-cost1 + 8*cost2 - 8*cost3 + cost4) / (12 * step_size)
                
            elif method == 'five_point':
                # 5-point central difference (O(h⁴) accuracy, more stable)
                if step_size == None:
                    step_size = 10e-3
                    
                circuit1, circuit2, circuit3, circuit4, circuit5 = (copy.deepcopy(circuit) for _ in range(5))
                circuit1[i][3][j] -= 2 * step_size
                circuit2[i][3][j] -= step_size
                circuit3[i][3][j] += 0  # center point
                circuit4[i][3][j] += step_size
                circuit5[i][3][j] += 2 * step_size
                
                cost1 = cost_func(circuit1)
                cost2 = cost_func(circuit2)
                cost3 = cost_func(circuit3)
                cost4 = cost_func(circuit4)
                cost5 = cost_func(circuit5)
                
                grad = (cost1 - 8*cost2 + 8*cost4 - cost5) / (12 * step_size)
            else:
                raise ValueError("No matching method found. Please choose from 'psr', 'theory', 'two_point', 'four_point', or 'file_point'.")
            grad_cost.append(grad)
            
    return grad_cost

# optimizer
def sgd(params, dev_cost, *args, **kwargs):
    """optimizer standard gradient descent
    
    Args:
        - params: parameters
        - dev_cost: gradient of the cost function
        - *args: use for iteration (not use here)
        - **kwargs: for learning rate
    
    Returns:
        - params
    """
    learning_rate = 0.2
    params -= learning_rate*dev_cost
    return params

def adam(params, dev_cost, iteration, learning_rate = 0.2):
    """optimizer adam
    
    Args:
        - params: parameters
        - dev_cost: gradient of the cost function
    
    Returns:
        - params
    """    
    num_params = len(params) #.shape[0]
    #beta1, beta2,epsilon = 0.8, 0.999, 10**(-8)
    beta1, beta2,epsilon = 0.95, 0.9999, 10**(-12)
    
    m, v = list(np.zeros(num_params)), list(
                    np.zeros(num_params))
    
    for i in range(num_params):
        
        # Gradient smoothing
        grad = np.clip(dev_cost[i], -1.0, 1.0)
        
        m[i] = beta1 * m[i] + (1 - beta1) * dev_cost[i]
        v[i] = beta2 * v[i] + (1 - beta1) * dev_cost[i]**2
        mhat = m[i] / (1 - beta1**(iteration + 1))
        vhat = v[i] / (1 - beta2**(iteration + 1))

        update = learning_rate * mhat / (np.sqrt(vhat) + epsilon)
    
        # Additional damping for very small gradients
        if abs(grad) < 1e-8:
            update *= 0.01

        params[i] -= update
        
    return params
    
def adam_high(params, dev_cost, iteration, learning_rate):
    """optimizer adam use for hybrid
    
    Args:
        - params: parameters
        - dev_cost: gradient of the cost function
    
    Returns:
        - params
    """    
    num_params = len(params) #.shape[0]
    beta1, beta2,epsilon = 0.95, 0.9999, 10**(-12)
    
    m, v = list(np.zeros(num_params)), list(
                    np.zeros(num_params))
    
    for i in range(num_params):

        # Gradient smoothing
        grad = np.clip(dev_cost[i], -1.0, 1.0)
        
        m[i] = beta1 * m[i] + (1 - beta1) * dev_cost[i]
        v[i] = beta2 * v[i] + (1 - beta1) * dev_cost[i]**2
        mhat = m[i] / (1 - beta1**(iteration + 1))
        vhat = v[i] / (1 - beta2**(iteration + 1))

        update = learning_rate * mhat / (np.sqrt(vhat) + epsilon)

        # Additional damping for very small gradients
        if abs(grad) < 1e-8:
            update *= 0.01

        params[i] -= update
    
    return params

class AdamHighPrecision:
    def __init__(self, 
                learning_rate=0.001,
                beta1=0.95,
                beta2=0.9999,
                epsilon=1e-12):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        
    def update(self, params, dev_cost, iteration):
        num_params = len(params)
        
        # Initialize momentum and velocity as lists if first iteration
        if self.m is None:
            self.m = [0.0] * num_params
            self.v = [0.0] * num_params
        
        # Process each parameter individually
        for i in range(num_params):
            # Gradient smoothing
            grad = np.clip(dev_cost[i], -1.0, 1.0)
            
            # Update momentum and velocity
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**(iteration + 1))
            v_hat = self.v[i] / (1 - self.beta2**(iteration + 1))
            
            # Compute update with extra precision
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Additional damping for very small gradients
            if abs(grad) < 1e-8:
                update *= 0.1
                
            params[i] -= update
            
        return params

class QuasiNewtonAdam:
    def __init__(self, 
                learning_rate=0.001,
                memory_size=10,
                epsilon=1e-12):
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.s_history = []  # Parameter differences
        self.y_history = []  # Gradient differences
        self.prev_params = None
        self.prev_grads = None
        
    def update(self, params, dev_cost, iteration):
        num_params = len(params)
        
        # Convert to numpy arrays for easier computation
        params_array = np.array(params)
        dev_cost_array = np.array(dev_cost)
        
        if self.prev_params is not None:
            s = params_array - self.prev_params
            y = dev_cost_array - self.prev_grads
            
            # Only store if significant change
            if np.linalg.norm(s) > self.epsilon:
                self.s_history.append(s)
                self.y_history.append(y)
                
                # Keep only recent history
                if len(self.s_history) > self.memory_size:
                    self.s_history.pop(0)
                    self.y_history.pop(0)
        
        # Store current values for next iteration
        self.prev_params = params_array.copy()
        self.prev_grads = dev_cost_array.copy()
        
        # If no history yet, fall back to simple gradient descent
        if not self.s_history:
            for i in range(num_params):
                params[i] -= self.learning_rate * dev_cost[i]
            return params
        
        # Compute quasi-Newton direction
        direction = dev_cost_array.copy()
        
        # L-BFGS two-loop recursion
        alphas = []
        for s, y in zip(reversed(self.s_history), reversed(self.y_history)):
            alpha = np.dot(s, direction) / (np.dot(y, s) + self.epsilon)
            direction = direction - alpha * y
            alphas.insert(0, alpha)
            
        # Scale the initial Hessian approximation
        s = self.s_history[-1]
        y = self.y_history[-1]
        H0 = np.dot(s, y) / (np.dot(y, y) + self.epsilon)
        direction *= H0
        
        # Second loop of L-BFGS
        for s, y, alpha in zip(self.s_history, self.y_history, alphas):
            beta = np.dot(y, direction) / (np.dot(y, s) + self.epsilon)
            direction = direction + (alpha - beta) * s
            
        # Convert back to list and update parameters
        direction = direction.tolist()
        for i in range(num_params):
            params[i] -= self.learning_rate * direction[i]
            
        return params

def adam_newton_hybrid(params, dev_cost, iteration, learning_rate):
    """
    Combined optimizer for higher precision with list parameters and dynamic learning rates.
    
    Args:
        params: list of parameters to optimize
        dev_cost: list of gradients of the cost function
        iteration: current iteration number
        
    Returns:
        updated list of parameters
    """  
            
    # Initialize optimizers with adjusted learning rates
    optimizers = [
        (AdamHighPrecision(learning_rate=learning_rate), 0.7),
        (QuasiNewtonAdam(learning_rate=learning_rate/2), 0.3)
    ]
    
    num_params = len(params)
    updates = []
    weights = []
    
    # Get updates from each optimizer
    for optimizer, weight in optimizers:
        # Create a copy of params list for each optimizer
        params_copy = params.copy()
        update = optimizer.update(params_copy, dev_cost, iteration)
        updates.append(update)
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]
    
    # Combine updates using weighted average
    final_params = [0.0] * num_params
    for i in range(num_params):
        weighted_sum = 0.0
        for update, weight in zip(updates, weights):
            weighted_sum += update[i] * weight
        final_params[i] = weighted_sum
        
    return final_params

