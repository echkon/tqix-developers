"""The Adam and AMSGRAD optimizers."""

from typing import Optional, Callable, Tuple
import logging
import os
import csv
import numpy as np
from tqix.pis import *
from tqix import * 
from scipy.sparse import block_diag
from scipy import linalg
import torch 
import math 
import time 
logger = logging.getLogger(__name__)

# pylint: disable=invalid-name
__all__ = ['ADAM','GD','gradient_num_diff']

def gradient_num_diff(x_center, f, epsilon, max_evals_grouped=1):
    """

    :param x_center: point around which we compute the gradient
    :type x_center: ndarray
    :param f: the function of which the gradient is to be computed.
    :type f: func
    :param epsilon: the epsilon used in the numeric differentiation.
    :type epsilon: float
    :param max_evals_grouped: max evals grouped, defaults to 1
    :type max_evals_grouped: int, optional
    :return: the gradient computed
    :rtype: ndarray
    """    

    # forig = f(*((x_center,)))

    grad = []
    ei = np.zeros((len(x_center),), float)
    todos_plus = []
    for k in range(len(x_center)):
        ei[k] = 1.0
        d = epsilon * ei
        todos_plus.append(x_center + d)
        ei[k] = 0.0

    todos_minus = []
    for k in range(len(x_center)):
        ei[k] = 1.0
        d = epsilon * ei
        todos_minus.append(x_center - d)
        ei[k] = 0.0

    counter = 0
    chunk_plus = []
    chunks_plus = []
    length = len(todos_plus)
    # split all points to chunks, where each chunk has batch_size points
    for i in range(length):
        x = todos_plus[i]
        chunk_plus.append(x)
        counter += 1
        # the last one does not have to reach batch_size
        if counter == max_evals_grouped or i == length - 1:
            chunks_plus.append(chunk_plus)
            chunk_plus = []
            counter = 0
    
    counter = 0
    chunk_minus = []
    chunks_minus = []
    length = len(todos_minus)
    # split all points to chunks, where each chunk has batch_size points
    for i in range(length):
        x = todos_minus[i]
        chunk_minus.append(x)
        counter += 1
        # the last one does not have to reach batch_size
        if counter == max_evals_grouped or i == length - 1:
            chunks_minus.append(chunk_minus)
            chunk_minus = []
            counter = 0

    for chnk_pl,chnk_mn in zip(chunks_plus,chunks_minus):  # eval the chunks in order
        parallel_parameters_plus = np.concatenate(chnk_pl)
        todos_results_plus = f(parallel_parameters_plus)  # eval the points in a chunk (order preserved)
        parallel_parameters_minus = np.concatenate(chnk_mn)
        todos_results_minus = f(parallel_parameters_minus)
        if isinstance(todos_results_plus, float) and isinstance(todos_results_minus, float):
            grad.append((todos_results_plus - todos_results_minus) / (2*epsilon) )
        else:
            for todor_pl,todor_ms in zip(todos_results_plus,todos_results_minus):
                grad.append((todor_pl - todor_ms) / (2*epsilon) )

    return np.array(grad)

class GD:
    """
        Gradient descent optimizer class (ref. qiskit)
    """    
    def __init__(self,lr: float,eps : float,maxiter: int,tol: float = 1e-6,use_qng=False,route=None,N=None,theta=None):
        """

        :param lr: learning rate
        :type lr: float
        :param eps: Value >=0, Epsilon to be used for finite differences if no analytic gradient method is given.
        :type eps: float
        :param maxiter: maximum iterations
        :type maxiter: int
        :param tol: tolerance for termination, defaults to 1e-6
        :type tol: float, optional
        :param use_qng: option to use quantum gradient descent, defaults to False
        :type use_qng: bool, optional
        :param route: the structure of circuit; for example, if we have a circuit which
        has route=(("RN2",),("OAT","Z"),("TNT","ZX"),("TAT","ZY")) , defaults to None
        :type route: Tuple[Tuple[str,str]], optional
        :param N: number of qubits, defaults to None
        :type N: int, optional
        :param theta: list of theta angle, defaults to None
        :type theta: List[float], optional
        """        
        self._maxiter = maxiter
        self._t = 0
        self.step_size = lr
        self._eps = eps
        self._tol = tol 
        self.use_qng = use_qng
        if self.use_qng:
            self.route = route
            self.N = N 
            self.theta = theta

    def calc_fubini_tensor(self,params):
        """
        We calculate fubini tensor (note currently only apply for OAT,TNT,TAT gates)

        :param params: list of parameters
        :type params: List
        :return: G - fubini tensor
        :rtype: ndarray, torch, sparse
        """        
        num_layers = len(self.route)-1 
        feature_map = self.route[0][0]
        if torch.is_tensor(params):
            qc = circuit(self.N,use_gpu=True,device=params.device)
            if feature_map == "RN2":
                qc.RN(np.pi/2,0) 
            gs = []
            for num in range(num_layers):
                type = self.route[num+1][1]
                if self.route[num+1][0] == "OAT":
                    g = torch.real(qc.expval("OAT'"+type+"2") - qc.expval("OAT'"+type)**2)
                    qc.OAT(params[num],type)
                elif self.route[num+1][0] == "TAT":
                    g = torch.real(qc.expval("TAT'"+type+"2") - qc.expval("TAT'"+type)**2)
                    qc.TAT(params[num],type)
                elif self.route[num+1][0] == "TNT":
                    g = torch.real(qc.expval("TNT'"+type+"2") - qc.expval("TNT'"+type)**2)
                    qc.TNT(params[num],omega=params[num]*self.N/2,gate_type=type)
                gs.append(g)
            G = torch.block_diag(*gs)
        else:
            qc = circuit(self.N)
            if feature_map == "RN2":
                qc.RN(np.pi/2,0) 
            gs = []
            for num in range(num_layers):
                type = self.route[num+1][1]
                if self.route[num+1][0] == "OAT":
                    g = np.real(qc.expval("OAT'"+type+"2") - qc.expval("OAT'"+type)**2)
                    qc.OAT(params[num],type)
                elif self.route[num+1][0] == "TAT":
                    g = np.real(qc.expval("TAT'"+type+"2") - qc.expval("TAT'"+type)**2)
                    qc.TAT(params[num],type)
                elif self.route[num+1][0] == "TNT":
                    g = np.real(qc.expval("TNT'"+type+"2") - qc.expval("TNT'"+type)**2)
                    qc.TNT(params[num],omega=params[num]*self.N/2,gate_type=type)
                gs.append(g)
            G = block_diag(gs,format="csc")
        return G 

    def optimize(self, num_vars: int, objective_function: Callable[[np.ndarray], float],
                 gradient_function: Optional[Callable[[np.ndarray], float]] = None,
                 initial_point: Optional[np.ndarray] = None,
                 return_loss_hist=None,loss_break=None,return_time_iters=None, 
                 ) -> Tuple[np.ndarray, float, int]:
        """
        :param num_vars: Number of parameters to be optimized.
        :type num_vars: int
        :param objective_function: Handle to a function that computes the objective function.
        :type objective_function: func
        :param gradient_function: Handle to a function that computes the gradient of the objective function., defaults to None
        :type gradient_function: func, optional
        :param initial_point: The initial point for the optimization., defaults to None
        :type initial_point: Optional[np.ndarray], optional
        :param return_loss_hist: return history of loss values, defaults to None
        :type return_loss_hist: bool, optional
        :param loss_break: early stopping, defaults to None
        :type loss_break: bool, optional
        :param return_time_iters: return time each iterations, defaults to None
        :type return_time_iters: bool, optional
        :return: output
        :rtype: Tuple[np.ndarray, float, int]
        """        
        if torch.is_tensor(initial_point):
            pass
        else:
            if initial_point is None:
                initial_point = np.random.default_rng(52).random(num_vars)
            if gradient_function is None:
                gradient_function = lambda params: gradient_num_diff(params,objective_function, self._eps)

        output = self.minimize(objective_function, initial_point, gradient_function,return_loss_hist,return_time_iters,loss_break)
        return output
        
    
    def minimize(self, objective_function: Callable[[np.ndarray], float], initial_point: np.ndarray,
                 gradient_function: Callable[[np.ndarray], float],return_loss_hist:bool,return_time_iters:bool, loss_break:bool) -> Tuple[np.ndarray, float, int]:
        """Run the minimization.

        :param objective_function: A function handle to the objective function.
        :type objective_function: func
        :param initial_point: The initial iteration point.
        :type initial_point: np.ndarray
        :param gradient_function: A function handle to the gradient of the objective function.
        :type gradient_function: func
        :param return_loss_hist: return loss history
        :type return_loss_hist: bool
        :param return_time_iters: return time each iterations
        :type return_time_iters: bool
        :param loss_break: early stopping
        :type loss_break: bool
        :return: A tuple of (optimal parameters, optimal value, number of iterations).
        :rtype: Tuple[np.ndarray, float, int]
        """        
        loss_history = []
        if torch.is_tensor(initial_point):
            tensor_times = []
            objective_function(initial_point).backward()
            derivative = initial_point.grad
            params = params_new = initial_point
            while self._t < self._maxiter:
                start = time.time()
                if self._t > 0:
                    params = params.detach().requires_grad_()
                    objective_function(params).backward()
                    derivative = params.grad
                self._t += 1
                if self.use_qng:
                    G = self.calc_fubini_tensor(params)
                    pinv_G = torch.linalg.pinv(G)
                    params_new = params - self.step_size * pinv_G @ derivative.type(torch.DoubleTensor).to(pinv_G.device)
                else: 
                    params_new = params - self.step_size * derivative
                new_loss = objective_function(params_new)
                loss_history.append(new_loss.item())               
                # print("derivative,params_new,new_loss:",derivative,params_new,new_loss)
                end = time.time()-start
                tensor_times.append(end)
                if torch.linalg.norm(params - params_new) < self._tol:
                    if return_loss_hist or return_time_iters:
                            return params_new, new_loss, self._t, loss_history,tensor_times
                    return params_new,new_loss, self._t 
                elif loss_break != None:
                    if new_loss < loss_break:
                        if return_loss_hist or return_time_iters:
                            return params_new, new_loss, self._t, loss_history, sparse_times
                        return params_new,new_loss, self._t
                    else:
                        params = params_new
                else:
                    params = params_new
            if return_loss_hist or tensor_times:
                        return params_new,new_loss, self._t, loss_history,return_time_iters
            return params_new,new_loss, self._t

        else:
            derivative = gradient_function(initial_point)
            params = params_new = initial_point
            sparse_times = []
            while self._t < self._maxiter:
                start = time.time()
                if self._t > 0:
                    derivative = gradient_function(params)
                self._t += 1
                if self.use_qng:
                    G = self.calc_fubini_tensor(params)
                    pinv_G = linalg.pinv(G.toarray())
                    params_new = params - self.step_size * pinv_G.dot(derivative) 
                else: 
                    params_new = params - self.step_size * derivative
                new_loss = objective_function(params_new)
                loss_history.append(new_loss)
                # print("derivative,params,loss:",derivative,params_new,new_loss)
                end = time.time() - start 
                sparse_times.append(end)
                if np.linalg.norm(params - params_new) < self._tol:
                    if return_loss_hist or return_time_iters:
                            return params_new,new_loss, self._t, loss_history,sparse_times
                    return params_new,new_loss, self._t
                elif loss_break != None:
                    if new_loss < loss_break:
                        if return_loss_hist or return_time_iters:
                            return params_new,new_loss, self._t, loss_history, sparse_times
                        return params_new,new_loss, self._t
                    else:
                        params = params_new
                else:
                    params = params_new
            
            if return_loss_hist or return_time_iters:
                    return params_new,new_loss, self._t, loss_history , sparse_times 
            return params_new,new_loss, self._t

class ADAM:
    """Adam and AMSGRAD optimizers.
    (ref. qiskit)
    """    

    _OPTIONS = ['maxiter', 'tol', 'lr', 'beta_1', 'beta_2',
                'noise_factor', 'eps', 'amsgrad', 'snapshot_dir']

    def __init__(self,
                 maxiter: int = 10000,
                 tol: float = 1e-6,
                 lr: float = 1e-3,
                 beta_1: float = 0.9,
                 beta_2: float = 0.99,
                 noise_factor: float = 1e-8,
                 eps: float = 1e-10,
                 amsgrad: bool = False,
                 snapshot_dir: Optional[str] = None) -> None:
        """

        :param maxiter: Maximum number of iterations, defaults to 10000
        :type maxiter: int, optional
        :param tol: Tolerance for termination, defaults to 1e-6
        :type tol: float, optional
        :param lr: Learning rate, defaults to 1e-3
        :type lr: float, optional
        :param beta_1: Value in range 0 to 1, Generally close to 1., defaults to 0.9
        :type beta_1: float, optional
        :param beta_2: Value in range 0 to 1, Generally close to 1., defaults to 0.99
        :type beta_2: float, optional
        :param noise_factor: Value >= 0, Noise factor, defaults to 1e-8
        :type noise_factor: float, optional
        :param eps: Value >=0, Epsilon to be used for finite differences if no analytic
                gradient method is given., defaults to 1e-10
        :type eps: float, optional
        :param amsgrad: True to use AMSGRAD, defaults to False
        :type amsgrad: bool, optional
        :param snapshot_dir: If not None save the optimizer's parameter after every step to the given directory, defaults to None
        :type snapshot_dir: Optional[str], optional
        """        
        super().__init__()
        self._maxiter = maxiter
        self._snapshot_dir = snapshot_dir
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad

        # runtime variables
        self._t = 0  # time steps
        self._m = np.zeros(1)
        self._v = np.zeros(1)
        if self._amsgrad:
            self._v_eff = np.zeros(1)

        if self._snapshot_dir:

            with open(os.path.join(self._snapshot_dir, 'adam_params.csv'),
                      mode='w', encoding="utf8") as csv_file:
                if self._amsgrad:
                    fieldnames = ['v', 'v_eff', 'm', 't']
                else:
                    fieldnames = ['v', 'm', 't']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

    def save_params(self, snapshot_dir: str) -> None:
        """
        Save the current iteration parameters to a file called ``adam_params.csv``.
        
        Note:
        
            The current parameters are appended to the file, if it exists already.

            The file is not overwritten.

        :param snapshot_dir: The directory to store the file in.
        :type snapshot_dir: str
        """        
        if self._amsgrad:
            with open(os.path.join(snapshot_dir, 'adam_params.csv'),
                      mode='a', encoding="utf8") as csv_file:
                fieldnames = ['v', 'v_eff', 'm', 't']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'v': self._v, 'v_eff': self._v_eff,
                                 'm': self._m, 't': self._t})
        else:
            with open(os.path.join(snapshot_dir, 'adam_params.csv'),
                      mode='a', encoding="utf8") as csv_file:
                fieldnames = ['v', 'm', 't']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'v': self._v, 'm': self._m, 't': self._t})

    def load_params(self, load_dir: str) -> None:
        """Load iteration parameters for a file called ``adam_params.csv``.

        :param load_dir: The directory containing ``adam_params.csv``
        :type load_dir: str
        """        

        with open(os.path.join(load_dir, 'adam_params.csv'),
                  mode='r', encoding="utf8") as csv_file:
            if self._amsgrad:
                fieldnames = ['v', 'v_eff', 'm', 't']
            else:
                fieldnames = ['v', 'm', 't']
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            for line in reader:
                v = line['v']
                if self._amsgrad:
                    v_eff = line['v_eff']
                m = line['m']
                t = line['t']

        v = v[1:-1]
        self._v = np.fromstring(v, dtype=float, sep=' ')
        if self._amsgrad:
            v_eff = v_eff[1:-1]
            self._v_eff = np.fromstring(v_eff, dtype=float, sep=' ')
        m = m[1:-1]
        self._m = np.fromstring(m, dtype=float, sep=' ')
        t = t[1:-1]
        self._t = np.fromstring(t, dtype=int, sep=' ')

    def minimize(self, objective_function: Callable[[np.ndarray], float], initial_point: np.ndarray,
                 gradient_function: Callable[[np.ndarray], float],return_loss_hist:bool,return_time_iters:bool,loss_break:int) -> Tuple[np.ndarray, float, int]:
        """

        :param objective_function: A function handle to the objective function.
        :type objective_function: func
        :param initial_point: The initial iteration point.
        :type initial_point: np.ndarray
        :param gradient_function: A function handle to the gradient of the objective function.
        :type gradient_function: func
        :param return_loss_hist: return history of losses
        :type return_loss_hist: bool
        :param return_time_iters: return time each iterations
        :type return_time_iters: bool
        :param loss_break: early stopping
        :type loss_break: int
        :return: output
        :rtype: Tuple[np.ndarray, float, int]
        """        
        loss_history = []
        if torch.is_tensor(initial_point):
            tensor_times = []
            objective_function(initial_point).backward()
            derivative = initial_point.grad
            self._t = 0
            self._m = torch.zeros(derivative.size()).to(initial_point.device)
            self._v = torch.zeros(derivative.size()).to(initial_point.device)    
            if self._amsgrad:
                self._v_eff = torch.zeros(derivative.size()).to(initial_point.device)
            params = params_new = initial_point
            while self._t < self._maxiter:
                start = time.time()
                if self._t > 0:
                    params = params.detach().requires_grad_()
                    objective_function(params).backward()
                    derivative = params.grad
                self._t += 1
                self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
                self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
                lr_eff = self._lr * math.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
                if not self._amsgrad:
                    params_new = (params - lr_eff * self._m.flatten()
                                / (torch.sqrt(self._v.flatten()) + self._noise_factor))
                else:
                    self._v_eff = torch.maximum(self._v_eff, self._v)
                    params_new = (params - lr_eff * self._m.flatten()
                                / (torch.sqrt(self._v_eff.flatten()) + self._noise_factor))
                new_loss = objective_function(params_new)
                loss_history.append(new_loss.item())
                # print("derivative,params,new_loss:",derivative,params_new,new_loss)
                end = time.time() - start 
                tensor_times.append(end)
                if self._snapshot_dir:
                    self.save_params(self._snapshot_dir)
                if torch.linalg.norm(params - params_new) < self._tol:
                    if return_loss_hist or return_time_iters:
                        return params_new, new_loss, self._t, loss_history, tensor_times
                    return params_new,new_loss, self._t
                if loss_break != None:
                    if new_loss < loss_break:
                        if return_loss_hist or return_time_iters:
                            return params_new, new_loss, self._t, loss_history, tensor_times
                        return params_new,new_loss, self._t
                    else:
                        params = params_new
                else:
                    params = params_new

            if return_loss_hist or return_time_iters:
                        return params_new, new_loss, self._t, loss_history, tensor_times
            return params_new, new_loss, self._t


        else:
            derivative = gradient_function(initial_point)
            sparse_times = []
            self._t = 0
            self._m = np.zeros(np.shape(derivative))
            self._v = np.zeros(np.shape(derivative))
            if self._amsgrad:
                self._v_eff = np.zeros(np.shape(derivative))

            params = params_new = initial_point
            while self._t < self._maxiter:
                start = time.time()
                if self._t > 0:
                    derivative = gradient_function(params)
                self._t += 1
                self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
                self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
                lr_eff = self._lr * np.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
                if not self._amsgrad:
                    params_new = (params - lr_eff * self._m.flatten()
                                / (np.sqrt(self._v.flatten()) + self._noise_factor))
                else:
                    self._v_eff = np.maximum(self._v_eff, self._v)
                    params_new = (params - lr_eff * self._m.flatten()
                                / (np.sqrt(self._v_eff.flatten()) + self._noise_factor))
                new_loss = objective_function(params_new)
                loss_history.append(new_loss)
                # print("derivative,params,new_loss:",derivative,params_new,new_loss)
                end = time.time() - start 
                sparse_times.append(end)
                if self._snapshot_dir:
                    self.save_params(self._snapshot_dir)
                if np.linalg.norm(params - params_new) < self._tol:
                    if return_loss_hist or return_time_iters:
                        return params_new,new_loss, self._t, loss_history, sparse_times
                    return params_new,new_loss, self._t
                if loss_break != None:
                    if new_loss < loss_break:
                        if return_loss_hist or return_time_iters:
                            return params_new,new_loss, self._t, loss_history, sparse_times
                        return params_new,new_loss, self._t
                    else:
                        params = params_new
                else:
                    params = params_new

            if return_loss_hist or return_time_iters:
                        return params_new,new_loss, self._t, loss_history, sparse_times
            return params_new,new_loss, self._t

    def optimize(self, num_vars: int, objective_function: Callable[[np.ndarray], float],
                 gradient_function: Optional[Callable[[np.ndarray], float]] = None,
                 initial_point: Optional[np.ndarray] = None, return_loss_hist = False,
                 loss_break = None,return_time_iters=None
                 ) -> Tuple[np.ndarray, float, int]:
        """Perform optimization.

        :param num_vars: Number of parameters to be optimized.
        :type num_vars: int
        :param objective_function: Handle to a function that computes the objective function.
        :type objective_function: func
        :param gradient_function: Handle to a function that computes the gradient of the objective function, defaults to None
        :type gradient_function: func, optional
        :param initial_point: The initial point for the optimization, defaults to None
        :type initial_point: Optional[np.ndarray], optional
        :param return_loss_hist: return history of loss values, defaults to False
        :type return_loss_hist: bool, optional
        :param loss_break: early stopping, defaults to None
        :type loss_break: bool, optional
        :param return_time_iters: return time each iterations, defaults to None
        :type return_time_iters: bool, optional
        :return: output
        :rtype: Tuple[np.ndarray, float, int]
        """        
        if initial_point is None:
            initial_point = np.random.default_rng(52).random(num_vars)
        if torch.is_tensor(initial_point):
            pass
        else:
            if gradient_function is None:
                gradient_function = lambda params: gradient_num_diff(params,objective_function, self._eps)
        output = self.minimize(objective_function, initial_point, gradient_function,return_loss_hist,return_time_iters,loss_break)
        return output