"""
>>> this file is a part of tqix: a Toolbox for Quantum in X
                                x: quantum measurement, quantum metrology, 
                                quantum tomography, and more.
________________________________
>>> copyright (c) 2019 and later
>>> authors: Binho Le
>>> contributors:
>>> all rights reserved
________________________________
"""

#from numpy import
import numpy as np
from tqix.qobj import *
from tqix.pis.util import *
from scipy.sparse import csc_matrix
from tqix.pis import *
import torch

#__all__ =['circuit','sobj',
#            'dbx','dicke_ghz']

def circuit(N,**kwargs):
    """

    :param N: number of qubits
    :type N: int
    :return: spin state of circuit 
    :rtype: instance of sobj
    """    
    use_gpu = kwargs.pop('use_gpu', False)
    init_state = kwargs.pop('init_state', None)
    num_process = kwargs.pop('num_process', None)
    if use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("your machine does not utilize gpu, the program will use torch tensor type on cpu")
            device = "cpu"
    else:
        device = None

    #if not init_state:
    if init_state is None:
        j = N/2
        psi = dbx(j,-j) # all spins down
        if use_gpu:
            psi = psi.todense()
            return sobj(torch.tensor(operx(psi)).to(device),N,use_gpu=use_gpu,device=device,num_process=num_process)
        return sobj(operx(psi).tolist(),N,use_gpu=use_gpu,device=device,num_process=num_process) 
    else:
        return sobj(init_state,N,use_gpu=use_gpu,device=device,num_process=num_process)

class sobj(Gates):
    """
    class stores state of qubits
    """    
    # to crate a spin-object
    def __init__(self,state,N,use_gpu=None,device=None,num_process=None):
        """

        :param state: state of qubits
        :type state: ndarray,tensor,sparse
        :param N: number of qubits
        :type N: int
        :param use_gpu: if use gpu, defaults to None
        :type use_gpu: bool, optional
        :param device: name of compute device, defaults to None
        :type device: str, optional
        :param num_process: number of processes, defaults to None
        :type num_process: int, optional
        """        
        super().__init__()
        self.state = state
        self.N = N
        self.use_gpu = use_gpu
        self.device = device
        self.num_process = num_process 
        
    def print_state(self):
        state = self.state
        return state

def dbx(j,m):
    """
    create dicke basis with pure state

    input: j,m 
    
    output: a vector basis

    :param j: index j
    :type j: int
    :param m: index m
    :type m: int
    :raises ValueError: j must in bound -j ≤ m ≤ j
    :return: state in dicked basis
    :rtype: ndarray, tensor, sparse
    """    
    # creat dicke basis with pure state
    # input: j,m 
    # output: a vector basis

    if m > j or m < -j:
        raise ValueError('j must in bound -j ≤ m ≤ j')
    
    dim = int(2*j + 1)
    state = np.zeros((dim,1))
    offset = get_vidx(j,m) #get vector's index
    state[offset,0] = 1.0
    return csc_matrix(state)

def dicke_ghz(N):
    """
    :param N: number of qubits
    :type N: int
    :return: state in dicked ghz basis
    :rtype: ndarray, tensor, sparse
    """    
    # this is an example to get ghz state
    # def: GHZ = (|N/2,N/2> + |N/2,-N/2>)√2
        
    j = N/2
    m = j
    m1 = -j
    
    e1 = dicke_bx(N,{(j,m,m):1})
    e2 = dicke_bx(N,{(j,m,m1):1})
    e3 = dicke_bx(N,{(j,m1,m):1})
    e4 = dicke_bx(N,{(j,m1,m1):1})
    
    return csc_matrix(0.5*(e1+e2+e3+e4))
