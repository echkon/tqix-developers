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
from tqix.qx import *
from tqix.pis.util import *
from scipy.sparse import csc_matrix
from tqix.pis import *
import torch 
__all__ =['circuit','sobj',
          'dbx','dicke_ghz']

def circuit(N,**kwargs):
    """create a quantum circuit

    Parameters:
    ----------
    N: particles number
    Return:
    -------
    init_state    
    """
    use_tensor = kwargs.pop('use_tensor', False)
    init_state = kwargs.pop('initial_state', None)
    num_process = kwargs.pop('num_process', None)
    if use_tensor:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = None

    if not init_state:
       j = N/2
       psi = dbx(j,-j) # all spins down
       if use_tensor:
            psi = psi.todense()
            return sobj(torch.tensor(operx(psi)).to(device),N,use_tensor=use_tensor,device=device,num_process=num_process)
       return sobj(operx(psi).tolist(),N,use_tensor=use_tensor,device=device,num_process=num_process) 
    else:
       return sobj(init_state,N,use_tensor=use_tensor,device=device,num_process=num_process)

class sobj(Gates):
    # to crate a spin-object
    def __init__(self,state,N,use_tensor=None,device=None,num_process=None):
        super().__init__()
        self.state = state
        self.N = N
        self.use_tensor = use_tensor
        self.device = device
        self.num_process = num_process 
        
    def print_state(self):
        state = self.state
        return state

def dbx(j,m):
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
    
if __name__ == "__main__":
    N=3
    print('test with dicke_ghz state')
    init_state = dicke_ghz(N)
    qc = circuit(N,init_state)
    state = qc.state
    print(state)
    print(typex(state))

    print('test gate with dicke_ghz state')
    qc.RZ(np.pi/3).RY().RZ(np.pi/4)
    print(qc.state)