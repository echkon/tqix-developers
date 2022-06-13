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

def circuit(N,use_tensor=None,device=None,*args):
    """create a quantum circuit

    Parameters:
    ----------
    N: particles number
    *args: initial state

    Return:
    -------
    init_state    
    """
    if not args:
       j = N/2
       psi = dbx(j,-j) # all spins down
       if use_tensor:
            psi = psi.todense()
            return sobj(torch.tensor(operx(psi)).to(device),N,use_tensor=use_tensor,device=device)
       return sobj(operx(psi).tolist(),N,use_tensor=use_tensor,device=device) 
    else:
       return sobj(operx(args[0].tolist()),N,use_tensor=use_tensor,device=device)

class sobj(Gates):
    # to crate a spin-object
    def __init__(self,state,N,use_tensor=None,device=None):
        super().__init__()
        self.state = state
        self.N = N
        self.use_tensor = use_tensor
        self.device = device
       
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