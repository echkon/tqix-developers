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
import cmath as cm

from sympy import csc
from tqix.qx import *
from tqix.qtool import dotx
from tqix.pis.util import *
from tqix.pis import *
from scipy.sparse import bsr_matrix,block_diag,csc_matrix
from scipy.sparse.linalg import expm

__all__ = ['RZ']

def RZ(sobj,theta):
    """ collective rotation gate around the z-axis
    RZ = expm(-i*theta*Jz)|state>

    Parameters
    ----------
    theta: rotation angle
    state: quantum state
    
    Return
    ----------
    new state
    """

    state = sobj.state.toarray()
    d = shapex(state)[0]
    Nds = get_Nds(d) #cannot use for pure
    Nin = sobj.N #get from input

    if not isoperx(state):
        j = (d-1)/2
        new_state = np.zeros((d,1),dtype = complex)
        for idx in np.nonzero(state)[0]:
            m = int(j - idx)
            new_state[idx,0] = cm.exp(-1j*theta*m)*state[idx,0]
    else:
        if Nin != Nds: #not full blocks
           j = Nin/2
           new_state = np.zeros((d,d),dtype = complex)
           iks = _get_non0_idx(state)
           mm1 = get_mm1_idx_max(Nin)[0]
           for ik in iks:
              (m, m1) = mm1[ik]
              new_state[ik] = state[ik]*cm.exp(-1j*theta*(m-m1)) 
        else: #full blocks
           new_state = np.zeros((d,d),dtype = complex)
           iks = _get_non0_idx(state)
           jmm1 = get_jmm1_idx(Nds)[0] 
           for ik in iks:
               (j,m,m1) = jmm1[ik]
               new_state[ik] = state[ik]*cm.exp(-1j*theta*(m-m1))
    sobj.state = bsr_matrix(new_state)

def _get_non0_idx(matrix):
    """get non zero indexs of a matrix"""
    lidx = []
    for i,j in enumerate(matrix):
        for k,l in enumerate(j):
          if l != 0.0:
             lidx.append((i,k))
    return lidx

def RZ_sparse(sobj,theta):
    state = sobj.state
    d_in = shapex(state)[0]
    N_in = sobj.N
    d_dicke = get_dim(N_in)

    if d_in != d_dicke:
        J_z = Sz(N_in/2)

    else:
        j_array = get_jarray(N_in)[::-1]
        blocks = []
        for j in j_array:
            blocks.append(Sz(j))
        J_z = block_diag(blocks)

    J_z = csc_matrix(J_z)
    expJ = expm(-1j*theta*J_z)
    expJ_conj = csc_matrix(daggx(expJ))
    new_state = expJ.dot(sobj.state).dot(expJ_conj)

    sobj.state = new_state
 
if __name__ == "__main__":
    N=3

    print('test circuit')
    qc = circuit(N)
    state = qc.state
    print(state)
    print(typex(state))

    print('test initial state')
    init_state = dotx(state,daggx(state))
    qc = circuit(N,init_state)
    state = qc.state
    print(state)
    print(typex(state))

    print('test gate with initial state')
    RZ(qc,np.pi/3)
    print(qc.state)
