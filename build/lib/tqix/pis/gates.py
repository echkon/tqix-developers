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
from tqix.qx import *
from tqix.qtool import dotx
from tqix.pis.util import *
from tqix.pis import *

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

    state = sobj.state
    d = shapex(state)[0]
    Nds = get_Nds(d) #cannot use for pure
    Nin = sobj.N #get from input

    if not isoperx(state):
        j = (d-1)/2
        new_state = np.zeros((d,1),dtype = complex)
        for idx in np.nonzero(state)[0]:
            m = int(j - idx)
            new_state[idx,0] = cm.exp(-1j*theta*m)*state[idx,0]
        sobj.state = new_state
    else:
        if Nin != Nds: #not full blocks
           j = Nin/2
           new_state = np.zeros((d,d),dtype = complex)
           iks = _get_non0_idx(state)
           mm1 = get_mm1_idx_max(Nin)[0]
           for ik in iks:
              (m, m1) = mm1[ik]
              new_state[ik] = state[ik]*cm.exp(-1j*theta*(m-m1)) 
           sobj.state = new_state
        else: #full blocks
           new_state = np.zeros((d,d),dtype = complex)
           iks = _get_non0_idx(state)
           jmm1 = get_jmm1_idx(Nds)[0] 
           for ik in iks:
               (j,m,m1) = jmm1[ik]
               new_state[ik] = state[ik]*cm.exp(-1j*theta*(m-m1))
           sobj.state = new_state

def _get_non0_idx(matrix):
    """get non zero indexs of a matrix"""
    lidx = []
    for i,j in enumerate(matrix):
        for k,l in enumerate(j):
          if l != 0.0:
             lidx.append((i,k))
    return lidx
