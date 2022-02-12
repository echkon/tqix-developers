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
from functools import partial
import numpy as np
import cmath as cm

from sympy import csc
from tqix.qx import *
from tqix.qtool import dotx
from tqix.pis.util import *
from tqix.pis import *
from scipy.sparse import bsr_matrix,block_diag,csc_matrix
from scipy.sparse.linalg import expm
from functools import *

__all__ = ['Gates']

class Gates(object):
    """ collective rotation gate around axis
        R = expm(-i*theta*J)|state>expm(-i*theta*J).T

        Parameters
        ----------
        theta: rotation angle
        state: quantum state

        Return
        ----------
        new state
    """
    def __init__(self):
        self.state = None        
        self.theta = None

    def RX(self,theta=None):
        self.check_input_param(theta)
        return self.gates("Rx")
    
    def RY(self,theta=None):
        self.check_input_param(theta) 
        return self.gates("Ry")
    
    def RZ(self,theta=None):
        self.check_input_param(theta)
        return self.gates("Rz")
    
    def RX2(self,theta=None):
        self.check_input_param(theta)
        return self.gates("Rx2")
    
    def RY2(self,theta=None):
        self.check_input_param(theta)
        return self.gates("Ry2")
    
    def RZ2(self,theta=None):
        self.check_input_param(theta) 
        return self.gates("Rz2")
    
    def R_plus(self,theta=None):
        self.check_input_param(theta)
        return self.gates("R+")
    
    def R_minus(self,theta=None):
        self.check_input_param(theta)
        return self.gates("R-")
    
    def check_input_param(self,theta):
        self.theta = theta
        if theta == None:
            raise ValueError("theta is None")

    def gates(self,type=""):
        
        state = self.state
        d_in = shapex(state)[0]
        N_in = self.N
        d_dicke = get_dim(N_in)

        if "Rx" in type:
            S = partial(Sx)
        
        elif "Ry" in type:
            S = partial(Sy)
        
        elif "Rz" in type:
            S = partial(Sz)
        
        elif "R+" in type:
            S = partial(S_plus)
        
        elif "R-" in type:
            S = partial(S_minus)

        if d_in != d_dicke:
            if "2" in type:
                J = S(N_in/2).dot(S(N_in/2))
            else:
                J = S(N_in/2)
        else:
            j_array = get_jarray(N_in)[::-1]
            blocks = []
            for j in j_array:
                if "2" in type:
                    blocks.append(S(j).dot(S(j)))
                else:
                    blocks.append(S(j))

            J = block_diag(blocks)
            
        J = csc_matrix(J)
        print("debug::",self.theta,J)
        expJ = expm(-1j*self.theta*J)
        expJ_conj = csc_matrix(daggx(expJ))
        new_state = expJ.dot(self.state).dot(expJ_conj)

        self.state = new_state

        return self



# def RZ(sobj,theta):
#     """ collective rotation gate around the z-axis
#     RZ = expm(-i*theta*Jz)|state>

#     Parameters
#     ----------
#     theta: rotation angle
#     state: quantum state

#     Return
#     ----------
#     new state
#     """

#     state = sobj.state.toarray()
#     d = shapex(state)[0]
#     Nds = get_Nds(d) #cannot use for pure
#     Nin = sobj.N #get from input

#     if not isoperx(state):
#         j = (d-1)/2
#         new_state = np.zeros((d,1),dtype = complex)
#         for idx in np.nonzero(state)[0]:
#             m = int(j - idx)
#             new_state[idx,0] = cm.exp(-1j*theta*m)*state[idx,0]
#     else:
#         if Nin != Nds: #not full blocks
#            j = Nin/2
#            new_state = np.zeros((d,d),dtype = complex)
#            iks = _get_non0_idx(state)
#            mm1 = get_mm1_idx_max(Nin)[0]
#            for ik in iks:
#               (m, m1) = mm1[ik]
#               new_state[ik] = state[ik]*cm.exp(-1j*theta*(m-m1)) 
#         else: #full blocks
#            new_state = np.zeros((d,d),dtype = complex)
#            iks = _get_non0_idx(state)
#            jmm1 = get_jmm1_idx(Nds)[0] 
#            for ik in iks:
#                (j,m,m1) = jmm1[ik]
#                new_state[ik] = state[ik]*cm.exp(-1j*theta*(m-m1))
#     sobj.state = bsr_matrix(new_state)

# def _get_non0_idx(matrix):
#     """get non zero indexs of a matrix"""
#     lidx = []
#     for i,j in enumerate(matrix):
#         for k,l in enumerate(j):
#           if l != 0.0:
#              lidx.append((i,k))
#     return lidx
