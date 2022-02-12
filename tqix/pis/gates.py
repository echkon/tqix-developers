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

class Gates:
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
        self.theta = None
        self.sobj = None
    
    def RX(self,sobj=None,theta=None):
        self.check_input_param(sobj,theta)
        return self.gates("Rx")
    
    def RY(self,sobj=None,theta=None):
        self.check_input_param(sobj,theta) 
        return self.gates("Ry")
    
    def RZ(self,sobj=None,theta=None):
        self.check_input_param(sobj,theta)
        return self.gates("Rz")
    
    def RX2(self,sobj=None,theta=None):
        self.check_input_param(sobj,theta)
        return self.gates("Rx2")
    
    def RY2(self,sobj=None,theta=None):
        self.check_input_param(sobj,theta)
        return self.gates("Ry2")
    
    def RZ2(self,sobj=None,theta=None):
        self.check_input_param(sobj,theta) 
        return self.gates("Rz2")
    
    def R_plus(self,sobj=None,theta=None):
        self.check_input_param(sobj,theta)
        return self.gates("R+")
    
    def R_minus(self,sobj=None,theta=None):
        self.check_input_param(sobj,theta)
        return self.gates("R-")
    
    def check_input_param(self,sobj,theta):
        old_sobj = self.sobj
        old_theta = self.theta
        new_sobj = sobj
        new_theta = theta 

        if old_sobj is not None and old_theta is not None:
            if new_sobj == None and new_theta == None:
                self.sobj = old_sobj
                self.theta = old_theta
            elif new_sobj == None and new_theta != None:
                self.theta = new_theta
            else:
                raise ValueError("new inputs override old state")
        
        elif old_sobj is None and old_theta is  None:
            if new_theta ==None or new_sobj == None:
                raise ValueError("theta or quantum object is None")
            elif new_theta !=None and new_sobj != None:
                self.sobj = new_sobj
                self.theta = new_theta
            

    def gates(self,type=""):
        
        state = self.sobj.state
        d_in = shapex(state)[0]
        N_in = self.sobj.N
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
        expJ = expm(-1j*self.theta*J)
        expJ_conj = csc_matrix(daggx(expJ))
        new_state = expJ.dot(self.sobj.state).dot(expJ_conj)

        self.sobj.state = new_state

        return self
if __name__ == "__main__":
    N=3
    print('test with dicke_ghz state')
    init_state = dicke_ghz(N)
    qc = circuit(N,init_state)
    state = qc.state
    print(state)
    print(typex(state))

    print('test gate with dicke_ghz state')
    Gates().RZ(qc,np.pi/3).RY().RZ(theta=np.pi/4)
    # Gates().RZ()
    print(qc.state)



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
