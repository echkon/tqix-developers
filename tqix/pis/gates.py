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
import random
from sympy import csc
from tqix.qx import *
from tqix.qtool import dotx
from tqix.pis.util import *
from tqix.pis import *
from scipy.sparse import bsr_matrix,block_diag,csc_matrix,csr_matrix,lil_matrix
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

    def RX(self,theta=None,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Rx",noise=noise)
    
    def RY(self,theta=None,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta}
        self.check_input_param(params) 
        return self.gates("Ry",noise=noise)
    
    def RZ(self,theta=None,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Rz",noise=noise)
    
    def OAT(self,theta,gate_type,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta,"gate_type":gate_type}
        self.check_input_param(params)
        return eval(f"self.R{gate_type.upper()}2({theta},noise={noise})")
    
    def TAT(self,theta,gate_type,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta,"gate_type":gate_type}
        self.check_input_param(params)
        return self.gates(type=gate_type+"TAT",noise=noise)
    
    def TNT(self,theta,gate_type,*args, **kwargs):
        omega = kwargs.pop('omega', None)
        noise = kwargs.pop('noise', None)
        params = {"theta":theta,"gate_type":gate_type,"omega":omega}
        self.check_input_param(params)
        return self.gates(type=gate_type+"TNT",omega=omega,noise=noise)

    def RX2(self,theta=None,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Rx2",noise=noise)
    
    def RY2(self,theta=None,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Ry2",noise=noise)
    
    def RZ2(self,theta=None,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Rz2",noise=noise)
    
    def R_plus(self,theta=None,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("R+",noise=noise)
    
    def R_minus(self,theta=None,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("R-",noise=noise)
    
    def GMS(self,theta,phi,gate_type,*args, **kwargs):
        noise = kwargs.pop('noise', None)
        params = {"theta":theta,"phi":phi,"gate_type":gate_type}
        self.check_input_param(params)
        return self.gates(type=gate_type+"GMS",phi=phi,noise=noise)

    def check_input_param(self,params):
        self.theta = params["theta"]
        for param,value in params.items():
            if value == None:
                raise ValueError(f"{param} is None")

    def get_J(self,N_in,d_in,d_dicke,type):
        if "x" in type:
            S = partial(Sx)
        
        elif "y" in type:
            S = partial(Sy)
        
        elif "z" in type:
            S = partial(Sz)
        
        elif "+" in type:
            S = partial(S_plus)
        
        elif "-" in type:
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

            J = block_diag(blocks,format="csc")
            
        return J

    def gates(self,type="",*args, **kwargs):
        state = self.state
        d_in = shapex(state)[0]
        N_in = self.N
        d_dicke = get_dim(N_in)
        
        get_J = partial(self.get_J,N_in,d_in,d_dicke)
        
        type = type.lower()
        count_ops = 0
        for ops in ["x","y","z","+","-"]:
            if ops in type:
                count_ops += 1
        
        if count_ops == 1:
            J = get_J(type)
            expJ = expm(-1j*self.theta*J)
        elif count_ops == 2:
            if "tat" in type:
                J_2 = get_J(type[0]+"2")
                J_2_prime = get_J(type[1]+"2")
                expJ = expm(-1j*self.theta*(J_2-J_2_prime))
            elif "tnt" in type:
                omega = kwargs.pop('omega', None)
                J_2 = get_J(type[0]+"2")
                J_prime = get_J(type[1])
                expJ = expm(-1j*(self.theta*J_2+omega*J_prime))
            elif "gms" in type:
                phi = kwargs.pop('phi', None)
                J = get_J(type[0])
                J_prime = get_J(type[1])
                S_phi = 2*(J*np.cos(phi)+J_prime*np.sin(phi))
                expJ = expm(-1j*self.theta*S_phi.dot(S_phi)/4)

        expJ_conj = daggx(expJ)
        new_state = expJ.dot(self.state).dot(expJ_conj)

        noise = kwargs.pop('noise', None)
        
        if noise is not None:
            new_state = add_noise(self,noise)
            self.state = new_state
        else:
            self.state = new_state
        
        return self
    
    def measure(self,num_shots = None):
        t_prob = np.real(csr_matrix(self.state).diagonal())
        result = np.zeros_like(t_prob)
        mask_zeros = t_prob != 0
        for _ in range(num_shots):    
            rand_prob = np.array([random.uniform(0,1)]*t_prob.shape[0])
            mask_prob_ge = t_prob > rand_prob
            result[mask_zeros & mask_prob_ge] += 1
        result /= num_shots
        result[-1] = 1 - np.sum(result[:-1])

        return result   


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
