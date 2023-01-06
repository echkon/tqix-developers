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
import math 
from scipy.sparse import csc_matrix

__all__ = ['get_Nds','get_dim','get_num_block','get_array_block',
           'get_jmin','get_jarray','get_marray',
           'get_vidx','get_midx','get_jmm1_idx','get_mm1_idx_max','get_A',
           'get_B','get_D','get_Lambda','get_alpha','dicke_bx','isclose','dicke_bx1','fit'
            ]
        
def get_Nds(d):
    """_summary_
    to get numper of particle from dimension d

    Args:
        d (int): dimension

    Returns:
        N: number of particle
    """    
    d4 = d*4
    root = np.sqrt(d4)
    if int(root + 0.5)**2 == d4: #perfect quare number
        return int(root - 2)
    else:
        return int(np.sqrt(1+d4)-2)
        
def get_dim(N):
    """_summary_
    to get dimension from number of particle N

    Args:
        N (int): number of particle

    Returns:
        d: dimension 
    """    
    
    d = (N/2 + 1)**2 - (N % 2)/4
    return int(d)


def get_num_block(N):
    """_summary_
    to get number of block

    Args:
        N (int): number of particle

    Returns:
        Nbj: Number of block j
    """    
    return int(N/2 + 1 -1/2*(N % 2))
    
def get_array_block(N):
    # an array of block
    num_block = get_num_block(N)
    array_block = [i * (N+2-i) for i in range(1, num_block+1)]
    return array_block

def get_jmin(N):
    """ to get j min

    Paramater:
    -------------
    N: number of particle

    Return:
    -------------
    jmin: min of j """

    if N % 2 == 0:
        return 0
    else:
        return 0.5
        
def get_jarray(N):
    """ to get array of j from N

    Paramater:
    -------------
    N: number of particle

    Return:
    -------------
    jarray: a array of j"""

    jarray = np.arange(get_jmin(N),N/2 + 1,1)
    return jarray
       
def get_marray(j):
    """ to get array of m from j

    Paramater:
    -------------
    j: number of j

    Return:
    -------------
    marray: a array of j"""

    marray = np.arange(-j,j+1,1)
    return marray
          
def get_vidx(j,m):
    """_summary_

    Args:
        j (int): j index
        m (int): m index

    Returns:
        vidx
    """    
    # to get index in vector state from j,m
    return int(j-m)

def get_mm1_idx_max(N):
    """_summary_

    Args:
        N (int): number of qubits

    Returns:
        [mm1,ik]
    """    
    # to get mm1 or ik

    ik = {}
    mm1 = {}
    j = N/2

    m_vary = get_marray(j)
    for m in m_vary:
        for m1 in m_vary:
            i, k = j - m, j - m1
            mm1[(i, k)] = (m, m1)
            ik[(m, m1)] = (i, k)
    return [mm1,ik]


def get_midx(N,j,m,m1,block):
    """_summary_

    Args:
        N (int): number of qubits
        j (int): j index
        m (int): m index
        m1 (int): m1 index
        block (List): list of blocks

    Returns:
        (i,k)
    """    
    # to get index in density matrix
    # ref. qutip
    k = int(j-m1)
    kp = int(j-m)
    block_num = int(N/2 - j) #0,1,2,...
    offset = 0
    
    if block_num > 0:
        offset = block[block_num - 1]
    i = kp + offset
    k = k + offset
    return (i,k)
    
def get_jmm1_idx(N):
    """_summary_

    Args:
        N (int): number of qubits

    Returns:
        [jmm1,ik]
    """    
    # get index i,k of density matrix from j,m,m1
    # and revert j,m,m1 from i,k
    # ref. qutip
    
    ik = {}
    jmm1 = {}
    
    block = get_array_block(N)
    j_vary = get_jarray(N)
    for j in j_vary:
        m_vary = get_marray(j)
        for m in m_vary:
            for m1 in m_vary:
                i, k = get_midx(N,j,m,m1,block)
                jmm1[(i, k)] = (j, m, m1)
                ik[(j, m, m1)] = (i, k)
    return [jmm1,ik]

def get_A(j,m,type=""):
    """_summary_

    Args:
        j (int): j index
        m (int): m index
        type (str, optional): type of A. Defaults to "".

    Returns:
        A_{type}
    """    
    if type == "+":
        return np.sqrt((j-m)*(j+m+1))
    elif type == "-":
        return np.sqrt((j+m)*(j-m+1))
    else:
        return m 

def get_B(j,m,type=""):
    """_summary_

    Args:
        j (int): j index
        m (int): m index
        type (str, optional): type of A. Defaults to "".

    Returns:
        B_{type}
    """   
    if type == "+":
        return np.sqrt((j-m)*(j-m-1))
    elif type == "-":
        return -np.sqrt((j+m)*(j+m-1))
    else:
        return np.sqrt((j+m)*(j-m))

def get_D(j,m,type=""):
    """_summary_

    Args:
        j (int): j index
        m (int): m index
        type (str, optional): type of A. Defaults to "".

    Returns:
        D_{type}
    """   
    if type == "+":
        return -np.sqrt((j+m+1)*(j+m+2))
    if type == "-":
        return np.sqrt((j-m+1)*(j-m+2))
    else:
        return np.sqrt((j+m+1)*(j-m+1))

def get_Lambda(N,j,type=""):
    """_summary_

    Args:
        j (int): j index
        m (int): m index
        type (str, optional): type of A. Defaults to "".

    Returns:
        Lambda_{type}
    """   
    if type == "a":
        return (N/2+1)/(2*j*(j+1))
    elif type == "b":
        return (N/2+j+1)/(2*j*(2*j+1))
    else:
        return (N/2-j)/(2*(j+1)*(2*j+1))

def get_alpha(N,j):
    """_summary_

    Args:
        N (int): number of qubits
        j (int): j indexes 

    Returns:
        alpha
    """    
    return math.factorial(N)/(math.factorial(N/2-j)*math.factorial(N/2+j))

def dicke_bx(N, jmm1):
    """_summary_

    Args:
        N (int): _description_
        jmm1 (dict): store p value at j,m,m1 indexes

    Returns:
        new state
    """    
    # create a dicke basis follow jmm1
    # jmm1 as {(j,m,m1):p}
    
    dim = get_dim(N)
    rho = np.zeros((dim,dim),dtype = complex)
    ik = get_jmm1_idx(N)[1] # return i,k from jmm1
    for key in jmm1:
        i,k = ik[key]
        rho[i,k] = jmm1[key]
    return csc_matrix(rho)

def dicke_bx1(N,jmm1,ik,dim):
    """_summary_

    Args:
        N (int): number of qubits
        jmm1 (dict): stores p value at j,m,m1 indexes 
        ik (dict): stores i,k indexes wrt j,m,m1
        dim (int): dimension of rho 

    Returns:
        new state
    """    
    # create a dicke basis follow jmm1
    # jmm1 as {(j,m,m1):p}

    rho = np.zeros((dim,dim),dtype = complex)
    for key in jmm1:
        i,k = ik[key]
        rho[i,k] = jmm1[key]
    return csc_matrix(rho)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """_summary_
    Check if 2 numbers is closed
    Args:
        a (float)
        b (float)
        rel_tol (float, optional): Defaults to 1e-09.
        abs_tol (float, optional): Defaults to 0.0.

    Returns:
        bool 
    """    
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def fit(objective_function,optimizer,init_params,return_loss_hist=None,loss_break=None,return_time_iters=None):
    """_summary_

    Args:
        objective_function (func): objetive function to minimize
        optimizer (func): optimizer function 
        init_params (List): list of initial parameters
        return_loss_hist (bool, optional): return history of objective values. Defaults to None.
        loss_break (bool, optional): early stopping. Defaults to None.
        return_time_iters (bool, optional): return time each iterations. Defaults to None.

    Returns:
        output
    """    
    output = optimizer.optimize(len(init_params), objective_function, initial_point=init_params,
                return_loss_hist=return_loss_hist,loss_break=loss_break,return_time_iters=return_time_iters)
    return output