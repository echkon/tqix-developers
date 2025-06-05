"""
>>> this file is a part of tqix: a Toolbox for Quantum in X
                                x: quantum measurement, quantum metrology, 
                                quantum tomography, and more.
________________________________
>>> copyright (c) 2019 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""
# quantum toolbox for operators

__all__ = ['dotx', 'tensorx', 'ptracex','deposex', 'itensorx','fidx']

import numpy as np
from numpy import arccos
from cmath import phase
from numpy.linalg import multi_dot
from scipy import sparse
from scipy.linalg import sqrtm

def dotx(*args):
    """
    calculate dot product of input operators
    Parametors
    ----------
    args : list or array

    Return
    ------
    dot product of input operators
    """
    if not args:
        raise TypeError("Please put at least one argument")

    return multi_dot(args)

def tensorx(*args):
    """
    calculate tensor product of input operators
    Parameters:
    ----------
    args: list or array
    
    Return:
    -------
    tensor prduct of input
    """
    if not args:
        raise TypeError("Please put at least one argument")

    if len(args) == 1 and \
        isinstance(args[0],(tuple,list,np.ndarray)):
        targs = args[0]
    else:
        targs = args
    
    t = targs[0]
    for i in range(1,len(targs)):
        t = np.kron(t,targs[i])
    return t

def ptracex(rho,sub_remove):
    """ Calculate the partial trace
    Parameters
    ----------
    rho: np.ndarray
        Density matrix
    *args: list
        Index of qubit to be kept after taking the trace
    Returns
    -------
    subrho: np.ndarray
        Density matrix after taking partial trace
    """
    n = int(np.log2(rho.shape[0]))
    a = 0
    for i in sub_remove:
        temp = _ptrace_(rho,n-a,i-a)
        #after each step, the number of qubits decreased by 1
        a += 1
    return temp
    
def _ptrace_(state,n,p):
    """
    ------------------------
    Input
    state: density matrix
    n: number of qubits
    p: traced qubit
    ------------------------
    Output: partially traced density matrix
    """
    d = 2**(n-1)
    temp = np.zeros((d,d),dtype=complex)
    for i in range(d):
        for j in range(d):
            temp[i,j] = state[_binary_(i,n,p,0),_binary_(j,n,p,0)]\
                    +state[_binary_(i,n,p,1),_binary_(j,n,p,1)]
    return temp

def _binary_(num,length,p,i):
    """
    First, represent number 'num' in binary string.
    Then, insert number 'i' into 'p' position of the string.
    ------------------------
    num: integer number
    length: number of qubits
    p: integer number
    i: either 0 or 1
    ------------------------
    Output: integer number
    """
    if (2**length < num):
        raise TypeError("Input can be truncated")
    if (length < p):
        print(length,p)
        raise TypeError("Out of range")
    if ((i != 0) and (i != 1)):
        raise TypeError("Inserted number must be 0 or 1")
    string = np.binary_repr(num,length)
    temp = string[:p]+str(i)+string[p:]
    result = int(temp,2)
    return result

def deposex(qubit):
    """
    depose a qubit into polar and azimuthal angles
    Parameters:
    ----------
    qubit
    
    Return:
    -------
    polar and azimuthal angles
    """
    polar = 2 * arccos(abs(qubit[0]))
    azimuthal = phase(qubit[1]) - phase(qubit[0])

    return float(polar), float(azimuthal)

def itensorx(oper,N):
    """calculate a list of tensors between oper and eye

    Args:
        oper (matrix): operator
        N (int): number of qubits
        
    Return
        list of tensors    
    """
    # convert oper to sparse matrix
    oper_csr = sparse.csr_matrix(oper)
    idn_csr = sparse.identity(2)
        
    ltensors = []
    for i in range(N):
        # set list of N opers with l[i] = oper
        itensors = [idn_csr] * N
        itensors[i] = oper_csr
        
        # compute tensor
        result = itensors[0]
        for j in range(1, N):
            result = sparse.kron(result, itensors[j])

        result_csr = sparse.csr_matrix(result.todense())  
        ltensors.append(result_csr)    
    return ltensors

def ftensorx(oper,N):
    """calculate full tensors between opers,
    i.e., oper \otimes oper \otimes...

    Args:
        oper (matrix): operator
        N (int): number of qubits
        
    Return
        list of tensors    
    """
    # convert oper to sparse matrix
    oper_csr = sparse.csr_matrix(oper)
    idn_csr = sparse.identity(2)
        
    ltensors = []
    for i in range(N):
        # set list of N opers with l[i] = oper
        itensors = [idn_csr] * N
        itensors[i] = oper_csr
        
        # compute tensor
        result = itensors[0]
        for j in range(1, N):
            result = sparse.kron(result, itensors[j])

        result_csr = sparse.csr_matrix(result.todense())  
        ltensors.append(result_csr)    
    return ltensors

def fidx(rho,sigma):
    #calcualte fidelity between rho and sigma
    sqrt_rho = sqrtm(rho)
    fid = np.trace(sqrtm(sqrt_rho @ sigma @ sqrt_rho)).real**2
    return fid