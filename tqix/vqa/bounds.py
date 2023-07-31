# -*- coding: utf-8 -*-
""" Toolbox for quantum metrology
    Calculate Holevo bound using semidefinite programing
    Calculate SLD and RLD bounds
    Author: Le Bin Ho @ 2022
"""

import qiskit
import tqix.vqa.circuits
import tqix.vqa.constants
import tqix.vqa.vqm

import numpy as np
from numpy.linalg import inv, multi_dot, norm
from scipy.linalg import sqrtm, solve_sylvester
import copy

from tqix.qobj import eigenx,dotx,daggx

def sld_qfim(qc, qcirs, method = 'eigens'):
    
    """calculate QFIM using SLD with different methods
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - sld_qfim
    """
    if method == 'eigens':
        return sld_qfim_eigens(qc, qcirs)
    elif method == 'inverse':
        return sld_qfim_inverse(qc, qcirs)
    elif method == 'sylvester':
        return sld_qfim_sylvester(qc, qcirs)
    else:
        raise ValueError(
            "Method is not avilable: please use 'eigens', 'inverse', or 'sylvester' ")
    
    
    
def sld_qfim_inverse(qc, qcirs):
    
    """ calculate the QFIM bases SLD 
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - qfim 
    """ 
    
    cir = tqix.vqa.vqm.qc_add(qc.copy(), qcirs)
    rho = tqix.vqa.circuits.state_density(cir.copy()) #rho after evolve
    grho = _grad_rho(qc.copy(), qcirs)
    
    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)
    invM = _inv_M(rho)
    
    vec_grho = []  
    for i in range(d):
        vec_grho.append(_vectorize(grho[i]))

    for i in range(d):
        for j in range(d): 
            H[i,j] = 2*multi_dot([np.conjugate(vec_grho[i]).T, invM, vec_grho[j]]) 
    return np.real(H)  


def sld_qfim_sylvester(qc, qcirs):
    
    """ calculate the QFIM bases SLD 
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - qfim 
    """ 
    
    cir = tqix.vqa.vqm.qc_add(qc.copy(), qcirs)
    rho = tqix.vqa.circuits.state_density(cir.copy()) #rho after evolve
    grho = _grad_rho(qc.copy(), qcirs)
    
    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)      
    L = sylvester(rho,rho,grho)
        
    for i in range(d):
        for j in range(d):
            H[i,j] = 0.5*np.trace(rho @ AntiCommutator(L[i],L[j]))

    return np.real(H)     


def sld_qfim_eigens(qc, qcirs):
    
    """ calculate the QFIM bases eigenvalues
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - qfim 
    """ 
    
    cir = tqix.vqa.vqm.qc_add(qc.copy(), qcirs)
    rho = tqix.vqa.circuits.state_density(cir.copy()) #rho after evolve
    grho = _grad_rho(qc.copy(), qcirs)
    
    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)      
    
    eigens = eigenx(rho)
        
    for k in range(d):
        for l in range(d):
            H[k,l] = 0.0
            for i in range(len(eigens[0])):
                for j in range(len(eigens[0])):
                    de = eigens[0][i] + eigens[0][j]
                    if de > 10e-15:
                        num1 = dotx(daggx(eigens[1][i]),grho[k],eigens[1][j])
                        num2 = dotx(daggx(eigens[1][j]),grho[l],eigens[1][i])
                        H[k,l] += 2*np.real(num1*num2)/de
                        
    return np.real(H)  


def sld_bound(qc, qcirs):
    """ return the SLD bound 
    
    Args:
        - qc
        - qcirs

    Returns:
        - sld bound
    """
    
    d = len(qcirs[1][2]) #[1]:u_phase,[1][2]:phases
    W = np.identity(d)
    
    sld = sld_qfim(qc.copy(), qcirs)
    return np.real(np.trace(W @ inv(sld + np.eye(len(sld)) * 10e-10)))


def rld_qfim(qc, qcirs): 
    
    """ calculate the QFIM bases RLD 
    
    Analytical:
        H_ij = tr(rho*L_i*L_j^\dag)
            = tr(L_j^\dag*rho*L_i)

        submit rho*L_i = der_i(rho) : this is rld
        we get: H_ij = tr(L_j^\dag*dev_i(rho))
        
        apply tr(A^dag*B) = vec(A)^dag*vec(B), 
        we have: H_ij = vec(L_j)^dag*vec[dev_i(rho)]
        here: vec[L_j)] = (I x rho)^{-1)*vec(L_j)
        
    Args:
        - qc
        - qcirs

    Returns:
        - qfim via rld   
    """ 
    qc_func = tqix.vqa.vqm.qc_add(qc.copy(), qcirs)
    rho = tqix.vqa.circuits.state_density(qc_func.copy())
    grho = _grad_rho(qc.copy(), qcirs)
    
    d = len(grho) # number of estimated parameters  
    R = np.zeros((d,d), dtype = complex)
    IR = _i_R(rho)
    
    vec_grho = []
    for i in range(d):
        vec_grho.append(_vectorize(grho[i]))
    
    for i in range(d):
        for j in range(d): 
            vecLj = np.conjugate(multi_dot([IR, vec_grho[j]])).T
            R[i,j] = multi_dot([vecLj, vec_grho[i]])  
            
    return R


def rld_bound(qc, qcirs):
    """ return the SLD bound 
    
    Args:
        - qc
        - cirs
        - params
        - coefs
        - W: weight matrix

    Returns:
        - rld bound
    """
    
    d = len(qcirs[1][2]) #[1]:u_phase,[1][2]:phases
    W = np.identity(d)
        
    rld = rld_qfim(qc, qcirs)
    invrld = inv(rld + np.eye(len(rld)) * 10e-10)
    R1 = np.trace(W @ np.real(invrld))
    R2 = norm(multi_dot([sqrtm(W), np.imag(invrld),sqrtm(W)])) 
    
    return R1 + R2


def cfim(qc, qcirs):
    """ return the classical fisher information matrix
    
    Args:
        - qc
        - qcirs
        
    Returns:
        - cfim
    """
    # measurements
    qc_copy = tqix.vqa.vqm.qc_add(qc.copy(), qcirs)
    pro = tqix.vqa.circuits.measure_born(qc_copy)
    
    dpro = []
    d = len(qcirs[1][2]) #[1]:u_phase,[1][2]:phases
    s = tqix.vqa.constants.step_size
    
    #remove zeros indexs
    idxs = np.where(pro<=10e-18)
    pro = np.delete(pro, idxs)

    
    for i in range(0, d):
        # We use the parameter-shift rule explicitly
        # to compute the derivatives
        qcirs1, qcirs2 = copy.deepcopy(qcirs), copy.deepcopy(qcirs)    
        qcirs1[1][2][i] += s #[1]:u_phase,[1][2]:phases
        qcirs2[1][2][i] -= s
        
        plus = tqix.vqa.vqm.qc_add(qc.copy(), qcirs1) 
        minus = tqix.vqa.vqm.qc_add(qc.copy(), qcirs2)
        gr = (tqix.vqa.circuits.measure_born(plus)-tqix.vqa.circuits.measure_born(minus))/(2*s)
        gr = np.delete(gr, idxs)
        dpro.append(np.array(gr))
    
    matrix = np.zeros((d,d), dtype = float)
    for i in range(d):
        for j in range(d):      
            matrix[i,j] = np.sum(dpro[i] * dpro[j] / pro)

    return matrix


def cls_bound(qc, qcirs):
    """ return the classical bound 
    
    Args:
        - qc
        - qcirs
        - W: weight matrix

    Returns:
        - rld bound
    """
    
    #list2str = list(map(lambda f: f.__name__, cirs)) 
    #idx = list2str.index('u_phase') 
    #d = len(params[idx])
        
    clf = cfim(qc.copy(), qcirs)
    W = np.identity(len(clf))
    
    return np.trace(W @ inv(clf + np.eye(len(clf)) * 10e-10))


def _vectorize(rho):
    # return a vectorized of rho
    # rho: a matrices (data)
    vec_rho = np.reshape(rho, (len(rho)**2,1), order='F')
    return vec_rho


def _grad_rho(qc, qcirs):
    """ calculate drho by parameter-shift rule
    
    Args:
        - qc: initial cuircuit (we need to reset for rho(+) and rho(-)
        - qcirs: circuits
        
    Return:
        - gradient of state density w.r.t. all phases
    """
    
    dp = [] #array of matrices rho
    s = tqix.vqa.constants.step_size 
    
    for i in range(0, len(qcirs[1][2])):
        qcirs1, qcirs2 = copy.deepcopy(qcirs), copy.deepcopy(qcirs)       
        qcirs1[1][2][i] += s
        qcirs2[1][2][i] -= s
        
        plus = tqix.vqa.vqm.qc_add(qc.copy(), qcirs1) 
        minus = tqix.vqa.vqm.qc_add(qc.copy(), qcirs2)       
        dp.append((tqix.vqa.circuits.state_density(plus)-tqix.vqa.circuits.state_density(minus))/(2*s))
        
    return dp


def _inv_M(rho, epsilon = 10e-10): 
    """ return inverse matrix M 
        M = rho.conj()*I + I*rho.conj()
    
    Args:
        - quantum state rho (data)

    Returns:
        - inverse matrix M 
    """
    
    d = len(rho)
    M = np.kron(np.conj(rho), np.identity(d)) + np.kron(np.identity(d), rho)
    return inv(M + np.eye(len(M)) * epsilon)


def _i_R(rho, epsilon = 10e-10):
    """ return inverse of matrix R 
        R = I*rho.conj()
    
    Args:
        - quantum state rho (data)

    Returns:
        - inverse matrix R 
    """    
    d = len(rho)
    R = np.kron(np.identity(d), rho)
    return inv(R) #inv(R + np.eye(len(R)) * epsilon)


def sylvester(A,B,C):
    """ solve the sylvester function:
        AX + XB = 2*C # the symmetric logarithmic derivative
        here A,B = rho for SLD, i.e., rho @ L + L @ rho = 2*drho
            A = rho, B = 0 for RLD, i.e., rho @ L = 2*drho
            C = drho
    Args:
        - rho: input quantum state
        - drho: input derivative of quantum state
    Returns:
        - X operator
    """
    
    lenC = len(C) #we have several drho (gradient of rho)
    X = []
    
    for i in range(lenC):
        L = solve_sylvester(A, B, 2*C[i])
        X.append(L)
        
    #print(X)  
    return X


def AntiCommutator(A,B):
    # AB + BA
    return A @ B + B @ A                     
