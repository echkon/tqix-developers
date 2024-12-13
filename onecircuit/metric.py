# -*- coding: utf-8 -*-
""" Toolbox for quantum metrology
    Calculate Holevo bound using semidefinite programing
    Calculate SLD and RLD bounds
    Author: Le Bin Ho @ 2022
"""

import qiskit.quantum_info as qi
import numpy as np
from numpy.linalg import inv, multi_dot, norm
from scipy.linalg import sqrtm, solve_sylvester
import copy
from tqix.qobj import eigenx,dotx,daggx

from onecircuit.algorithm import QuantumCircuit
from onecircuit.util import step_size
from onecircuit.measurement import QuantumMeasurement

def sld_qfim(circuit, method = 'eigens'):
    
    """calculate QFIM using SLD with different methods
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - sld_qfim
    """
    if method == 'eigens':
        return sld_qfim_eigens(circuit)
    elif method == 'inverse':
        return sld_qfim_inverse(circuit)
    elif method == 'sylvester':
        return sld_qfim_sylvester(circuit)
    else:
        raise ValueError(
            "Method is not avilable: please use 'eigens', 'inverse', or 'sylvester' ")
    
    
    
def sld_qfim_inverse(circuit):
    
    """ calculate the QFIM bases SLD 
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - qfim 
    """ 
    
    circuit_qc = QuantumCircuit(circuit)
    rho = qi.DensityMatrix(circuit_qc).data #rho after evolve
    grho = _grad_rho(circuit)
    
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


def sld_qfim_sylvester(circuit):
    
    """ calculate the QFIM bases SLD 
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - qfim 
    """ 
    
    circuit_qc = QuantumCircuit(circuit)
    rho = qi.DensityMatrix(circuit_qc).data #rho after evolve
    grho = _grad_rho(circuit)
    
    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)      
    L = sylvester(rho,rho,grho)
        
    for i in range(d):
        for j in range(d):
            H[i,j] = 0.5*np.trace(rho @ AntiCommutator(L[i],L[j]))

    return np.real(H)     


def sld_qfim_eigens(circuit):
    
    """ calculate the QFIM bases eigenvalues
    
    Args:
        - qc : initial circuit
        - qcirs: set of circuits (we always qc_add as model, no call here)

    Returns:
        - qfim 
    """ 
    
    circuit_qc = QuantumCircuit(circuit)
    rho = qi.DensityMatrix(circuit_qc).data #rho after evolve
    grho = _grad_rho(circuit)
    
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


def sld_bound(circuit):
    """ return the SLD bound 
    
    Args:
        - qc
        - qcirs

    Returns:
        - sld bound
    """
    
    d = len(circuit[1][3])
    W = np.identity(d)
    
    sld = sld_qfim(circuit)
    return np.real(np.trace(W @ inv(sld + np.eye(len(sld)) * 10e-10)))


def rld_qfim(circuit): 
    
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
    circuit_qc = QuantumCircuit(circuit)
    rho = qi.DensityMatrix(circuit_qc).data #rho after evolve
    grho = _grad_rho(circuit)
    
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


def cfim(circuit):
    """ return the classical fisher information matrix
    
    Args:
        - circuit
        
    Returns:
        - cfim
    """
    # measurements
    circuit_qc = QuantumCircuit(circuit)
    prob = QuantumMeasurement(circuit_qc)
    
    dpro = []
    d = len(circuit[1][3]) #[1]:sensing_qc,[1][3]:phases
    s = step_size
    
    #remove zeros indexs
    idxs = np.where(prob<=10e-18)
    prob = np.delete(prob, idxs)

    
    for i in range(0, d):
        # We use the parameter-shift rule explicitly
        # to compute the derivatives
        circuit1, circuit2 = copy.deepcopy(circuit), copy.deepcopy(circuit)    
        circuit1[1][3][i] += s #[1]:sensing_qc,[1][3]:phases
        circuit2[1][3][i] -= s
        
        plus = QuantumCircuit(circuit1) 
        minus = QuantumCircuit(circuit2)
        gr = (QuantumMeasurement(plus)-QuantumMeasurement(minus))/(2*s)
        gr = np.delete(gr, idxs)
        dpro.append(np.array(gr))
    
    matrix = np.zeros((d,d), dtype = float)
    for i in range(d):
        for j in range(d):      
            matrix[i,j] = np.sum(dpro[i] * dpro[j] / prob)

    return matrix
    

def cls_bound(circuit):
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
        
    clf = cfim(circuit)
    #print(f'CFIM = {clf}')
    W = np.identity(len(clf))
    
    return np.trace(W @ inv(clf + np.eye(len(clf)) * 10e-10))


def rd_bound(circuit):
    sld = sld_bound(circuit)
    cls = cls_bound(circuit)
    
    return 1-sld/cls


def _vectorize(rho):
    # return a vectorized of rho
    # rho: a matrices (data)
    vec_rho = np.reshape(rho, (len(rho)**2,1), order='F')
    return vec_rho


def _grad_rho(circuit):
    """ calculate drho by parameter-shift rule
    
    Args:
        - circuit
        
    Return:
        - gradient of state density w.r.t. all phases
    """
    
    dp = [] #array of matrices rho
    s = step_size
    
    for i in range(0, len(circuit[1][3])):
        circuit1, circuit2 = copy.deepcopy(circuit), copy.deepcopy(circuit)       
        circuit1[1][3][i] += s
        circuit2[1][3][i] -= s
        
        plus = QuantumCircuit(circuit1) 
        minus = QuantumCircuit(circuit2)      
        dp.append((qi.DensityMatrix(plus)-qi.DensityMatrix(minus))/(2*s))
        
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

def calculate_SLD(circuit):
    # to calculate SLD
        
    circuit_qc = QuantumCircuit(circuit)
    rho = qi.DensityMatrix(circuit_qc).data #rho after evolve
    grho = _grad_rho(circuit)
    
    d = len(grho) # number of paramaters     
    Ls = []
    eigens = eigenx(rho)
    
    for i in range(d):
        L = np.zeros((len(rho),len(rho)), dtype = complex) 
        for k in range(len(eigens[0])):
            for l in range(len(eigens[0])):
                de = eigens[0][k] + eigens[0][l]
                if de > 10e-15:
                    coe1 = dotx(daggx(eigens[1][k]),grho[i],eigens[1][l])/de
                    coe2 = eigens[1][k]@daggx(eigens[1][l])
                    L += 2*coe1*coe2
        Ls.append(L)                
    return Ls                  

def fidelity_state(uvd):
    
    """calculate fidelity
    
    Args:
        - qc : initial circuit

    Returns:
        - fidelity of two operators:
        - 1/4^N * Tr[UVdagg]
    """
    qc = uvdCircuit(uvd)
    rho = qi.DensityMatrix(qc).data #rho after evolve
    return np.abs(rho[0][0])**2

def mse(uvd,cost_func, step_size):
    """calculate Mean Square Error
    
    Args:
        - qc : initial circuit

    Returns:
        - mse
    """
    grad_theo = gradient_central_difference(uvd,cost_func,method='psr',step_size=step_size)
    grad_psr = gradient_central_difference(uvd,cost_func,method='psr',step_size=step_size)
    grad_2point = gradient_central_difference(uvd,cost_func,method='two_points',step_size=step_size)

    mse_psr = np.abs(np.array(grad_theo) - np.array(grad_psr))
    mse_2point = np.abs(np.array(grad_theo) - np.array(grad_2point))

    return mse_psr, mse_2point, grad_psr, grad_2point

def frobenius_norm(uvd):
    """calculate the frobenius norm 
    
    Args:
        - uvd
        
    Returns:
        ||u-vd_dagger||      
    """
    num_qubits = uvd[0][1] 
    u = QuantumCircuit(num_qubits)
    vd = QuantumCircuit(num_qubits)

    u &= uvd[0][0](uvd[0][1], uvd[0][2], uvd[0][3])
    vd &= uvd[1][0](uvd[1][1], uvd[1][2], uvd[1][3])

    u_data = qi.Operator(u).data
    vd_data = qi.Operator(vd).data

    # Compute the difference matrix U - V^\dagger
    diff = (u_data.real - vd_data.conj().T.real) + (u_data.imag - vd_data.conj().T.imag)

    # Compute and return the Frobenius norm of the difference
    return np.linalg.norm(diff, 'fro')

    
    return np.linalg.norm(diff, ord='fro') 
    

def infidelity_state(uvd):
    return 1 - fidelity_state(uvd)

def fidelity_operator(uvd):
    """calculate fidelity
    
    Args:
        - qc : initial circuit

    Returns:
        - fidelity of two operators:
        - 1/4^N * Tr[UVdagg]
    """
    qc = uvdCircuit(uvd)
    uvd_op = qi.Operator(qc).data
    fid = 1/4 ** qc.num_qubits * np.abs(np.trace(uvd_op))**2
    return fid

def infidelity_operator(uvd):
    return 1-fidelity_operator(uvd) 

def avg_fidelity_operator(uvd):
    
    sum_fid = 0
    for i in range(len(qci)):
        sv1 = Statevector(qci[i])
        sv2 = Statevector(qcs[i])

        sum_fid += np.abs(np.dot(np.conjugate(sv1).T,sv2))**2

    return sum_fid / len(qcs)
