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
# for quantum metrology

__all__ = ['qfimx', 'qboundx', 'Hx', 'Ux', 'Ax']

import numpy as np
from tqix.qobj import daggx, dotx, eigenx, operx, tracex
from tqix.qoper import eyex
from scipy.linalg import expm,inv

def qfimx(inp_state,h_opt,c_opt,t):
    """calculate quantum fisher information

    Args:
        inp_state (matrix): input quantum state 
        h_opt(list): list of Hamiltonian
        c_opt(list): list of coefficients
        t(float): time
        
    Returns:
        qfim: quantum fisher information matrix    
    """
    
    # length of parameters
    d = len(h_opt)
    
    # get rho
    if isinstance(inp_state, sparse.spmatrix):
        inp_state = inp_state.toarray()
    rho = operx(inp_state)
    
    # unitary, #Ax #final state
    uni = Ux(h_opt,c_opt,t)
    ax = Ax(h_opt,c_opt,t)
    fin_rho = dotx(uni,rho,daggx(uni))
    
    # eigen_rho,  # derivative of fin_state
    eigens = eigenx(fin_rho)
    drhos = _droh(rho,uni,ax) #use inp_rho to calculate drho,
    
    # qfim
    Q = np.zeros((d,d), dtype = complex)  
    for k in range(d):
        for l in range(d):
            Q[k,l] = 0.0
            for i in range(len(eigens[0])):
                for j in range(len(eigens[0])):
                    de = eigens[0][i] + eigens[0][j]
                    if de > 10e-15:
                        num1 = dotx(daggx(eigens[1][i]),drhos[k],eigens[1][j])
                        num2 = dotx(daggx(eigens[1][j]),drhos[l],eigens[1][i])
                        Q[k,l] += 2*np.real(num1*num2)/de

    return np.real(Q)     

def qboundx(inp_state,h_opt,c_opt,t):
    """calculate quantum bound

    Args:
        inp_state (matrix): input quantum state 
        h_opt(list): list of Hamiltonian
        c_opt(list): list of coefficients
        t(float): time
        
    Returns:
        qbound: quantum bound   
    """
    d = len(h_opt)
    W = eyex(d)
    qfim = qfimx(inp_state,h_opt,c_opt,t)
    result = np.real(tracex(W @ inv(qfim + W * 10e-10)))
    
    return result


def Hx(h_opt,c_opt):
    """to create Hamiltonian from h_opt and c_opt

    Args:
        h_opt (list): list of hamiltonian 
        c_opt (list): list of coefficients

    Returns:
        Hx: Hamiltonian
    """
    if len(h_opt) != len(c_opt):
        raise TypeError("list length does not match")
    else:
        H = 0.0
        for i in range(len(h_opt)):
            H += h_opt[i]*c_opt[i]
        return H    

def Ux(h_opt,c_opt,t):
    """to create Unitary operator from h_opt and c_opt

    Args:
        h_opt (list): list of hamiltonian 
        c_opt (list): list of coefficients
        t (float): time

    Returns:
        Ux: Unitary operator
    """
    return expm(-1j*t*Hx(h_opt,c_opt))
            
###
#def _drho():
    
def Ax(h_opt,c_opt,t):
    """calculate Ax = exp(itH)@A@exp(-itH)

    Args:
        h_opt (list): list of Hamiltonian
        c_opt (list): list of coefficients
        t (float): time

    Returns:
        Ax: list of Ax
    """
    if len(h_opt) != len(c_opt):
        raise TypeError("list length does not match")
    
    axl = []
    hamil = Hx(h_opt,c_opt)
    xv = np.linspace(0,t,1000)
    for i in range(len(h_opt)):
        f = lambda u: dotx(expm(1j*u*hamil),h_opt[i],expm(-1j*u*hamil))
        result = np.apply_along_axis(f,0,xv.reshape(1,-1))
        axl.append(np.trapz(result,xv))
    return axl #list of Ax    

def _droh(rho,U,A):
    """calculate d_rho following Eq.(C1) PRA 102, 022602 (2020)

    Args:
        rho (matrix): quantum state need to calculate deri
        U (matrix): unitary
        A (list): list of Ak
    """
    drhos = []
    for i in range(len(A)):
        dU = -1j*dotx(U,A[i])
        dUU = dotx(dU,rho,daggx(U))
        UdU = dotx(U,rho,daggx(dU))
        drhos.append(dUU + UdU)
    return drhos    
