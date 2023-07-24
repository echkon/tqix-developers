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

__all__ = ['qfimx', 'Hx', 'Ux']

import numpy as np
from scipy.linalg import expm,inv

def qfimx(state,H):
    """calculate quantum fisher information

    Args:
        state (matrix): quantum state (density matrix)
        H (function): Hamiltonian
        
    Returns:
        qfim: quantum fisher information matrix    
    """
    return 1

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
    
def _integrate(A,H,t):
    """calculate integrate exp(itH)@A@exp(-itH)

    Args:
        A (matrix): Hermitian operator
        H (matrix): Hamiltonian
        t (float): time

    Returns:
        Ak: Hermition opetertor Ak = int_0^t exp(itH)@A@exp(-itH)
    """
    f = lambda u: dotx(expm(1j*u*H),A,expm(-1j*u*H))
    xv = np.linspace(0,t,1000)
    result = np.apply_along_axis(f,0,xv.reshape(1,-1))
    return np.trapz(result,xv)


