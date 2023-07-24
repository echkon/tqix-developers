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

__all__ = ['qfim']

import numpy as np
from numpy import arccos
from cmath import phase
from numpy.linalg import multi_dot
from scipy.linalg import expm,inv

def qfim():
    

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

