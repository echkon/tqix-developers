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

from numpy import sqrt,real,sum,abs
import numpy as np
from numpy.linalg import eigh
#from scipy.linalg import sqrtm
from random import random
from tqix.qtool import dotx
from tqix.qx import *

__all__ = ['gtrace','gfide','ginfide']

def gtrace(a,b):
    """
    To get trace distance between a and b
    See:  Nielsen & Chuang, "Quantum Computation and Quantum Information"
    
    Input
    ---------
    a : density matrix or state vector
    b : the same as a

    Result
    ---------
    fidelity : float
    """
    # check if they are quantum object
    if isqx(a) and isqx(b):
       a = operx(a)
       b = operx(b)
       if a.shape != b.shape:
          raise TypeError('a and b do not have the same dimensions.')

       diff = a - b
       diff = dotx(daggx(diff),diff)
       # get eigenvalues of sparse matrix
       vals, vecs= eigh(diff)
       return float(real(0.5 * np.sum(sqrt(np.abs(vals)))))
    else:
       msg = 'a or b is not a quantum object'
       raise TypeError(msg)

def gfide(a,b):
    """
    To get fidelity of a and b
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"
    
    Input
    ---------
    a : density matrix or state vector
    b : the same as a

    Result
    ---------
    fidelity : float
    """

    # check if they are quantum onject
    if isqx(a) and isqx(b):
       if typex(a) != 'oper':
          sqrtma = operx(a)
          if typex(b) != 'oper':
             b = operx(b)
       
       else:
          if typex(b) != 'oper':
             #swap a and b
             return gfide(b,a)
          sqrtma = sqrtx(a)

       if sqrtma.shape != b.shape:
          raise TypeError('Density matrices do not have same dimensions.')
    
       eig_vals, eig_vecs = eigh(dotx(sqrtma,b,sqrtma)) 
       return float(real(sum(sqrt(eig_vals[eig_vals > 0]))))
    else:
        msg = ' a or b is not a quantum object'
        raise TypeError(msg)

def ginfide(a,b):
    #to get infidelity
    return 1-gfide(a,b)
