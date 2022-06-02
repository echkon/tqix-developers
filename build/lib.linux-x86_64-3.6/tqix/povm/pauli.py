from tqix import *
from tqix.qtool import dotx,tensorx
import numpy as np
from itertools import product

__all__ = ['_pauli_']

def _pauli_(n):
    """
    Pauli POVM

    Parameters:
    ----------
    n : number of qubits
    
    Return:
    ------
    set of POVM accroding Pauli matrics
    """
    squbit = _pauli_eigens()
    nqubit = []
    T = []
    T = list(product((squbit),repeat = n))
    for i in range(len(T)):
        M = []
        for j in T[i]:
           M.append(j)
        nqubit.append(tensorx(M))
    return nqubit

def _pauli_eigens():
    """
    to get eigenstate of pauli matrices
    there are 3 bases: {0, 1}, {plus, minus}, {L, R}

    Return:
    -------
    Pauli eigenstate POVM set
    """
    u = obasis(2,0)
    d = obasis(2,1)
    h = dotx(u,daggx(u))
    v = dotx(d,daggx(d))
    p = dotx(normx(u+d),daggx(normx(u+d)))
    m = dotx(normx(u-d),daggx(normx(u-d)))
    l = dotx(normx(u+1j*d),daggx(normx(u+1j*d)))
    r = dotx(normx(u-1j*d),daggx(normx(u-1j*d)))
    m = [h,v,p,m,l,r]
    return m

