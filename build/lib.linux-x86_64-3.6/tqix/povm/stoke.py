from tqix import *
from tqix.qtool import dotx,tensorx
import numpy as np
from itertools import product

__all__ = ['_stoke_']

def _stoke_(n):
    """
    Stoke POVM

    Parameters:
    ----------
    n : number of qubits
    
    Return:
    ------
    set of POVM accroding Stoke parameters
    """
    squbit = _stoke_base()
    nqubit = []
    T = []
    T = list(product((squbit),repeat = n))
    for i in range(len(T)):
        M = []
        for j in T[i]:
           M.append(j)
        nqubit.append(tensorx(M))
    return nqubit

def _stoke_base():
    """
    to define four stock parameters
    there are defined to be H, V, D, R
    see: PRA.64.052312

    Return:
    -------
    stoke povm 
    """
    # default
    u = obasis(2,0)
    d = obasis(2,1)
    h = dotx(u,daggx(u))
    v = dotx(d,daggx(d))
    d = dotx(normx(u-d),daggx(normx(u-d)))
    r = dotx(normx(u-1j*d),daggx(normx(u-1j*d)))
    m = [h,v,d,r]
    return m

