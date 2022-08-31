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

# description: math tools for a quantum object x

__all__ = ['qx','typex','shapex',
           'isqx','isbrax','isketx','isoperx','ishermx','isnormx',
           'operx','diagx','conjx','transx','daggx','tracex','eigenx',
           'groundx','expx','sqrtx','l2normx','normx']

from numpy import (sqrt,allclose,array_equal)
from scipy import (randn,diagonal,absolute,multiply)
from scipy.linalg import qr
import warnings
import numpy as np
from random import random
from tqix.qtool import dotx
import scipy
import torch 

def qx(x):
    """ 
    to turn x into a quantum object.
    we define a quantum object is an array dims [[n],[m]]
    or shape (n, m)
    >>> for example: 
        qx([[1],[2],[3]]) will return a ket state:
        => [[1]
            [2]
            [3]]
    """
    if isinstance(x,(int,float,complex)):
       return np.array([[x]])
    elif isinstance(x,(tuple,list)):
       if len(get_shape(x)) == 1: #1d list
          qox = np.array([x])
       else:
          qox = np.array(x)
       return qox 
    else: # ndarray already 
       qox = np.array([x]) if x.ndim == 1 else np.array(x)
       return qox

def typex(x):
    """
    return a quantum type of x:
    bar, ket, oper
    """
    if scipy.sparse.issparse(x):
      x = x.toarray()
    if isqx(x):
       if x.shape[0]!=1:
          return 'oper' if x.shape[1] != 1 else 'ket'
       elif x.shape == (1,1):
          return 'oper'
       else:
          return 'bra'
    else:
       msg = 'not a quantum object.'
       raise TypeError(msg)

def shapex(x):
    """
    to return the dimention of x
    """
    if torch.is_tensor(x):
          x = x.detach().cpu().numpy()
    if scipy.sparse.issparse(x):
          x = x.toarray()
    if isqx(x):
       return x.shape
    else:
       msg = 'not a quantum object.'
       raise TypeError(msg)

def isqx(x):
    """
    to check if x is a quantum object or not
    """
    if isinstance(x,np.ndarray):
       return True if x.ndim == 2 else False
    else:
       return False

def isbrax(x):
    """
    to check if quantum object x is bra vector
    """
    return True if typex(x)=='bra' else False

def isketx(x):
    """
    to check if quantum object x is ket vector
    """
    return True if typex(x)=='ket' else False

def isoperx(x):
    """
    to check if x is operator (mixed state, Hamiltonian,...)
    """
    return True if typex(x)=='oper' else False

def ishermx(x):
    """
    to check if x is Hermit or not (for oper only)
    """
    if typex(x) == 'oper':
       return True if array_equal(daggx(x),x) else False
    else:
       msg = 'not an operator'
       raise TypeError(msg)

def isnormx(x):
    """ 
    to check if quantum object x is normalzed or not
    """
    if typex(x) != 'oper':
       opx = operx(x)
    else:
       opx = x
    if opx.shape[0] == opx.shape[1]:
       return True if allclose(tracex(opx),1.0) else False
    else:
       msg = 'not a square matrix'
       raise TypeError(msg)

def operx(x):
    """
    to convert a bra or ket vector into oper
    """
    if typex(x) == 'oper':
       return qx(x)
    elif typex(x) == 'bra':
       return qx(dotx(daggx(x),x))
    else:
       return qx(dotx(x,daggx(x)))

def diagx(x):
    # return diagonal of x
    w,v = eigenx(x)
    return np.diag(w)

def conjx(x):
    return np.conj(x)

def transx(x):
    return np.transpose(x)

def daggx(x):
    if torch.is_tensor(x):
         return torch.transpose(torch.conj(x),0,1)
    x = conjx(x)
    return transx(x)

def tracex(x):
    """
    to calculate trace for x
    """
    if typex(x) == 'oper':
       return np.trace(x)
    else:
       msg = 'not a square matrix.'
       raise TypeError(msg)

from numpy import linalg as LA
def eigenx(x):
    """
    eigenvalue and eigenvector

    Parameters:
    -----------
    x : operator

    Returns:
    --------
    w : [..., ...] ndarray
    vk : [[...,...],[...,...]] matrix
        vk[i] is the normalized "ket" eigenvector
        accroding to eigenvelue w[i]
    """
    if typex(x) != 'oper':
       raise TypeError('not an oper')
    else:
       if x.shape[0] != x.shape[1]:
          raise TypeError('not a square oper')
       else:
          w, v = LA.eigh(x)
          vk = []
          for i in range (len(w)):
              vk.append(transx(qx(v[:,i])))
          return w, vk

def groundx(x):
    """
    get ground state for a given Hamiltonian

    Parameters:
    -----------
    x : operator

    Returns:
    --------
    the normalized "ket" eigenvector
        accroding to minimum eigenvelue
    """

    w, vk = eigenx(x)
    m = np.argmin(w)
    return vk[m]

from scipy.linalg import expm,sqrtm
def expx(x):
    # exponential x
    if typex(x) != 'oper':
       raise TypeError('not an oper')
    else:
       return expm(x)
           
def sqrtx(x):
    # square root x
    if typex(x) != 'oper':
       raise TypeError('not an oper')
    else:
       return sqrtm(x)

def l2normx(x):
    """
    to calculate norm 2 of a vector or trace operater
    >>> l2norm = <psi|psi>
    >>> l2morm = √tr(Aˆ†*A)
    """
    if typex(x) != 'oper':
       x = operx(x) #get oper x
    return np.sqrt(tracex(dotx(daggx(x),x)))

def normx(x):
    """
    to normalize x
    use the Frobeius norm
    >>> |psi>/sqrt(<psi|psi>) # for vector states
    >>> (Aˆ†*A)/tr(Aˆ†*A) # for oper
    """
    if typex(x) != 'oper':
       return x/np.sqrt(l2normx(x))
    else:
       if x.shape[0] == x.shape[1]:
          return dotx(daggx(x),x)/l2normx(x)**2
       else:
          raise TypeError('not a square matrix')

# -----------
# hidden defs
from collections.abc import Sequence
def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst), )

    # recurse
    shape = get_shape(lst[0], shape)

    return shape

