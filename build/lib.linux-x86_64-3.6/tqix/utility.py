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

__all__ = ['randunit','krondel','randnormal','haar','ndiff']

import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import random
from tqix.qx import *
from scipy import (randn,diagonal,absolute,multiply)
from scipy.linalg import qr

def randunit():
    # to generate a random number [0,1]
    return random.random()

def krondel(i,j):
    # to generate kronecker delta
    return 1 if i==j else 0

def randnormal(m=0.0,s=0.0,n=1):
    """
    to generate normal distribution
    
    Parameters:
    -----------
    m: mean
    s: standard deviation
    n: size
    """
    return np.random.normal(m,s,n)
    
def haar(d):
    """
    to generate random matrix for Haar measure
    see https://arxiv.org/pdf/math-ph/0609050.pdf
    """
    array = (randn(d,d) + 1j*randn(d,d))/np.sqrt(2.0)
    ortho,upper = qr(array)
    diag = diagonal(upper)
    temp = diag/absolute(diag)
    result = multiply(ortho,temp,ortho)
    return result

def ndiff(x,y):
    """
    to calculate numerical differential of y(x)
    input: x, y: arrays,
    output: dy: differential
    """
    dy = np.zeros(x.shape,float)
    dy[0:-1] = np.diff(y)/np.diff(x)
    dy[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])
    return dy
