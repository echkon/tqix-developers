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

__all__ = ['cdf','mc']

import numpy as np
import matplotlib.pyplot as plt

def cdf(f):
    """ comulative distribution function
    input: a function f
    return: average of original function

    """
    dim = np.int(len(f))
    nbin = 100
    binh,binb = np.histogram(f,nbin)
    
    s = 0.0 
    for i in range(np.int(nbin/2)):
         s += binh[i] 
    return s/(np.amax(f)/nbin*dim)/(nbin/2.)

from tqix.utility import randunit,randnormal
def mc(f,niter = 1000):
    """
    Markov chain Monte Carlo simulation
    input: a probability
    return: average of original function
    """
    count = 0
    for i in range (niter):
        if randunit() <= f:
           count += 1
    return count/float(niter)
