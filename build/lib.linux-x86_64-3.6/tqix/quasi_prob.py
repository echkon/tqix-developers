"""
>>> this file is a part of tqix: a Toolbox for Quantum in X
                              x: quantum measurement, quantum metrology, 
                                 quantum tomography, and more.
________________________________
>>> copyright (c) 2019 and later
>>> authors: Binho Le
>>> contributors: Quangtuan Kieu
>>> all rights reserved
________________________________
"""

__all__ = ['husimi', 'husimi_spin', 'wigner', 'wigner_spin']

import numpy as np
from scipy import (zeros, array, arange, exp, real, conj, pi,
copy, sqrt, meshgrid, size, polyval, fliplr, conjugate,
cos, sin)
from numpy import pi,exp,arange
from scipy.special import genlaguerre,factorial,sph_harm,binom
from tqix import *

def husimi(state,x_array,y_array):
    """ to calculate Husimi Q Function of a quantum state
    by this formula
    Q(a) = 1/pi * <a|rho|a>
    where |a> is a coherent state, a is complex
    rho is density-matrix form of the input state

    Parameters
    ----------
    state: a quantum state
    x_array: an array of coordinate x
    y_array: an array of coordinate y

    ----------
    Return
    Array of Husimi Q function over array [x_array,y_array]

    """
    if not isoperx(state):
        state = operx(state)
    d = shapex(state)[0] #dimension of state
    M = np.shape(x_array)[0]
    N = np.shape(y_array)[0]

    xx,yy = np.meshgrid(x_array,y_array)
    alpha = (xx + 1j*yy)/(np.sqrt(2))
    value = np.zeros((M,N),dtype=float)

    for i in range(M):
        for j in range(N):
            temp = coherent(d,alpha[i,j])
            qfunc = dotx(daggx(temp),state,temp)/(pi)
            value[i,j] = float(np.real(qfunc))

    return value

def husimi_spin(state,x_array,y_array):
    """ to calculate Husimi Q Function of a quantum state
    by this formula
    Q(a) = 1/pi * <a|rho|a>
    where |a> is a coherent state, a is complex
    rho is density-matrix form of the input state

    Parameters
    ----------
    state: a quantum state
    x_array: an array of coordinate x
    y_array: an array of coordinate y

    ----------
    Return
    Array of Husimi Q function over array [x_array,y_array]

    """
    if not isoperx(state):
        state = operx(state)
    d = shapex(state)[0] #dimension of state
    M = np.shape(x_array)[0]
    N = np.shape(y_array)[0]

    xx,yy = np.meshgrid(x_array,y_array)
    alpha = (xx + 1j*yy)
    value = np.zeros((M,N),dtype=float)

    for i in range(M):
        for j in range(N):
            temp = spin_coherent(float((d-1)/2.0),np.real(alpha[i,j]),np.imag(alpha[i,j]))
            #temp = coherent(d,alpha[i,j])
            qfunc = dotx(daggx(temp),state,temp) /(pi)
            value[i,j] = float(np.real(qfunc))

    return value

def wigner(state,x_array,y_array):
    """ to calculate Wigner Function of a quantum state,
    see formulae (5.105) and (5.108) in 
    Measuring the Quantum State of Light (CUP,1997)
    by Ulf Leonhardt

    Parameters
    ----------
    state: a quantum state
    x_array: an array of coordinate x
    y_array: an array of coordinate y

    ----------
    Return
    Array of Wigner function over array [x_array,y_array]

    """
    if not isoperx(state):
        state = operx(state)
    d = shapex(state)[0] #dimension of state
    M = np.shape(x_array)[0]
    N = np.shape(y_array)[0]

    xx,yy = np.meshgrid(x_array,y_array)
    alpha = (xx + 1j*yy)/(np.sqrt(2))
    value = np.zeros((M,N),dtype=float)

    for i in range(M):
        for j in range(N):
            value[i,j] = float(np.real(_wigner(d,state,alpha[i,j])))
    return value

def _wigner(d,state,alpha):
    """to calculate Wigner Function of a quantum state,
    see formulae (5.105) and (5.108) in 
    Measuring the Quantum State of Light (CUP,1997)
    by Ulf Leonhardt
    """
    result = 0.0
    for i in range(d):
        for j in range(d):
            if (j > i):
                laguerre = genlaguerre(i,j-i)(4 * abs(alpha)**2)
                temp = (-1)**j / (pi) * np.sqrt(factorial(i)/factorial(j))\
                        *exp(-2 * abs(alpha)**2) * (-2*conjx(alpha))**(j-i)\
                        *laguerre
                result += temp * state[i,j]
            else:
                laguerre = genlaguerre(j,i-j)(4 * abs(alpha)**2)
                temp = (-1)**j / (pi) * np.sqrt(factorial(j)/factorial(i))\
                        *exp(-2 * abs(alpha)**2) * (-2*conjx(-alpha))**(i-j)\
                        *laguerre
                result += conjx(temp) * state[i,j]
    return result

def clebsch_coe(j1, j2, j3, m1, m2, m3):

    if m3 != m1 + m2:
        return 0
        
    minval = int(np.max([-j1+j2+m3,-j1+m1,0]))
    maxval = int(np.min([j2+j3+m1,j3-j1+j2,j3+m3]))

    CC = np.sqrt((2.0*j3 + 1.0) * factorial(j3 + j1 - j2) *
            factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) *
            factorial(j3 + m3) * factorial(j3 - m3) /
            (factorial(j1 + j2 + j3 + 1) *
            factorial(j1 - m1) * factorial(j1 + m1) *
            factorial(j2 - m2) * factorial(j2 + m2)))
    SC = 0
    for v in range(minval, maxval + 1):
        SC += (-1.0) ** (v + j2 + m2) / factorial(v) * \
           factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
           factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
           factorial(v + j1 - j2 - m3)
    return CC * SC

def _rho_kq(rho, j, k, q):
    # state in kq basis
    if not isoperx(rho):
        state = operx(rho)
    d = shapex(rho)[0] #dimension of state
    
    rkq = 0j
    jrange = np.linspace(-j,j,d)
    for m1 in jrange: #range(-j, j+1):
        for m2 in jrange: #range(-j, j+1):
            rkq += (-1)**(j - m1 - q) * clebsch_coe(j, j, k, m1, -m2,q) * dotx(daggx(zbasis(j,m1)),rho,zbasis(j,m2))
    return rkq


def wigner_spin(state,x_array,y_array):
    """ to calculate spin Wigner Function of a quantum state,
    see 
    Parameters
    ----------
    state: a quantum state
    x_array: an array of coordinate x
    y_array: an array of coordinate y

    ----------
    Return
    Array of spin Wigner function over array [x_array,y_array]

    """
    if not isoperx(state):
        state = operx(state)
    d = shapex(state)[0] #dimension of state
    j = (d-1)/2
    
    M = np.shape(x_array)[0]
    N = np.shape(y_array)[0]

    xx,yy = np.meshgrid(x_array,y_array)
    alpha = (xx + 1j*yy)

    qfunc = np.zeros_like(xx, dtype=complex)
    for k in range(0,int(2 * j)+1):
         for q in range(-k,k+1):
             qfunc += _rho_kq(state, j, k, q) * sph_harm(q,k,yy,xx)

    return np.real(qfunc)
