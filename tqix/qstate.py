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

__all__ = ['bx','bz',
           'obasis','dbasis','zbasis','dzbasis',
           'coherent','squeezed', 'position','spin_coherent',
           'ghz', 'w', 'dicke', 'random', 
           'add_random_noise','add_white_noise']

from numpy import conj,transpose, kron, sqrt, exp, pi, sin, cos

import numpy as np
from itertools import combinations
from scipy.special import comb,factorial
from tqix.utility import randunit,randnormal,haar
from tqix.qtool import dotx
from tqix.qoper import *
from tqix.qx import *

def bx(d,e = 0):
    """ to generate an orthogonal basis of d dimension at e
    For example:
    base(2,0) = [[1]
                 [0]
                 [0]]
    """
    if (not isinstance(d,int)) or d < 0:
       raise ValueError("d must be integer d >= 0")
    if (not isinstance(e,int)) or e < 0:
       raise ValueError("e must be interger e>= 0")
    if e > d-1:
       raise ValueError("basis vector index need to be in d-1")
    ba = np.zeros((d,1)) #column vector
    ba[e,0] = 1.0
    return qx(ba)

def obasis(d,e = 0):
    #old version
    if (not isinstance(d,int)) or d < 0:
       raise ValueError("d must be integer d >= 0")
    if (not isinstance(e,int)) or e < 0:
       raise ValueError("e must be interger e>= 0")
    if e > d-1:
       raise ValueError("basis vector index need to be in d-1")
    print('Warnings: obasis(d,e) is an old version, please use bx(d,e) instead.')
    ba = np.zeros((d,1)) #column vector
    ba[e,0] = 1.0
    return qx(ba)

def dbasis(d,e = 0):
    # generate a dual basis
    return daggx(bx(d,e))

def bz(j,m):
    """to generate Zeeman basis |j,m> of Sz spin operator
    ie., Sz|j,m> = j|j,m>
    Parameters
    -----------
    j: real number
    m: integer number
    """
    return qx(bx(int(2*j+1),int(j-m)))

def zbasis(j,m):
    #old version
    print('Warnings: zbasis(j,m) is an old version, please use bz(d,e) instead.')
    return qx(bx(int(2*j+1),int(j-m)))

def dzbasis(j,m):
    # generate a dual Zeeman basis
    return daggx(bz(j,m))

def coherent(d,alpha):
    """ to generate coherent state
    Parameters
    ----------
    alpha: complex number
    d: number of dimensions
    """
    ground = bx(d,0)
    coherent = dotx(displacement(d,alpha), ground)
    coherent = normx(coherent)
    return qx(coherent)
    
def squeezed(d,alpha,beta):
    """ to generate squeezed state
    Parameters
    ----------
    alpha: complex number
    d: number of dimensions
    """
    ground = bx(d,0)
    squeezed = dotx(displacement(d,alpha),
               squeezing(d,beta),ground)
    squeezed = normx(squeezed)
    return qx(squeezed)

def position(d,x):
    """ to generate position state
    ie., X|x> = x|x>
    Parameters
    -----------
    x: position coordinate, real number
    d: number of dimensions
    """
    ground = bx(d,0)
    power = -0.5 * dotx(raising(d),raising(d)) + \
            sqrt(2) * x * raising(d)
    exponential = expx(power)
    position = exp(-x**2/2) /(pi ** (1./4)) * dotx(exponential, ground)
    position = normx(position)
    return qx(position)

def spin_coherent(j,theta,phi):
    """ to generate spin coherent state 
    defined as |j,theta,phi> = exp(i*phi*Sz)*exp(i*theta*Sy)|j,j>
    Parameters
    -----------
    j: real number
    theta: polar angle
    phi: azimuthal angle
    z_rotate = expx(1j * phi * soper(j,'z'))
    y_rotate = expx(1j * theta * soper(j,'y'))
    total_rotate = dotx(z_rotate,y_rotate)
    return qx(dotx(total_rotate,spinx(j,j)))
    """
    state = 0.0
    array = np.arange(-j,j+1,1)
    for m in array:
        state += sqrt(factorial(2*j)/(factorial(j+m)*factorial(j-m)))\
                * (cos(theta/2.0))**(j+m) * (sin(theta/2.0))**(j-m)\
                * exp(-1j*m*phi) * bz(j,m)
    return state

def ghz(n):
    """ to generate GHZ state
    Parameters
    ----------
    n: number of qubits

    Return: GHZ state, 
    ie. (|00...0> + |11...1>)/sqrt(2)

    """
    dim = 2**n
    up,down = _up_down()
    ups,upd = up,down
    for i in range(n-1):
        ups = kron(ups,up)
        upd = kron(upd,down)
    GHZ = (ups+upd)/sqrt(2.0)
    return qx(GHZ)

def w(n):
    """ to generate W state
    Parameters
    ----------
    n: number of qubits

    Return: dicke(n,1)

    """
    return dicke(n,1)

def dicke(n,e):
    """ to generate Dicke state
    Parameters
    ----------
    n: number of qubits
    e: excited qubits

    Return: dicke state

    """
    dim = 2**n
    comb_array = _place_ones(n,e)
    #array of possible values of qubits
    row, col = comb_array.shape
    temp_sum = np.zeros((dim,1))

    for i in range(row):
        temp_vec = _upside_down(comb_array[i,0])
        for j in range(1,col):
            temp_vec0 = kron(temp_vec,_upside_down(comb_array[i,j]))
            temp_vec = temp_vec0
        temp_sum += temp_vec

    dicke = temp_sum/(np.sqrt(row))
    return qx(dicke)

def random(d):
    """ to generate a random state
    Parameters:
    ----------
    d: dimension

    Return: random state

    """
    rpsi = np.zeros((d,1))
    ipsi = np.zeros((d,1))
    for i in range (d):
        rpsi[i,0] = randunit()
        ipsi[i,0] = randunit()
    ppsi = rpsi + 1j*ipsi
    ppsi = normx(ppsi)
    M = haar(d)
    prime = dotx((eyex(d)+M),ppsi)
    prime = normx(prime)
    return qx(prime)    

def add_random_noise(psi,m = 0.0,st = 0.0):
    """ to generate 1 perturbed random state from psi
    Parameters:
    ----------
    d: dims
    m: mean
    st: standard derivative
    """
    if isqx(psi):
       dim = psi.shape[0]
       per = [randnormal(m,st,dim)+1j*randnormal(m,st,dim)]
       if typex(psi)=='ket':
          per = daggx(per) #to get ket
       elif typex(psi)=='oper':
          per = dotx(daggx(per),per)

       psi = psi + per
       psi = normx(psi)
       return qx(psi)
    else:
       msg = 'psi is not a quantum object'
       raise TypeError(msg)

def add_white_noise(state,p = 0.0):
    """ add white noise to quantum state
    Parameters:
    ----------
    state: quantum state
    p: error
    
    Return
    ------
    (1-p)*state + p*I/d
    """
    if typex(state) != 'oper':
       state = operx(state)
    dim = state.shape[0]
    return qx((1-p)*state+p*eyex(dim)/dim)

#####
def _up_down():
    # to generate up and down state
    up = np.zeros((2,1))
    up[0,0] = 1.0
    down = np.zeros((2,1))
    down[1,0] = 1.0
    return qx(up),qx(down)

def _upside_down(a):
    """
    return |0> or |1> or none
    """
    up, down = _up_down()
    if a == 0:
        return up
    elif a == 1:
        return down
    else:
        msg = 'Out of acceptable range'
        raise TypeError(msg)

#from math import comb
def _place_ones(size,count):
    """
    return an array of possible values of Dicke state
    Eg.
    place_ones(3,2) <=> |110>, |101> |011>
    [[1. 1. 0.]
     [1. 0. 1.]
     [0. 1. 1.]]
    """
    c = int(comb(size,count))
    result = np.zeros((c,size))
    k = 0
    for positions in combinations(range(size),count):
        for i in positions:
            result[k,i] = 1
        k += 1
    return result

