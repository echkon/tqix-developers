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

__all__ = ['eyex','soper','sigmax','sigmay','sigmaz',
            'sigmap','sigmam','lowering','raising',
            'displacement','squeezing',
            'joper','jpauli','jnoper', 
            'dephasing_chl', 'bitflip_chl','phaseflip_chl', 
            'depolarizing_chl','markovian_chl',
            'nonmarkovian_chl']

import numpy as np
from numpy import conj
from scipy import sparse
import tqix 

def eyex(n):
    # to gererate an n x n identity matrix
    return np.identity(n)

#
# spin operater
#
def soper(s,*args):
    # spin matrices
    # represented in the Zeeman basis
    # s : float: 1/2, 1, 3/2,...
    # args : 'x','y','z','+','-'
    if not args:
        return soper(s,'x'),soper(s,'y'),soper(s,'z')
    if args[0] == '+':
        M = _sp(s)
    elif args[0] == '-':
        M = tqix.qobj.daggx(_sp(s))
    elif args[0] == 'x':
        M = 0.5 * (_sp(s) + tqix.qobj.daggx(_sp(s)))
    elif args[0] == 'y':
        M = -0.5 * 1j * (_sp(s) - tqix.qobj.daggx(_sp(s)))
    elif args[0] == 'z':
        M = _sz(s)
    else:
        raise TypeError('Invalid type')
    return M

def _sz(s):
    # define sz
    d = int(2*s + 1)
    data = np.zeros((d,d),dtype = complex)
    for i in range(0,d):
        for j in range(0,d):
            data[i,j] = tqix.utility.krondel(s-i,s-j)*(s-i)
    return data

def _sp(s):
    # define s+
    d = int(2*s + 1)
    data = np.zeros((d,d),dtype = complex)
    for i in range(0,d):
        for j in range(0,d):
            data[i,j] = tqix.utility.krondel(s-i,s-j+1)*\
            np.sqrt(s*(s+1)-(s-i)*(s-j))
    return data

#
# Pauli matrices
#
def sigmax():
    # Pauli matrix sigma x
    return 2.0 * soper(1.0 / 2, 'x')

def sigmay():
    # Pauli matrix sigma y
    return 2.0 * soper(1.0 / 2, 'y')

def sigmaz():
    # Pauli matrix sigma z
    return 2.0 * soper(1.0 / 2, 'z')

def sigmap():
    # Sigma plus
    return (sigmax() + 1j*sigmay())/2.0

def sigmam():
    # Sigma minus
    return (sigmax() -1j*sigmay())/2.0

#
# annihilation operator or the lowering operator
#
def lowering(d):
    # lowering operator
    # d : dimension
    if not isinstance(d,(int, np.integer)):
        raise ValueError("dimension must be integer")
    M = np.zeros((d,d),dtype = float)
    for i in range (d-1):
        M[i,i+1] = np.sqrt(i + 1.0)
    return M

#
# creation operator or the raising operator
#
def raising(d):
    # raising operator
    # d : dimension
    if not isinstance(d,(int, np.integer)):
        raise ValueError("dimension must be integer")
    return tqix.qobj.daggx(lowering(d))
    
#
# displacement operator
#
def displacement(d,alpha):
    power = alpha*raising(d) - conj(alpha)*lowering(d)
    M = tqix.qobj.expx(power)
    return M

#
# squeezing operator
#
def squeezing(d,beta):
    power = 0.5 * (conj(beta)*tqix.qtool.dotx(lowering(d),lowering(d))\
            - beta*tqix.qtool.dotx(raising(d),raising(d)))
    M = tqix.qobj.expx(power)
    return M
#
# collective spin operators
#
def joper(N,*args):
    # collective spin operators
    # N: integer: number of spin-1/2
    # args: 'x','y','z','+','-'

    if not args:
        return joper(N,'x'),joper(N,'y'),joper(N,'z')
    if args[0] == 'x':
        s = 0.0
        for i in range(N): 
            s += jnoper(N,i,'x')           
        return s/2. 

    if args[0] == 'y':
        s = 0.0
        for i in range(N):
            s += jnoper(N,i,'y')
        return s/2.

    if args[0] == 'z':
        s = 0.0
        for i in range(N):
            s += jnoper(N,i,'z')
        return s/2. 

    if args[0] == 'p':
        return joper(N,'x')+1j*joper(N,'y')

    if args[0] == 'm':
        return joper(N,'x')-1j*joper(N,'y')

def jpauli(N,*args):
    # collective Pauli operators
    # N: integer: number of spin-1/2
    # args: 'x','y','z','+','-'

    if not args:
        return jpauli(N,'x'),jpauli(N,'y'),jpauli(N,'z')
    if args[0] == 'x':
        s = 0.0
        for i in range(N): 
            s += jnoper(N,i,'x')           
        return s 

    if args[0] == 'y':
        s = 0.0
        for i in range(N):
            s += jnoper(N,i,'y')
        return s

    if args[0] == 'z':
        s = 0.0
        for i in range(N):
            s += jnoper(N,i,'z')
        return s

    if args[0] == 'p':
        return jpauli(N,'x')+1j*jpauli(N,'y')

    if args[0] == 'm':
        return jpauli(N,'x')-1j*jpauli(N,'y')

def jnoper(N,i,*args):
    # spin jn operators @ site i
    # N: interager: number of spin-1/2
    # args: 'x','y','z','+','-'

    if not args:
        raise TypeError("please put 'x' or 'y' or 'z'")
    if args[0] == 'x':
        if i < 0 or i >= N:
            raise TypeError('i out of range: [0,N-1]')
        elif i == 0:
            res = sigmax()
            for j in range(1,N):
                res = tqix.qtool.tensorx(res,eyex(2))
        else:
            res = eyex(2)
            for j in range(1,i):
                res = tqix.qtool.tensorx(res,eyex(2))
            res = tqix.qtool.tensorx(res,sigmax())
            for j in range(i,N-1):
                res = tqix.qtool.tensorx(res,eyex(2))
        return res

    if args[0] == 'y':
        if i < 0 or i >= N:
            raise TypeError('i out of range: [0,N-1]')
        elif i == 0:
            res = sigmay()
            for j in range(1,N):
                res = tqix.qtool.tensorx(res,eyex(2))
        else:
            res = eyex(2)
            for j in range(1,i):
                res = tqix.qtool.tensorx(res,eyex(2))
            res = tqix.qtool.tensorx(res,sigmay())
            for j in range(i,N-1):
                res = tqix.qtool.tensorx(res,eyex(2))
        return res

    if args[0] == 'z':
        if i < 0 or i >= N:
            raise TypeError('i out of range: [0,N-1]')
        elif i == 0:
            res = sigmaz()
            for j in range(1,N):
                res = tqix.qtool.tensorx(res,eyex(2))
        else:
            res = eyex(2)
            for j in range(1,i):
                res = tqix.qtool.tensorx(res,eyex(2))
            res = tqix.qtool.tensorx(res,sigmaz())
            for j in range(i,N-1):
                res = tqix.qtool.tensorx(res,eyex(2))
        return res

# add noise channel via kraus operators
def dephasing_chl(x, lamb):
    """Add dephasing to the system

    Args:
        - x quantum system state
        - lamb: noise
    
    Return
        - x
    """ 
    # number of qubits 
    N = tqix.qobj.qubitx(x)
    
    # kraus operators
    kraus1 = np.array([[1, 0],[0, np.sqrt(1 - lamb)]])
    kraus2 = np.array([[0, 0],[0, np.sqrt(lamb)]])
    
    lkraus1 = tqix.qtool.itensorx(kraus1, N)
    lkraus2 = tqix.qtool.itensorx(kraus2, N)
    
    # apply Kraus to qobj x
    x_csr = sparse.csr_matrix(tqix.operx(x))
    for i in range(N):
        x_csrn = lkraus1[i] @ x_csr @ lkraus1[i] \
                    + lkraus2[i] @ x_csr @ lkraus2[i]
        x_csr = x_csrn
    
    return x_csr

def bitflip_chl(x, lamb):
    """Add bitflip to the system

    Args:
        - x quantum system state
        - lamb: noise
    
    Return
        - x (sparse matrix)
    """ 
    # number of qubits 
    N = tqix.qobj.qubitx(x)
    
    # kraus operators
    kraus1 = np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]])
    kraus2 = np.sqrt(lamb) * np.array([[0, 1], [1, 0]])
    
    lkraus1 = tqix.qtool.itensorx(kraus1, N)
    lkraus2 = tqix.qtool.itensorx(kraus2, N)
    
    # apply Kraus to qobj x
    x_csr = sparse.csr_matrix(tqix.operx(x))
    for i in range(N):
        x_csrn = lkraus1[i] @ x_csr @ lkraus1[i] \
                    + lkraus2[i] @ x_csr @ lkraus2[i]
        x_csr = x_csrn
    
    return x_csr

def phaseflip_chl(x, lamb):
    """Add phaseflip to the system

    Args:
        - x quantum system state
        - lamb: noise
    
    Return
        - x (sparse matrix)
    """ 
    # number of qubits 
    N = tqix.qobj.qubitx(x)
    
    # kraus operators
    kraus1 = np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]])
    kraus2 = np.sqrt(lamb) * np.array([[1, 0], [0, -1]])
    
    lkraus1 = tqix.qtool.itensorx(kraus1, N)
    lkraus2 = tqix.qtool.itensorx(kraus2, N)
    
    # apply Kraus to qobj x
    x_csr = sparse.csr_matrix(tqix.operx(x))
    for i in range(N):
        x_csrn = lkraus1[i] @ x_csr @ lkraus1[i] \
                    + lkraus2[i] @ x_csr @ lkraus2[i]
        x_csr = x_csrn
    
    return x_csr

def depolarizing_chl(x, lamb):
    """Add depolarizing to the system

    Args:
        - x quantum system state
        - lamb: noise
    
    Return
        - x (sparse matrix)
    """ 
    # number of qubits 
    N = tqix.qobj.qubitx(x)
    
    # kraus operators
    kraus1 = np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]])
    kraus2 = np.sqrt(lamb/3.) * np.array([[0, 1], [1, 0]])
    kraus3 = np.sqrt(lamb/3.) * np.array([[0, -1j], [1j, 0]])
    kraus4 = np.sqrt(lamb/3.) * np.array([[1, 0], [0, -1]])
    
    lkraus1 = tqix.qtool.itensorx(kraus1, N)
    lkraus2 = tqix.qtool.itensorx(kraus2, N)
    lkraus3 = tqix.qtool.itensorx(kraus3, N)
    lkraus4 = tqix.qtool.itensorx(kraus4, N)
    
    # apply Kraus to qobj x
    x_csr = sparse.csr_matrix(tqix.operx(x))
    for i in range(N):
        x_csrn = lkraus1[i] @ x_csr @ lkraus1[i] \
                    + lkraus2[i] @ x_csr @ lkraus2[i] \
                    + lkraus3[i] @ x_csr @ lkraus3[i] \
                    + lkraus4[i] @ x_csr @ lkraus4[i]
        x_csr = x_csrn
    
    return x_csr

def markovian_chl(x, t, y):
    """Add markovian to the system

    Args:
        - x quantum system state
        - t: time
        - y: noise
    
    Return
        - x (sparse matrix)
    """ 
    # number of qubits 
    N = tqix.qobj.qubitx(x)
    
    # kraus operators
    qt = 1 - np.exp(-y*t)
    kraus1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    kraus2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    lkraus1 = tqix.qtool.itensorx(kraus1, N)
    lkraus2 = tqix.qtool.itensorx(kraus2, N)
    
    # apply Kraus to qobj x
    x_csr = sparse.csr_matrix(tqix.operx(x))
    for i in range(N):
        x_csrn = lkraus1[i] @ x_csr @ lkraus1[i] \
                    + lkraus2[i] @ x_csr @ lkraus2[i]
        x_csr = x_csrn
    
    return x_csr

def nonmarkovian_chl(x, t, y, tc = 20.0):
    """Add nonmarkovian to the system

    Args:
        - x quantum system state
        - t: time
        - y: noise
    
    Return
        - x (sparse matrix)
    """ 
    # number of qubits 
    N = tqix.qobj.qubitx(x)
    
    # kraus operators
    # tc = 20.0 #fixed in rXiv:2305.08289
    qt = 1 - np.exp(-y*t**2/(2*tc))
    
    kraus1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    kraus2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    lkraus1 = tqix.qtool.itensorx(kraus1, N)
    lkraus2 = tqix.qtool.itensorx(kraus2, N)
    
    # apply Kraus to qobj x
    x_csr = sparse.csr_matrix(tqix.operx(x))
    for i in range(N):
        x_csrn = lkraus1[i] @ x_csr @ lkraus1[i] \
                    + lkraus2[i] @ x_csr @ lkraus2[i]
        x_csr = x_csrn
    
    return x_csr