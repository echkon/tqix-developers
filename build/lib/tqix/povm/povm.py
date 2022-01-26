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

import numpy as np
from tqix.povm.pauli import *
from tqix.povm.stoke import *
from tqix.povm.mub import *
from tqix.povm.sic import *

__all__: ['_pauli','_stoke','_mub','_sic']

def _pauli(n):
    # to return Pauli POVM of
    # n qubits
    return _pauli_(n)

def _stoke(n):
    # to return Stoke POVM of
    # n qubits
    return _stoke_(n)

def _mub(d):
    # to return MUB POVM of
    # d dimension
    return _mub_(d)

def _sic(d):
    # to return SIC POVM of
    # d dimension
    return _sic_(d)

