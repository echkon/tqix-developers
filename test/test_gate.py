
from tqix import *
import numpy as np
import time 
from tqix.pis import *
from tqix.pis import spin_operators
import time 
import big_o
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm

def cal_expm(N):
    s = spin_operators.Sx(N/2)
    theta = 0.1
    return expm(-1j*theta*s)

print(big_o.big_o(cal_expm, big_o.datagen.n_, n_measures=5)[0])

