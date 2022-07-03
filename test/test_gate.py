
from tqix import *
import numpy as np
import time 
from tqix.pis import *
import time 
import big_o
from scipy.sparse import csc_matrix

N=2
times = []
qc = circuit(N,use_tensor=True)
qc.OAT(np.pi/3,"Z",noise=0.05,num_process=25)  
print(qc.state)

