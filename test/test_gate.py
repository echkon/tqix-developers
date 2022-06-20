
from tqix import *
import numpy as np
import time 
from tqix.pis import *
import time 
# n_qubit = 25
# times = []
# for num_q in range(n_qubit):
#     num_q += 1
num_q=100
qc = circuit(num_q)
start = time.time()
qc.TNT(np.pi/3,omega=np.pi/3,gate_type="XY",noise=0.05,num_processes=25)
print(qc.state)
print(qc.state.diagonal().sum())
print("time:",time.time()-start)