
from tqix import *
import numpy as np
import time 
from tqix.pis import *
import time 
# n_qubit = 25
# times = []
# for num_q in range(n_qubit):
#     num_q += 1
num_q=5
qc = circuit(num_q,num_process=10)
start = time.time()
qc.RX(np.pi/3,noise=0.05)
print(get_xi_2_S(qc))
print(qc.state)
print(qc.state.diagonal().sum())
print("time:",time.time()-start)