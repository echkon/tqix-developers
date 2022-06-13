
from tqix import *
import numpy as np
import time 
from tqix.pis import *


# n_qubit = 25
# times = []
# for num_q in range(n_qubit):
#     num_q += 1
num_q=500
start = time.time()
qc = circuit(num_q)
t = time.time()-start
start = time.time()
qc.RZ(np.pi/3)
t = time.time()-start
start = time.time()
qc.RX(np.pi/3)
t = time.time()-start
start = time.time()
qc.RZ(np.pi/3)
t = time.time()-start
# qc.measure(num_shots=1)
