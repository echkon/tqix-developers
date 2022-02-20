
from tqix.pis import *
from tqix import *
import numpy as np
import scipy.sparse as sparse

N=3
print('test with dicke_ghz state')
init_state = dicke_ghz(N)
qc = circuit(N,init_state)
state = qc.state
print(state)
print(typex(state))

print('test gate with dicke_ghz state')
qc.GMS(np.pi/3,40,"XY").RZ(np.pi/5).RY(np.pi/8).OAT(np.pi/4,"X").TAT(np.pi/9,"XZ").TNT(np.pi/3,"YZ",omega=12)
print(qc.state)
prob = qc.measure(num_shots=100000)
print(prob)
print(prob.sum())