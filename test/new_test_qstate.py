
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
qc.RZ(np.pi/3).RY(np.pi/5).RZ(np.pi/4)
print(qc.state)
prob = qc.measure(num_shots=1000000)
print(prob)
print(prob.sum())