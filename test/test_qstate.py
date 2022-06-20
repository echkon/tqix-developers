from tqix import *
import numpy as np
from tqix.pis import *

N = 3

print('test circuit')
qc = circuit(N)
state = qc.state
print(state)
print(typex(state))

print('test gate with initial state')
qc.RZ(np.pi/3)
print(qc.state)

print('test with dicke_ghz state')
init_state = dicke_ghz(N)
qc = circuit(N,None,None,init_state)
state = qc.state
print(state)
print(typex(state))

print('test gate with dicke_ghz state')
qc.RZ(np.pi/3)
qc.RY(np.pi/5)
qc.RZ(np.pi/4)
qc.RX(np.pi/3)
print(qc.state.toarray())