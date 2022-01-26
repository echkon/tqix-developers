from tqix import *
import numpy as np
from tqix.pis import *

N = 3

print('test circuit')
qc = circuit(N)
state = qc.state
print(state)
print(typex(state))

print('test gate')
RZ(qc,np.pi/3)
print(qc.state)

print('test initial state')
init_state = dotx(state,daggx(state))
qc = circuit(N,init_state)
state = qc.state
print(state)
print(typex(state))

print('test gate with initial state')
RZ(qc,np.pi/3)
print(qc.state)

print('test with dicke_ghz state')
init_state = dicke_ghz(N)
qc = circuit(N,init_state)
state = qc.state
print(state)
print(typex(state))

print('test gate with dicke_ghz state')
RZ(qc,np.pi/3)
print(qc.state)
