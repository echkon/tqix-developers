
from tqix.pis import *
from tqix import *
import numpy as np
import scipy.sparse as sparse

from tqix.pis.noise import add_noise

N=3
qc = circuit(N)
state = qc.state
init_state = dotx(state,daggx(state))
qc = circuit(N,init_state)
qc = add_noise(qc)
qc.state.toarray()

qc.RX(np.pi/3)
# qc.GMS(np.pi/3,40,"XY").RZ(np.pi/5).RY(np.pi/8).OAT(np.pi/4,"X").TAT(np.pi/9,"XZ").TNT(np.pi/3,"YZ",omega=12)
print(sum(qc.state.diagonal()))
prob = qc.measure(num_shots=10000000)
print(prob)
print(prob.sum())