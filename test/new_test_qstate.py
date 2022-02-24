
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
qc.GMS(np.pi/3,40,"XY",noise=0.3).RZ(np.pi/5,noise=0.2).RY(np.pi/8,noise=0.5).OAT(np.pi/4,"X",noise=0.1).TAT(np.pi/9,"XZ",noise=0.5).TNT(np.pi/3,"YZ",omega=12,noise=0.2)
print(sum(qc.state.diagonal()))
prob = qc.measure(num_shots=10000)
print(prob)
print(prob.sum())