
from tqix.pis import *
from tqix import *
import numpy as np
import scipy.sparse as sparse

from tqix.pis.noise import add_noise

N=13
qc = circuit(N)
state = qc.state
init_state = dotx(state,daggx(state))
qc = circuit(N,init_state)
qc.RN(np.pi/3,40,"XY",noise=0.3)
print(sum(qc.state.diagonal()))
prob = qc.measure(num_shots=10000)
print(prob)
print(prob.sum())