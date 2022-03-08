
from tqix.pis import *
from tqix import *
import numpy as np

N=3
qc = circuit(N)
state = qc.state
init_state = dotx(state,daggx(state))
qc = circuit(N,init_state)
qc.OAT(np.pi/3,"Y",noise=0.05)
print(get_xi_2_H("x","z",qc))
print(get_xi_2_S(qc))
print(get_xi_2_R(qc))
print(get_xi_2_D(qc,n=[1,0,3]))
print(get_xi_2_E(qc,n=[1,0,3]))