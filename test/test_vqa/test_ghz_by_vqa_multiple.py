import qiskit
import numpy as np
import sys

import tqix as tq

# define number of qubits
num_qubits = 3
tm = 3.0

phases = [np.pi/6.,np.pi/6.,np.pi/6.]

#Get Axyz from intergate
# call angular momentum operator
[jx, jy, jz] = tq.joper(num_qubits)

# create ghz_minmax
ghx = tq.ghz_minmax(jx)
ghy = tq.ghz_minmax(jy)
ghz = tq.ghz_minmax(jz)

state = tq.normx(ghx + ghy + ghz)
print(state)
state_dp = tq.markovian_chl(state,tm,y=0.1)
print(state_dp)

# calculate qfim
h_opt = [jx, jy, jz]
c_opt = phases
t = tm
qfim = tq.qfimx(state_dp,h_opt,c_opt,t)
print(qfim)

# calculate quantum bound
qb = tq.qboundx(state_dp,h_opt,c_opt,t)
print(qb)

