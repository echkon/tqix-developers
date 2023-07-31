import qiskit
import numpy as np
import sys

import tqix as tq

# define number of qubits
num_qubits = 5
tm = 1.0

phases = [np.pi/2.]

#Get Axyz from intergate
# call angular momentum operator
[jx, jy, jz] = tq.joper(num_qubits)

# create ghz_minmax
ghx = tq.ghz_minmax(jx)
ghy = tq.ghz_minmax(jy)
ghz = tq.ghz_minmax(jz)

state = ghx

# calculate qfim
h_opt = [jx]
c_opt = phases
t = tm

dp0 = []
dp6 = []
dp3 = []
dp2 = []
lambs = np.linspace(0,1,20)

for lamb in lambs:
    state_dp = tq.dephasing_chl(state,lamb)
    c_opt = [0.0]
    dp0.append(tq.qfimx(state_dp,h_opt,c_opt,t)[0,0])
    c_opt = [np.pi/6.]
    dp6.append(tq.qfimx(state_dp,h_opt,c_opt,t)[0,0])
    c_opt = [np.pi/3.]
    dp3.append(tq.qfimx(state_dp,h_opt,c_opt,t)[0,0])
    c_opt = [np.pi/2.]
    dp2.append(tq.qfimx(state_dp,h_opt,c_opt,t)[0,0])

# plot figure
import matplotlib.pyplot as plt
plt.plot(lambs,dp0, label = '0')
plt.plot(lambs,dp6,label = '6')
plt.plot(lambs,dp3,label = '3')
plt.plot(lambs,dp2,label = '2')
plt.legend()
plt.show()

