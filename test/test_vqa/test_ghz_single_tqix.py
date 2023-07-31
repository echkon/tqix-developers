import qiskit
import numpy as np
import sys

import tqix as tq

# define number of qubits
num_qubits = 4
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

dp = []
bit = []
phase = []
dpo = []
lambs = np.linspace(0,1,20)

for lamb in lambs:
    state_dp = tq.dephasing_chl(state,lamb)
    dp.append(tq.qfimx(state_dp,h_opt,c_opt,t)[0,0])
    state_bit = tq.bitflip_chl(state,lamb)
    bit.append(tq.qfimx(state_bit,h_opt,c_opt,t)[0,0])
    state_phase = tq.phaseflip_chl(state,lamb)
    phase.append(tq.qfimx(state_phase,h_opt,c_opt,t)[0,0])
    state_dpo = tq.depolarizing_chl(state,lamb)
    dpo.append(tq.qfimx(state_dpo,h_opt,c_opt,t)[0,0])

# plot figure
import matplotlib.pyplot as plt
plt.plot(lambs,dp, label = 'dephasing')
plt.plot(lambs,bit,label = 'bit')
plt.plot(lambs,phase,label = 'phase')
plt.plot(lambs,dpo,label = 'depolarizing')
plt.legend()
plt.show()

"""
# run for N
num_qubits = 3
tm = 3.0
y = 0.5 #fix
phases = [np.pi/6.,np.pi/6.,np.pi/6.]

qbound_mar = []
qbound_nonmar = []
ts = np.linspace(0.1,tm,100)

for t in ts:   
    # set intial circuits
    qcir1_ghz = [tq.ghz, None, None]
    qcir1_star = [tq.star_graph, None, None]   
    qcir2 = [tq.u_phase,t,phases]    
    qcir3_mar = [tq.markovian,t,y]
    qcir3_nonmar = [tq.non_markovian,t,y]


    # input circuit
    qcirs_star_mar =[qcir1_star, qcir2, qcir3_mar]
    qcirs_star_nonmar =[qcir1_star, qcir2, qcir3_nonmar]
        
    # setup a model 
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)   
    
    # quantum bound
    qbound_mar.append(tq.sld_bound(qc.copy(),qcirs_star_mar))
    qbound_nonmar.append(tq.sld_bound(qc.copy(),qcirs_star_nonmar))

    
# find min qbound
mar_min = min(qbound_mar)
mar_index = np.argmin(qbound_mar)
nonmar_min = min(qbound_nonmar)
nonmar_index = np.argmin(qbound_nonmar)

print(ts[mar_index],mar_min)
print(ts[nonmar_index],nonmar_min)
"""

