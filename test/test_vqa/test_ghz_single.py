import qiskit
import numpy as np
import sys

import tqix as tq

# define number of qubits
num_qubits = 4
tm = 1.0

phases = [np.pi/3.]

#Get Axyz from intergate
# call angular momentum operator
[jx, jy, jz] = tq.joper(num_qubits)

# create ghz_minmax
ghx = tq.ghz_minmax(jx)
ghy = tq.ghz_minmax(jy)
ghz = tq.ghz_minmax(jz)

state = ghz
print(state)

print(tq.dephasing_chl(state,lamb = 0.2))

# calculate qfim
h_opt = [jx]
c_opt = phases
t = tm
qfim = tq.qfimx(state,h_opt,c_opt,t)
print('qfim', qfim)

# calculate quantum bound
qb = tq.qboundx(state,h_opt,c_opt,t)
print(qb)


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

