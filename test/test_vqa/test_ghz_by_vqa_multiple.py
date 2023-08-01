import qiskit
import numpy as np

import tqix as tq

# run for N
num_qubits = 3
tm = 4.0
y = 1.0 #fix
phases = [np.pi/6.,np.pi/6.,np.pi/6.]

qbound_mar = []
qbound_nonmar = []
ts = np.linspace(0.1,tm,50)

for t in ts:   
    # set intial circuits
    qcir1_ghz = [tq.vqa.ghz_cir, None, None]
    qcir1_star = [tq.vqa.star_graph, None, None]   
    qcir2 = [tq.vqa.u_phase,t,phases]    
    qcir3_mar = [tq.vqa.markovian,t,y]
    qcir3_nonmar = [tq.vqa.non_markovian,t,y]


    # input circuit
    qcirs_star_mar =[qcir1_star, qcir2, qcir3_mar]
    qcirs_star_nonmar =[qcir1_star, qcir2, qcir3_nonmar]
        
    # setup a model 
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)   
    
    # quantum bound
    qbound_mar.append(tq.vqa.sld_bound(qc.copy(),qcirs_star_mar))
    qbound_nonmar.append(tq.vqa.sld_bound(qc.copy(),qcirs_star_nonmar))


# find min qbound
mar_min = min(qbound_mar)
mar_index = np.argmin(qbound_mar)
nonmar_min = min(qbound_nonmar)
nonmar_index = np.argmin(qbound_nonmar)

print(ts[mar_index],mar_min)
print(ts[nonmar_index],nonmar_min)

#plot figure to check min
import matplotlib.pyplot as plt
#first plot
plt.subplot(1, 2, 1)
plt.ylim(0,4)
plt.plot(ts, qbound_mar, '-.',label='Mar_0.5_pi_6')
plt.plot(ts, qbound_nonmar, '-',label='Mar_0.5_pi_6')
plt.savefig('multiple_N5_time_vqa.eps')
