import numpy as np

import tqix as tq

# define number of qubits
num_qubits = 5
tm = 4.0
y = 1.0

phases = [np.pi/6.,np.pi/6.,np.pi/6.]

#Get Axyz from intergate
# call angular momentum operator
[jx, jy, jz] = tq.jpauli(num_qubits)

# create ghz_minmax
ghx = tq.ghz_minmax(jx)
ghy = tq.ghz_minmax(jy)
ghz = tq.ghz_minmax(jz)

#state = tq.normx(ghx+ghy+ghz)
# thu voi star
import qiskit
qc = qiskit.QuantumCircuit(5, 5)
#print(tq.qx(tq.vqa.state_vector(tq.vqa.star_graph(qc,None,None))))
state = tq.qx([[ 0.1767767+0.j],[ 0.1767767+0.j ],[  0.1767767+0.j],[  -0.1767767+0.j],[ 
   0.1767767+0.j ],[ -0.1767767+0.j ],[  0.1767767+0.j ],[  0.1767767+0.j],[ 
   0.1767767+0.j],[  -0.1767767+0.j ],[  0.1767767+0.j ],[  0.1767767+0.j],[ 
   0.1767767+0.j],[   0.1767767+0.j],[   0.1767767+0.j],[  -0.1767767+0.j],[ 
   0.1767767+0.j],[  -0.1767767+0.j],[   0.1767767+0.j],[   0.1767767+0.j],[ 
   0.1767767+0.j],[   0.1767767+0.j],[   0.1767767+0.j ],[ -0.1767767+0.j],[ 
   0.1767767+0.j],[   0.1767767+0.j ],[  0.1767767+0.j],[  -0.1767767+0.j],[ 
   0.1767767+0.j],[  -0.1767767+0.j ],[  0.1767767+0.j],[   0.1767767+0.j]])
print(state)

# calculate qfim
h_opt = [jx, jy, jz]
c_opt = phases
ts = np.linspace(0.1,tm,30)
qbmk = []
qbnmk = []

for t in ts:
    #state_mk = tq.markovian_chl(state,t,y)
    #qbmk.append(tq.qboundx(state_mk,h_opt,c_opt,t))
    state_nmk = tq.nonmarkovian_chl(state,t,y)
    qbnmk.append(tq.qboundx(state_nmk,h_opt,c_opt,t))

print(qbnmk)
# find min qbound
#mar_min = min(qbmk)
#mar_index = np.argmin(qbmk)
nonmar_min = min(qbnmk)
nonmar_index = np.argmin(qbnmk)

#print(ts[mar_index],mar_min)
print(ts[nonmar_index],nonmar_min)


#plot figure to check min
import matplotlib.pyplot as plt
#first plot
plt.subplot(1, 2, 1)
plt.ylim(0,4)
#plt.plot(ts, qbmk, '-.',label='Mar_0.5_pi_6')
plt.plot(ts, qbnmk, '-',label='Mar_0.5_pi_6')
plt.savefig('multiple_N5_time_tqix.eps')

