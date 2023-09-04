from tqix import *
from tqix.pis import *
import numpy as np

numq=50

#call the initial circuit
qc = circuit(numq)
psi = qc.state

#print psi
print(psi) #sparse matrix
print(psi.toarray()) #full matrix

#apply the rotation gate RN on the circuit
qc.RN(-np.pi/2,np.pi/4)
psi2 = qc.state

#visualize state
THETA = [0, np.pi]
PHI = [0, 2* np.pi]
husimi_spin_3d(psi.toarray()+psi2.toarray(),
                   THETA ,PHI,cmap = cmindex(1),fname ="husimi_sphere.png",view
                   =(0,0))
#get probability
prob = qc.measure(num_shots=1000)
#plot figure
from matplotlib import pyplot as plt
x = np.arange(0,numq+1,1)
plt.bar(x,prob)
plt.savefig("Pjm.png")


