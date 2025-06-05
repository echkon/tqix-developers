from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt
import os
N=100
THETA = [0, np.pi]
PHI = [0, 2* np.pi]
angles = np.linspace(0,0.5,100).tolist()

if os.path.isdir("./OAT"):
      pass 
else:
      os.mkdir("./OAT")

if os.path.isdir("./TNT"):
      pass 
else:
      os.mkdir("./TNT")

if os.path.isdir("./GMS"):
      pass 
else:
      os.mkdir("./GMS")

if os.path.isdir("./TAT"):
      pass 
else:
      os.mkdir("./TAT")

# OAT
OAT_xi_2_S = []
OAT_xi_2_R = []
for theta in angles:
      qc = circuit(N)
      qc.RN(np.pi/2,0)
      qc.OAT(theta,"Z")
      OAT_xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
      OAT_xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

# plot sphere 
for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
     qc = circuit(N)
     qc.RN(np.pi/2,0)
     qc.OAT(theta,"Z")
     husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./OAT",fname =str(theta)+"husimi_sphere.eps",view=(180,0))


#TNT
TNT_xi_2_S = []
TNT_xi_2_R = []

for theta in angles:
      qc = circuit(N)
      qc.RN(np.pi/2,0)
      omega = N*theta
      qc.TNT(theta,omega=omega,gate_type="ZX")
      TNT_xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
      TNT_xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

# plot sphere 
for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
     qc = circuit(N)
     qc.RN(np.pi/2,0)
     qc.TNT(theta,"ZX")
     husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./TNT",fname =str(theta)+"husimi_sphere.eps",alpha=1,view=(180,0))


#GMS
GMS_xi_2_S = []
GMS_xi_2_R = []
phi = np.pi/4 
for theta in angles:
      qc = circuit(N)
      qc.GMS(theta,phi)
      GMS_xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
      GMS_xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

# plot sphere 
for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
     qc = circuit(N)
     qc.GMS(theta,phi)
     husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./GMS",fname =str(theta)+"husimi_sphere.eps",view=(-90,0))

# TAT
TAT_xi_2_S = []
TAT_xi_2_R = []
for theta in angles:
       qc = circuit(N)
       qc.RN(np.pi/2,0)
       qc.TAT(theta,"ZY")
       TAT_xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
       TAT_xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

# plot sphere 
for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
     qc = circuit(N)
     qc.RN(np.pi/2,0)
     qc.TAT(theta,"ZY")
     husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./TAT",fname =str(theta)+"husimi_sphere.eps",view=(180,0))

ax = plt.gca() 
ax.plot(angles, OAT_xi_2_S,'c-o',label=r'$10log_{10}(\xi^{2}_{S})-OAT$')
ax.plot(angles, TNT_xi_2_S,'r-s',label=r'$10log_{10}(\xi^{2}_{S})-TNT$')
ax.plot(angles, TAT_xi_2_S,'g-*',label=r'$10log_{10}(\xi^{2}_{S})-TAT$')
ax.plot(angles, GMS_xi_2_S,'y-o',label=r'$10log_{10}(\xi^{2}_{S})-GMS$')
ax.plot(angles, OAT_xi_2_R,'c--o',label=r'$10log_{10}(\xi^{2}_{R})-OAT$')
ax.plot(angles, TNT_xi_2_R,'r--s',label=r'$10log_{10}(\xi^{2}_{R})-TNT$')
ax.plot(angles, TAT_xi_2_R,'g--*',label=r'$10log_{10}(\xi^{2}_{R})-TAT$')
ax.plot(angles, GMS_xi_2_R,'y--o',label=r'$10log_{10}(\xi^{2}_{R})-GMS$')

ax.set_xlabel("theta")
ax.set_ylabel("Db")
lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
dirname= ""
fname ="xi_2_S_R_graph.eps"
plt.savefig(os.path.join(dirname,fname))
plt.close()