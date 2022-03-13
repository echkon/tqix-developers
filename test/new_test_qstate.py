
from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt
import os
N=100
THETA = [0, np.pi]
PHI = [0, 2* np.pi]
xi_2_S = []
xi_2_R = []
angles = np.linspace(0,0.13,500).tolist()

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

#OAT

for theta in angles:
      qc = circuit(N)
      qc.OAT(theta,"X")
      xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
      xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

ax = plt.gca() 
ax.plot(angles, xi_2_S,label=r'$10log_{10}(\xi^{2}_{S})$')
ax.plot(angles, xi_2_R, '--',label=r'$10log_{10}(\xi^{2}_{R})$')
ax.set_xlabel("theta")
ax.set_ylabel("Db")
ax.legend()
dirname= "./OAT"
fname ="xi_2_graph.eps"
ax.savefig(os.path.join(dirname,fname))

for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
     qc = circuit(N)
     qc.OAT(theta,"X")
     husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./OAT",fname =str(theta)+"husimi_sphere.eps",alpha=0.5,view=(-90,0))

#TNT
omega = 0.05
for theta in angles:
      qc = circuit(N)
      qc.TNT(theta,omega,"XZ")
      xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
      xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

ax = plt.gca() 
ax.plot(angles, xi_2_S,label=r'$10log_{10}(\xi^{2}_{S})$')
ax.plot(angles, xi_2_R, '--',label=r'$10log_{10}(\xi^{2}_{R})$')
ax.set_xlabel("theta")
ax.set_ylabel("Db")
ax.legend()
dirname= "./TNT"
fname ="xi_2_graph.eps"
ax.savefig(os.path.join(dirname,fname))

for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
     qc = circuit(N)
     qc.TNT(theta,omega,"XZ")
     husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./TNT",fname =str(theta)+"husimi_sphere.eps",alpha=0.5,view=(-90,0))

#GMS
phi = 0.05
for theta in angles:
      qc = circuit(N)
      qc.GMS(theta,phi,"XY")
      xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
      xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

ax = plt.gca() 
ax.plot(angles, xi_2_S,label=r'$10log_{10}(\xi^{2}_{S})$')
ax.plot(angles, xi_2_R, '--',label=r'$10log_{10}(\xi^{2}_{R})$')
ax.set_xlabel("theta")
ax.set_ylabel("Db")
ax.legend()
dirname= "./GMS"
fname ="xi_2_graph.eps"
ax.savefig(os.path.join(dirname,fname))

for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
     qc = circuit(N)
     qc.GMS(theta,phi,"XY")
     husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./GMS",fname =str(theta)+"husimi_sphere.eps",alpha=0.5,view=(-90,0))

#TAT
for theta in angles:
      qc = circuit(N)
      qc.TAT(theta,"XY")
      xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
      xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

ax = plt.gca() 
ax.plot(angles, xi_2_S,label=r'$10log_{10}(\xi^{2}_{S})$')
ax.plot(angles, xi_2_R, '--',label=r'$10log_{10}(\xi^{2}_{R})$')
ax.set_xlabel("theta")
ax.set_ylabel("Db")
ax.legend()
dirname= "./TAT"
fname ="xi_2_graph.eps"
ax.savefig(os.path.join(dirname,fname))

for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
     qc = circuit(N)
     qc.TAT(theta,"XY")
     husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./TAT",fname =str(theta)+"husimi_sphere.eps",alpha=0.5,view=(-90,0))