
from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt
import os
import qutip
N=3
THETA = [0, np.pi]
PHI = [0, 2* np.pi]
angles = np.linspace(0,0.12,500).tolist()

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
# plt.figure()
# OAT_xi_2_S = []
# OAT_xi_2_R = []
# for theta in angles:
#       qc = circuit(N)
#       qc.OAT(theta,"X")
#       OAT_xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
#       OAT_xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

# ax = plt.gca() 
# ax.plot(angles, xi_2_S,label=r'$10log_{10}(\xi^{2}_{S})$')
# ax.plot(angles, xi_2_R, '--',label=r'$10log_{10}(\xi^{2}_{R})$')
# ax.set_xlabel("theta")
# ax.set_ylabel("Db")
# ax.legend()
# dirname= "./OAT"
# fname ="xi_2_graph.eps"
# plt.savefig(os.path.join(dirname,fname))

# for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
#      qc = circuit(N)
#      qc.OAT(theta,"X")
#      husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./OAT",fname =str(theta)+"husimi_sphere.eps",alpha=0.5,view=(-90,0))


#TNT
angles = np.linspace(0,5,100).tolist()
plt.figure()
TNT_xi_2_S = []
TNT_xi_2_R = []

theta = 0.1
qc = circuit(N)
qc.RN(np.pi/2,0)
qc.TNT(theta,"ZX")
print(get_xi_2_S(qc))

# plot and save \xi graph
# for theta in angles:
#       qc = circuit(N)
#       qc.RN(np.pi/2,0)
#       qc.TNT(theta,"ZX")
#       TNT_xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))

# ax = plt.gca() 
# ax.plot(angles, TNT_xi_2_S,label=r'$10log_{10}(\xi^{2}_{S})$')
# ax.set_xlabel("theta")
# ax.set_ylabel("Db")
# ax.legend()
# dirname= "./TNT"
# fname ="xi_2_graph.eps"
# plt.savefig(os.path.join(dirname,fname))

#plot sphere respect to theta
# for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
#      qc = circuit(N)
#      qc.TNT(theta,omega,"XZ")
#      husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./TNT",fname =str(theta)+"husimi_sphere.eps",alpha=0.5,view=(-90,0))




#GMS
# plt.figure()
# GMS_xi_2_S = []
# GMS_xi_2_R = []
# phi = np.pi/4 
# for theta in angles:
#       qc = circuit(N)
#       qc.GMS(theta,phi,"XY")
#       GMS_xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
#       GMS_xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

# ax = plt.gca() 
# ax.plot(angles, xi_2_S,label=r'$10log_{10}(\xi^{2}_{S})$')
# ax.plot(angles, xi_2_R, '--',label=r'$10log_{10}(\xi^{2}_{R})$')
# ax.set_xlabel("theta")
# ax.set_ylabel("Db")
# ax.legend()
# dirname= "./GMS"
# fname ="xi_2_graph.eps"
# plt.savefig(os.path.join(dirname,fname))

# for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
#      qc = circuit(N)
#      qc.GMS(theta,phi,"XY")
#      husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./GMS",fname =str(theta)+"husimi_sphere.eps",alpha=0.5,view=(-90,0))

#TAT
# plt.figure()
# TAT_xi_2_S = []
# TAT_xi_2_R = []
# for theta in angles:
#        qc = circuit(N)
#        qc.TAT(theta,"XZ")
#        TAT_xi_2_S.append(10*np.log10(np.real(get_xi_2_S(qc))))
#        TAT_xi_2_R.append(10*np.log10(np.real(get_xi_2_R(qc))))

# ax = plt.gca() 
# ax.plot(angles, TAT_xi_2_S,label=r'$10log_{10}(\xi^{2}_{S})$')
# ax.plot(angles, TAT_xi_2_R, '--',label=r'$10log_{10}(\xi^{2}_{R})$')
# ax.set_xlabel("theta")
# ax.set_ylabel("Db")
# ax.legend()
# dirname= "./TAT"
# fname ="xi_2_graph.eps"
# plt.savefig(os.path.join(dirname,fname))

# for theta in ([0.0, 0.02, 0.04, 0.06, 0.08,0.1]):
#      qc = circuit(N)
#      qc.TAT(theta,"ZY")
#      husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),dirname="./TAT",fname =str(theta)+"husimi_sphere.eps",alpha=0.5,view=(-90,0))

# ax = plt.gca() 
# ax.plot(angles, OAT_xi_2_S,marker='.',color='r', linewidth=0.2,label=r'$10log_{10}(\xi^{2}_{S})-OAT$')
# ax.plot(angles, OAT_xi_2_R,marker='.',color='g',linewidth=0.2,label=r'$10log_{10}(\xi^{2}_{R})-OAT$')

# ax.plot(angles, TNT_xi_2_S,marker='+',color='c',linewidth=0.2,label=r'$10log_{10}(\xi^{2}_{S})-TNT$')
# ax.plot(angles, TNT_xi_2_R,marker='+',color='m',linewidth=0.2,label=r'$10log_{10}(\xi^{2}_{R})-TNT$')

# ax.plot(angles, TAT_xi_2_S,marker='*',color='b',linewidth=0.2,label=r'$10log_{10}(\xi^{2}_{S})-TAT$')
# ax.plot(angles, TAT_xi_2_R,marker='*',color='y',linewidth=0.2,label=r'$10log_{10}(\xi^{2}_{R})-TAT$')

# ax.plot(angles, GMS_xi_2_S,marker='o',color='k',linewidth=0.2,label=r'$10log_{10}(\xi^{2}_{S})-GMS$')
# ax.plot(angles, GMS_xi_2_R,marker='o',color='b',linewidth=0.2,label=r'$10log_{10}(\xi^{2}_{R})-GMS$')

# ax.set_xlabel("theta")
# ax.set_ylabel("Db")
# ax.legend()
# dirname= ""
# fname ="xi_2_graph.eps"
# plt.savefig(os.path.join(dirname,fname))