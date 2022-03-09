
from tqix.pis import *
from tqix import *
import numpy as np

N=100
qc = circuit(N)
# husimi_3d(qc.state.toarray(), x ,y ,cmap = cmindex(1),fname ="husimi3d.eps")
THETA = [0, np.pi]
PHI = [0, 2* np.pi]
# husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),fname ="husimi_sphere.eps",alpha=0.5,view=(-90,0))
qc.OAT(0,"X")
print(qc.state.diagonal().sum())
print("xi_2_H:",get_xi_2_H("x","y","z",qc))
print("xi_2_S:",get_xi_2_S(qc))
print("xi_2_R:",get_xi_2_R(qc))
print("xi_2_D:",get_xi_2_D(qc,n=[1,0,0]))
print("xi_2_E:",get_xi_2_E(qc,n=[1,0,0]))
print("xi_2_F:",get_xi_2_F(qc,[1,0,0],[0,1,0],[0,0,1]))
# husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),fname ="afterhusimi_sphere.eps",alpha=0.5,view=(-90,0))
