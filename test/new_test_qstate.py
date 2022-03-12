
from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt

from tqix.pis.squeeze_param import get_xi_2_R
N=100
# husimi_3d(qc.state.toarray(), x ,y ,cmap = cmindex(1),fname ="husimi3d.eps")
THETA = [0, np.pi]
PHI = [0, 2* np.pi]
# husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),fname ="husimi_sphere.eps",alpha=0.5,view=(-90,0))
# theta = np.linspace(0,0.5,100000)
# for t in theta:
#     xi_2_s = get_xi_2_S(qc)
#     if xi_2_s < 1 and t != 0:
#         break 
# print(t)
qc = circuit(N)
qc.OAT(np.pi/100,"X")
print("xi_2_S:",get_xi_2_S(qc))
print("xi_2_R:",get_xi_2_R(qc))
# qc.state.toarray()
# y = []
# x = np.linspace(0,0.7,300).tolist()
# for theta in x:
#     qc = circuit(N)
#     qc.OAT(theta,"X")
#     y.append(10*np.log10(np.real(get_xi_2_F(qc,[0,1,0],[0,0,1],[1,0,0]))))
# # plt.ylim([0,-0.10])
# plt.plot(x,y)
# plt.show()
# husimi_spin_3d(qc.state.toarray(), THETA ,PHI ,cmap = cmindex(1),fname ="afterhusimi_sphere.eps",alpha=0.5,view=(-90,0))
