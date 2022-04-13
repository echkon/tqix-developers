from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt
import time 
N=100
angles = np.linspace(0,0.1,50).tolist()
# OAT
plt.figure()
OAT_xi_2_S = []
OAT_xi_2_R = []
min_xi_s = np.inf 
min_theta = None
for theta in angles:
      qc = circuit(N)
      qc.RN(np.pi/2,0)
      start = time.time()
      qc.OAT(theta,"Z")
      end = time.time()-start 
      xi_2_s = np.real(get_xi_2_S(qc))
      if xi_2_s < min_xi_s:
            min_xi_s = xi_2_s
            min_theta = theta
      elif xi_2_s > min_xi_s:
            break 
print(min_xi_s,min_theta)

# ax = plt.gca() 
# ax.plot(angles, OAT_xi_2_S,label=r'$\xi^{2}_{S}$')
# # ax.plot(angles, xi_2_R, '--',label=r'$10log_{10}(\xi^{2}_{R})$')
# ax.set_xlabel("theta")
# ax.set_ylabel(r"$\xi$")
# plt.xticks(angles)
# plt.show()