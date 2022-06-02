from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt
import time
from tqix.pis.noise import add_noise 
# import warnings
# import traceback
# import sys

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback
N=100
angles = np.linspace(0,0.1,35).tolist()

def find_mean_xi_r(noise,gate=""):
      min_xi_r = np.inf 
      min_theta = None
      for theta in angles:
            qc = circuit(N)
            qc.RN(np.pi/2,0)
            start = time.time()
            if gate=="OAT":
                  qc.OAT(theta,"Z",noise=noise,num_processes=25)
            elif gate == "TNT":
                  qc.TNT(theta,"ZX",noise=noise,num_processes=25)
            elif gate == "TAT":
                  qc.TAT(theta,"ZY",noise=noise,num_processes=25)
            xi_2_r = np.real(get_xi_2_R(qc))
            end = time.time() - start
            print(end)
            print(qc.state.diagonal().sum())
            if xi_2_r < min_xi_r:
                  min_xi_r = xi_2_r
                  min_theta = theta
            elif xi_2_r > min_xi_r:
                  break 

      print("noise,xi_2_r,theta:",noise,min_xi_r,min_theta)

find_mean_xi_r(None,gate="OAT")
find_mean_xi_r(0.05,gate="OAT")
find_mean_xi_r(0.1,gate="OAT")
find_mean_xi_r(0.15,gate="OAT")
find_mean_xi_r(0.2,gate="OAT")

find_mean_xi_r(None,gate="TNT")
find_mean_xi_r(0.05,gate="TNT")
find_mean_xi_r(0.1,gate="TNT")
find_mean_xi_r(0.15,gate="TNT")
find_mean_xi_r(0.2,gate="TNT")

find_mean_xi_r(None,gate="TAT")
find_mean_xi_r(0.05,gate="TAT")
find_mean_xi_r(0.1,gate="TAT")
find_mean_xi_r(0.15,gate="TAT")
find_mean_xi_r(0.2,gate="TAT")

