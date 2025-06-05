from tqix import *
import numpy as np

psi = zbasis(3,2)

# 3d visualization
x = [-3, 3]
y = [-3, 3]
husimi_3d(psi, x ,y ,cmap = cmindex(1),fname='husimi3d.eps')
wigner_3d(psi, x ,y ,cmap = cmindex(1),fname='wigner3d.eps')

# Bloch sphere visualization
THETA = [0, np.pi]
PHI = [0, 2* np.pi]
husimi_spin_3d(psi, THETA ,PHI ,cmap =cmindex(1),fname = 'husimi_sphere.eps')
wigner_spin_3d(psi, THETA ,PHI ,cmap =cmindex(1),fname = 'wigner_sphere.eps')

