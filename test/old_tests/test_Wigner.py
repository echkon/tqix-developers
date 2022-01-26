from tqix import *
import numpy as np
from numpy import dot,pi,exp,conj
from scipy.special import genlaguerre,factorial
import matplotlib.pyplot as plt

d = 10
xrange = [-5,5]
yrange = [-5,5]

state = 1./np.sqrt(2)*(coherent(d,2.5)+
                       coherent(d,-2.5))

state = spin_coherent(31/2, 0.5*np.pi, -0.3*np.pi)

theta = [0, np.pi]
phi = [0, 2*np.pi]

wigner_spin_3d(state,theta,phi,N=50,cmap=cmindex(1),fname='visual_Wigner3d.eps')



