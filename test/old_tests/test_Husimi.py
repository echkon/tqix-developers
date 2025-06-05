from tqix import *
import numpy as np

xm = [-4,4]
ym = [-4,4]

#state = coherent(20,np.sqrt(2)) #ghz(4)

#husimi_2d(state,xm,ym,cmap = cmindex(20),fname = 'visua2d_hu.eps',alpha = 0.5)
#husimi_3d(state,xm,ym,fname= 'visua3d_hu.eps')

state = ghz(5)
prime = add_random_noise(state,0,0.2)

state = spin_coherent(37/2,0.5*np.pi,-0.3*np.pi)
prime = add_random_noise(state,0,0.1)

theta = [0, np.pi]
phi = [0, 2* np.pi]

husimi_spin_3d(prime,theta,phi,cmap = cmindex(1),fname = 'spin_coherent_01.eps')

