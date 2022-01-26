import numpy as np
from tqix import spin_coherentx,ghz,husimi_spin,coherentx,spinx
from tqix import dotx,haarx,add_random_noise

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
except:
    pass

N = 30
theta = np.linspace(0, 1.0*np.pi, N)
phi = np.linspace(0, 2 * np.pi, N)
THETA,PHI = np.meshgrid(theta,phi)

#state = spinx(17/2,17/2)
state = ghz(4)
d = np.shape(state)[0]
prime = add_random_noise(state,0,0.2)

X = np.sin(THETA) * np.cos(PHI)
Y = np.sin(THETA) * np.sin(PHI)
Z = np.cos(THETA)

H = husimi_spin(state,theta,phi)
K = husimi_spin(prime,theta,phi)

if H.min() < -1e12:
        cmap = cm.summer
        norm = mpl.colors.Normalize(-H.max(), H.max())
else:
        cmap = cm.summer
        norm = mpl.colors.Normalize(H.min(), H.max())

if K.min() < -1e12:
        cmap = cm.summer
        norm = mpl.colors.Normalize(-K.max(), K.max())
else:
        cmap = cm.summer
        norm = mpl.colors.Normalize(K.min(), K.max())

fig = plt.figure(figsize=(15,6.5))
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, shade=False,
                    facecolors=cmap(norm(H)), linewidth=0,alpha=1)
plt.axis('off')
ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, shade=False,
                    facecolors=cmap(norm(K)), linewidth=0,alpha=1)

plt.axis('off')
#plt.show()
fig.savefig('spin.pdf')
