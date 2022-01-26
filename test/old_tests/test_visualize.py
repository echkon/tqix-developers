import numpy as np
from numpy import kron, dot
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqix import *

def spin_mat(a):
    """
    return matrices, Sy, Sx for 3 entangled qubits
    """
    if a == 'x':
        return 0.5 * (kron(sigmax(),kron(eyex(2),eyex(2))) +\
               kron(eyex(2),kron(sigmax(),eyex(2))) +\
               kron(eyex(2),kron(eyex(2),sigmax()))) 
    elif a == 'y':
        return 0.5 * (kron(sigmay(),kron(eyex(2),eyex(2))) +\
               kron(eyex(2),kron(sigmay(),eyex(2))) +\
               kron(eyex(2),kron(eyex(2),sigmay()))) 
    elif a == 'z':
        return 0.5 * (kron(sigmaz(),kron(eyex(2),eyex(2))) +\
               kron(eyex(2),kron(sigmaz(),eyex(2))) +\
               kron(eyex(2),kron(eyex(2),sigmaz()))) 

def spin_val(a,state):
    return float(np.real(dot(daggx(state),dot(spin_mat(a),\
            state))))

def draw_vec(ax, state, color):
    x = spin_val('x',state)
    y = spin_val('y',state)
    z = spin_val('z',state)
    vec = [x, y, z]
    #print(np.linalg.norm(vec))
    ax.scatter(x, y, z, s=30, c=color, marker="o")

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

fig = plt.figure(figsize = (5,7))
ax = fig.add_subplot(111, projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 500)
v = np.linspace(0, np.pi, 10)
x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.scatter(x, y, z, s=1)

#plt.show()
N = 40
m = 3.0
st = 0.05
state = dicke(3,1)
m = spin_val('x',state)
n = spin_val('y',state)
p = spin_val('z',state)
vec = [m,n,p]

for i in range(N):
    state_new = add_random_noise(state, m, st)
    draw_vec(ax, state_new, 'orange')

ax.scatter(m,n,p, s=100, c='red', marker="o")

points = np.zeros((1000,3))
data = np.linspace(0, 2*np.pi, 1000)
points[:,0] = np.sin(data)
points[:,2] = np.cos(data)

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)

fig.savefig('spin.pdf')
