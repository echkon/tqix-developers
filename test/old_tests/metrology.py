from tqix import *
from numpy import pi
import matplotlib.pyplot as plt

n = 20
j = n/2
theta = 0.0*pi
phi = 0.0

#spin cat state
def cat(j,theta,phi):
    sc = spin_coherent(j,theta,phi)
    scm = spin_coherent(j,pi-theta,phi) 
    s = normx(sc + scm)
    return s

# spin observable
j3 = soper(j)

def u(x):
    return np.exp(-1j*x*j3[2])
#print(u(0.5*pi))

t = np.linspace(0, 0.2, 100)
theta = [0.0, 0.15*pi, 0.25*pi, 0.35*pi]
f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111,aspect = 0.003)
ax2 = f2.add_subplot(111,aspect = 0.115)
for i in theta:
    r, r2, dt = [], [], []
    for k in t:
       #eigenvalue
       model = qmeas(dotx(u(k*pi),cat(j,i,phi)), [j3[1],j3[1]**2]) 
       pr0 = model.probability()[0]
       pr1 = model.probability()[1]
       r.append(pr0)
       r2.append(pr1)
       dt.append(np.sqrt(np.abs(pr1-pr0**2)))

    #differental dr and delta_phi
    dr = ndiff(t,r)
    dp = dt/np.abs(dr)
    
    #plot
    ax1.plot(t,r,'-')
    ax2.plot(t,dp,'--')

#standard quantum limit and Heisenberg limit
sql = 1./np.sqrt(float(n))*np.ones(t.shape)
hl = 1./float(n)*np.ones(t.shape)

ax2.plot(t,sql)
ax2.plot(t,hl)

plt.ylim(0,1)
plt.savefig('eigen_metro.eps')
plt.show()


