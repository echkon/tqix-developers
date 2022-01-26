from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,cos,sqrt,exp,pi
from tqix import deposex, basesx, isnormx

fig = plt.figure(figsize=(6.5,6))
ax = fig.gca(projection='3d')

#make data
N = 50
u = np.linspace(0,2*np.pi,N)
v = np.linspace(0,np.pi,N)
x = 1*np.outer(np.cos(u),np.sin(v))
y = 1*np.outer(np.sin(u),np.sin(v))
z = 1*np.outer(np.ones(np.size(u)),np.cos(v))

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

#def vectorizex()
u = basesx(2,0)
d = basesx(2,1)
p1 = u
p2 = 1/sqrt(3.)*u + sqrt(2/3.)*d
p3 = 1/sqrt(3.)*u + sqrt(2/3.)*exp(1j*2*pi/3.)*d
p4 = 1/sqrt(3.)*u + sqrt(2/3.)*exp(1j*4*pi/3.)*d

def vis_vec(state,ax,mutation,l_w,style,lc):
    if not (isnormx(state)):
        msg = 'state is not normalized'
        raise TypeError(msg)
    theta,phi = deposex(state)
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)
    #print(x,y,z)
    arrow = Arrow3D([0,x],[0,y],[0,z],mutation_scale=mutation,lw=l_w,arrowstyle=style,color=lc)
    ax.add_artist(arrow)

def vis_tri(p1,p2,p3,ax,lc):
    theta,phi = deposex(p1)
    x1 = sin(theta) * cos(phi)
    y1 = sin(theta) * sin(phi)
    z1 = cos(theta)
    theta,phi = deposex(p2)
    x2 = sin(theta) * cos(phi)
    y2 = sin(theta) * sin(phi)
    z2 = cos(theta)
    theta,phi = deposex(p3)
    x3 = sin(theta) * cos(phi)
    y3 = sin(theta) * sin(phi)
    z3 = cos(theta)
    ax.plot_trisurf([x1,x2,x3],[y1,y2,y3],[z1,z2,z3],color=lc,alpha=0.2)

vis_vec(p1,ax,20,2.5,"-|>",'r')
vis_vec(p2,ax,20,2.5,"-|>",'g')
vis_vec(p3,ax,20,2.5,"-|>",'b')
vis_vec(p4,ax,20,2.5,"-|>",'orange')

vis_tri(p1,p2,p3,ax,'red')
vis_tri(p1,p2,p4,ax,'yellow')
vis_tri(p1,p3,p4,ax,'black')
vis_tri(p2,p3,p4,ax,'green')

#plot the surface
ax.plot_surface(x,y,z,rstride=1,cstride=1,linewidth=0,\
        color='gray',alpha=0.2)
ax.plot_wireframe(x,y,z,rstride=5,cstride=5,color='blue',lw=0.2,\
        alpha=0.15)

"""
coor = np.linspace(-1,1,2)
ax.plot(coor,0*coor,zs=0,zdir='z',label='X',lw=1.5,color='green')
ax.plot(0*coor,coor,zs=0,zdir='z',label='Y',lw=1.5,color='green')
ax.plot(cos(u),sin(u),zs=0,zdir='z',linewidth=1.5,color='green',\
        alpha=0.5)
ax.plot(cos(u),sin(u),zs=0,zdir='x',linewidth=1.5,color='green',\
        alpha=0.5)
ax.plot(cos(u),sin(u),zs=0,zdir='y',linewidth=1.5,color='green',\
        alpha=0.5)
"""
ax.view_init(30,10)
plt.axis('off')
#plt.show()
plt.savefig('MIC.pdf')
