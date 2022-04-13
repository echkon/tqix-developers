"""
>>> this file is a part of tqix: a Toolbox for Quantum in X
                              x: quantum measurement, quantum metrology, 
                                 quantum tomography, and more.
________________________________
>>> copyright (c) 2019 and later
>>> authors: Binho Le
>>> contributors: Quangtuan Kieu
>>> all rights reserved
________________________________
"""

__all__ = ['husimi_2d','husimi_3d','wigner_2d','wigner_3d',
           'husimi_spin_3d','wigner_spin_3d','cmindex']

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import amax, meshgrid
import numpy as np
from tqix.quasi_prob import *
import os 

def husimi_2d(state,xrange,yrange,N = 100,fname='fig_husimi_2d.eps',
              cmap = 'viridis',alpha = 1.0):
    """
    to visualize a Husimi Q function

    Parameters:
    ------------
    state: quantum object
        A given quantum state needed to visualize
    xrange, yrange: array-like(2)
        The minimum and maximum values of the coordinates 
    N: integer
        number of steps for xrange, yrange
    fname: string
        File name  
    cmap: str or Colormap
        A colormap instance or colormap name (default: 'viridis') 

    Returns:
    A file with fname
    """

    xarray = np.linspace(xrange[0],xrange[1],N)
    yarray = np.linspace(yrange[0],yrange[1],N)
    zarray = husimi(state,xarray,yarray)
    zarray /= amax(zarray)

    fig,axes = plt.subplots(1,1,figsize=(6,6))
    cont = axes.contourf(xarray,yarray,zarray,80,cmap = cmap,alpha = alpha)
    plt.xlabel("x")
    plt.ylabel("y")

    for c in cont.collections:
        c.set_edgecolor("face")

    _printout(fname)
    plt.savefig(fname, dpi=25)

def husimi_3d(state,xrange,yrange,N = 100,fname='fig_husimi_3d.eps',
              cmap = 'viridis',alpha = 1.0):
    """
    to visualize a 3d Husimi function

    Parameters:
    ------------
    state: quantum object
        A given quantum state needed to visualize
    xrange, yrange: array-like(2)
        The minimum and maximum values of the coordinates 
    N: integer
        number of steps for xrange, yrange
    fname: string
        File name  
    cmap: str or Colormap
        A colormap instance or colormap name (default: 'viridis') 

    Returns:
    A file with fname
    """
   
    xarray = np.linspace(xrange[0],xrange[1],N)
    yarray = np.linspace(yrange[0],yrange[1],N)
    zarray = husimi(state,xarray,yarray)
    zarray /= amax(zarray)

    xx, yy = meshgrid(xarray,yarray)

    norm = plt.Normalize(zarray.min(), zarray.max())
    colors = cm.viridis(norm(zarray))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zarray, cmap=cmap,
                    rstride=1, cstride=1,linewidth=0,facecolors=colors)

    plt.xlabel("x")
    plt.ylabel("y")

    _printout(fname)
    plt.savefig(fname, dpi=25)

def wigner_2d(state,xrange,yrange,N = 100,fname='fig_wigner_2d.eps',
              cmap = 'viridis',alpha = 1.0):

    """
    to visualize a Wigner function
        Parameters:
    ------------
    state: quantum object
        A given quantum state needed to visualize
    xrange, yrange: array-like(2)
        The minimum and maximum values of the coordinates 
    N: integer
        number of steps for xrange, yrange
    fname: string
        File name  
    cmap: str or Colormap
        A colormap instance or colormap name (default: 'viridis') 

    Returns:
    A file with fname
    """

    xarray = np.linspace(xrange[0],xrange[1],N)
    yarray = np.linspace(yrange[0],yrange[1],N)
    zarray = wigner(state,xarray,yarray)
    zarray /= amax(zarray)

    fig,axes = plt.subplots(1,1,figsize=(6,6))
    cont = axes.contourf(xarray,yarray,zarray,80,cmap = cmap,alpha = alpha)

    plt.xlabel("x")
    plt.ylabel("y")

    for c in cont.collections:
        c.set_edgecolor("face")

    _printout(fname)
    plt.savefig(fname, dpi=25)

def wigner_3d(state,xrange,yrange,N = 100,fname='fig_husimi_3d.eps',
              cmap = 'viridis',alpha = 1.0):
    """
    to visualize a 3d Wigner function

    Parameters:
    ------------
    state: quantum object
        A given quantum state needed to visualize
    xrange, yrange: array-like(2)
        The minimum and maximum values of the coordinates 
    N: integer
        number of steps for xrange, yrange
    fname: string
        File name  
    cmap: str or Colormap
        A colormap instance or colormap name (default: 'viridis') 

    Returns:
    A file with fname
    """

    xarray = np.linspace(xrange[0],xrange[1],N)
    yarray = np.linspace(yrange[0],yrange[1],N)
    zarray = wigner(state,xarray,yarray)
    zarray /= amax(zarray)

    xx, yy = meshgrid(xarray,yarray)

    norm = plt.Normalize(zarray.min(), zarray.max())
    colors = cm.viridis(norm(zarray))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zarray, cmap=cmap,
                    rstride=1, cstride=1,linewidth=0,facecolors=colors)

    plt.xlabel("x")
    plt.ylabel("y")

    _printout(fname)
    plt.savefig(fname, dpi=25)

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
except:
    pass

def husimi_spin_3d(state,theta,phi,N = 100,cmap = 'viridis',dirname ="",
                   fname = 'fig_husimi_spin_3d.eps',alpha = 1,view=(120,120),use_axis=False):
    """ to plot Husimi visualization in Bloch sphere
    
    Parameters:
    state: quantum object
        A given quantum state needed to visualize
    theta, phi: array-like(2)
        The minimum and maximum values of the coordinates 
    cmap: str or Colormap
        A colormap instance or colormap name (default: 'viridis') 
    fname: string
        File name  
 
    Returns:
    A file with fname
    """

    theta_array = np.linspace(theta[0], theta[1], N)
    phi_array = np.linspace(phi[0], phi[1], N)
    theta_grid, phi_grid = np.meshgrid(theta_array,phi_array)

    #convert to x,y,z
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    h = husimi_spin(state,theta_array,phi_array)

    a = str('cm.')+cmap
    if h.min() < -10e10:
        cmap = eval(a)
        norm = mpl.colors.Normalize(-h.max(), h.max())
    else:
        cmap = eval(a)
        norm = mpl.colors.Normalize(h.min(), h.max())

    fig = plt.figure(figsize=(6,6))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, shade=False,
                    facecolors=cmap(norm(h)),linewidth=0,alpha=alpha)
    if use_axis:
        ax.quiver(1,0,0,1,0,0,color = 'navy', alpha = .8, lw = 3) #x arrow
        ax.text(1.6,0,0.1,"Sx","x")
        ax.quiver(0,1,0,0,1,0,color = 'navy', alpha = .8, lw = 3)#y arrow
        ax.text(0,1.6,0.1,"Sy","y")
        ax.quiver(0,0,1,0,0,1,color = 'navy', alpha = .8, lw = 3)#z arrow
        ax.text(0.05,0.05,1.6,"Sz","z")
    plt.axis('off')
    _printout(fname)
    ax.view_init(view)
    elev,azim = view
    ax.view_init(elev=elev, azim=azim)
    fig.savefig(os.path.join(dirname,fname),dpi=50,bbox_inches='tight')
    plt.close()

def wigner_spin_3d(state,theta,phi,N = 100,cmap = 'viridis',
                   fname = 'fig_wigner_spin_3d.eps',alpha = 1,view=(120,120),use_axis=False):
    """ to plot Husimi visualization in Bloch sphere
    
    Parameters:
    state: quantum object
        A given quantum state needed to visualize
    theta, phi: array-like(2)
        The minimum and maximum values of the coordinates 
    cmap: str or Colormap
        A colormap instance or colormap name (default: 'viridis') 
    fname: string
        File name  
 
    Returns:
    A file with fname
    """

    theta_array = np.linspace(theta[0], theta[1], N)
    phi_array = np.linspace(phi[0], phi[1], N)
    theta_grid, phi_grid = np.meshgrid(theta_array,phi_array)

    #convert to x,y,z
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    h = wigner_spin(state,theta_array,phi_array)
    #print(h)

    a = str('cm.')+cmap
    if h.min() < -10e10:
        cmap = eval(a)
        norm = mpl.colors.Normalize(-h.max(), h.max())
    else:
        cmap = eval(a)
        norm = mpl.colors.Normalize(h.min(), h.max())

    fig = plt.figure(figsize=(6,6))
#    ax = fig.gca(projection='3d')
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, shade=False,
                    facecolors=cmap(norm(h)),linewidth=0,alpha=alpha)
    if use_axis:
        ax.quiver(1,0,0,1,0,0,color = 'navy', alpha = .8, lw = 3) #x arrow
        ax.text(1.6,0,0.1,"Sx","x")
        ax.quiver(0,1,0,0,1,0,color = 'navy', alpha = .8, lw = 3)#y arrow
        ax.text(0,1.6,0.1,"Sy","y")
        ax.quiver(0,0,1,0,0,1,color = 'navy', alpha = .8, lw = 3)#z arrow
        ax.text(0.05,0.05,1.6,"Sz","z")
    plt.axis('off')
    _printout(fname)
    ax.view_init(view)
    elev,azim = view
    ax.view_init(elev=elev, azim=azim)
    fig.savefig(f"{elev},{azim},{fname}",dpi=50,bbox_inches='tight')
    plt.show(block=True)
###
def _printout(fname):
    print('***')
    print('Figure ', fname, ' has created')

def cmindex(d):
    #https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
    cmaps = ['viridis', #1
             'plasma',  #2
             'inferno', #3
             'magma',   #4 
             'cividis', #5
             'Greys',   #6
             'Purples', #7
             'Blues',   #8
             'Greens',  #9
             'Oranges', #10
             'Reds',    #11
             'YlOrBr',  #12
             'YlOrRd',  #13
             'OrRd',    #14
             'PuRd',    #15
             'RdPu',    #16
             'BuPu',    #17
             'GnBu',    #18
             'PuBu',    #19
             'YlGnBu',  #20
             'PuBuGn',  #21
             'BuGn',    #22
             'YlGn',    #23
             'binary',  #24
             'gist_yarg', #25
             'gist_gray', #26
             'gray',    #27
             'bone',    #28
             'pink',    #29
             'spring',  #30
             'summer',  #31
             'autumn',  #32
             'winter',  #33
             'cool',    #34
             'Wistia',  #35
             'hot',     #36
             'afmhot',  #37
             'gist_heat',#38
             'copper',   #39
             'PiYG',     #40
             'PRGn',     #41
             'BrBG',     #42
             'PuOr',     #43
             'RdGy',     #44
             'RdBu',     #45
             'RdYlBu',   #46
             'RdYlGn',   #47
             'Spectral', #48
             'coolwarm', #49
             'bwr',      #50
             'seismic',  #51
             'twilight', #52
             'twilight_shifted', #53 
             'hsv',              #54
             'Pastel1',  #55
             'Pastel2',  #56
             'Paired',   #57
             'Accent',   #58
             'Dark2',    #59
             'Set1',     #60
             'Set2',     #61
             'Set3',     #62
             'tab10',    #63
             'tab20',    #64
             'tab20b',   #65
             'tab20c',   #66
             'flag',     #67
             'prism',    #78
             'ocean',    #69
             'gist_earth',#70
             'terrain',   #71
             'gist_stern',#72
             'gnuplot',   #73
             'gnuplot2',  #74
             'CMRmap',    #75
             'cubehelix', #76
             'brg',       #77
             'gist_rainbow', #78
             'rainbow',      #79
             'jet',          #80
             'nipy_spectral',#81
             'gist_ncar'     #82
             ]
    if d > len(cmaps):
       raise IndexError('Out of index range')
    else:
       return cmaps[d]
