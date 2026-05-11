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
    ax = fig.add_subplot(projection='3d')
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
    ax = fig.add_subplot(projection='3d')
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

def husimi_spin_3d(state,theta,phi,N = 100,cmap = 'Blues',dirname ="",
                fname = 'fig_husimi_spin_3d.eps',alpha = 0.65,view=(120,120),use_axis=False):
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
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, shade=False,
                    facecolors=cmap(norm(h)),linewidth=0,alpha=alpha)
    
    # Create a mappable object for the color bar
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(norm(h))

    # Add the color bar
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Husimi Q Function')  # Label for the color bar

    if use_axis:
        ax.quiver(1,0,0,1,0,0,color = 'navy', alpha = .8, lw = 1,length=0.5) #x arrow
        ax.text(1.6,0,0.1,"x","x")
        ax.quiver(0,1,0,0,1,0,color = 'navy', alpha = .8, lw = 1,length=0.5)#y arrow
        ax.text(0,1.6,0.1,"y","y")
        ax.quiver(0,0,1,0,0,1,color = 'navy', alpha = .8, lw = 1,length=0.5)#z arrow
        ax.text(0.05,0.05,1.6,"z","z")
    plt.axis('off')
    _printout(fname)
    ax.view_init(view)
    elev,azim = view
    ax.view_init(elev=elev, azim=azim)
    #fig.colorbar()  
    fig.savefig(os.path.join(dirname,fname), dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.1)
    #plt.close()

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

# ============================================================
# Spin operators in |J,m> basis ordered m=-J..J
# ============================================================
def spin_operators(J: float):
    twoJ = int(round(2 * J))
    if abs(twoJ - 2 * J) > 1e-12:
        raise ValueError("J must be integer or half-integer (so 2J is an integer).")

    dim = twoJ + 1
    ms = np.arange(-J, J + 1, 1.0)

    Jp = np.zeros((dim, dim), dtype=np.complex128)
    Jm = np.zeros((dim, dim), dtype=np.complex128)

    for i, m in enumerate(ms):
        if i + 1 < dim:
            Jp[i + 1, i] = np.sqrt(J * (J + 1.0) - m * (m + 1.0))
        if i - 1 >= 0:
            Jm[i - 1, i] = np.sqrt(J * (J + 1.0) - m * (m - 1.0))

    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    Jz = np.diag(ms.astype(np.complex128))
    return Jx, Jy, Jz, ms


def evolve_state(H: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    """
    psi(t) = exp(-i H t) psi0 using eigen-decomposition (H Hermitian).
    """
    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * evals * t)
    psi_t = evecs @ (phases * (evecs.conj().T @ psi0))
    psi_t = psi_t / np.linalg.norm(psi_t)
    return psi_t


# ============================================================
# Spin coherent state |theta,phi> in Dicke basis |J,m>, m=-J..J
# ============================================================
def spin_coherent_state(J: float, theta: float, phi: float) -> np.ndarray:
    """
    |theta,phi> = sum_m sqrt(C(2J, J+m)) (cos(theta/2))^(J+m) (sin(theta/2))^(J-m)
                  * exp(-i (J-m) phi) |J,m>
    Basis ordering: m=-J..J.
    """
    twoJ = int(round(2 * J))
    dim = twoJ + 1
    ms = np.arange(-J, J + 1, 1.0)
    k = (J + ms).astype(int)

    from math import comb
    binom = np.array([comb(twoJ, int(kk)) for kk in k], dtype=np.float64)

    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    amps = np.sqrt(binom) * (c ** (J + ms)) * (s ** (J - ms)) * np.exp(-1j * (J - ms) * phi)
    amps = amps / np.linalg.norm(amps)
    return amps.astype(np.complex128).reshape((dim,))


# ============================================================
# Husimi Q distribution: Q(theta,phi)=|<theta,phi|psi>|^2
# ============================================================
def husimi_Q_spin(psi: np.ndarray, J: float, theta_vals: np.ndarray, phi_vals: np.ndarray) -> np.ndarray:
    Q = np.zeros((len(theta_vals), len(phi_vals)), dtype=np.float64)
    for i, th in enumerate(theta_vals):
        for j, ph in enumerate(phi_vals):
            css = spin_coherent_state(J, th, ph)
            amp = np.vdot(css, psi)  # <css|psi>
            Q[i, j] = np.abs(amp) ** 2
    return Q


def draw_sphere_grid(ax, R=1.02, n_long=13, n_lat=9,
                     color="k", alpha=0.9, lw=1.0):
    # latitude circles
    latitudes = np.linspace(-np.pi/2, np.pi/2, n_lat)
    phi = np.linspace(0, 2*np.pi, 400)
    for lam in latitudes:
        th = np.pi/2 - lam
        x = R*np.sin(th)*np.cos(phi)
        y = R*np.sin(th)*np.sin(phi)
        z = R*np.cos(th)*np.ones_like(phi)
        ax.plot(x, y, z, color=color, alpha=alpha, lw=lw)

    # longitude circles
    longitudes = np.linspace(0, 2*np.pi, n_long)
    th = np.linspace(0, np.pi, 400)
    for ph in longitudes:
        x = R*np.sin(th)*np.cos(ph)
        y = R*np.sin(th)*np.sin(ph)
        z = R*np.cos(th)
        ax.plot(x, y, z, color=color, alpha=alpha, lw=lw)


def plot_husimi_panels(J, H, psi0, times, theta_vals, phi_vals, SY, SZ, visible, suptitle, filename):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for ax, t in zip(axes, times):
        psi_t = evolve_state(H, psi0, t)
        Q = husimi_Q_spin(psi_t, J, theta_vals, phi_vals)

        Q_plot = np.where(visible, Q, np.nan)
        Q_plot = Q_plot / np.nanmax(Q_plot)

        draw_sphere_grid(ax, J)
        ax.pcolormesh(SY, SZ, Q_plot, shading="auto", cmap="jet", vmin=0, vmax=1)

        ax.set_aspect("equal")
        ax.set_xlim(-J, J)
        ax.set_ylim(-J, J)
        ax.set_xlabel(r"$S_y$", fontsize=18)
        ax.set_ylabel(r"$S_z$", fontsize=18)
        ax.set_title(rf"$\chi t_1={t:.4f}$" if t > 0 else r"$\chi t_1=0$", fontsize=24)
        ax.tick_params(labelsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(suptitle, fontsize=18, y=1.02)
    plt.tight_layout()
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.show()

def plot_husimi_sphere(H, X, Y, Z, 
                       cmap_idx=79,
                       colorbar = False):
        
        H = H / np.max(H)  # normalize to [0,1]

        cmap_name = cmindex(cmap_idx)     # e.g. 'jet'
        cmap = plt.get_cmap(cmap_name) 

        norm = mpl.colors.Normalize(vmin=0.0, vmax=H.max())
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1, shade=False,
        facecolors=cmap(norm(H)),
        linewidth=0, alpha=0.7,   # make sphere slightly transparent
        )

        # color bar
        if colorbar:
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(H)     # data for the scale
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.05)
            cbar.set_label("Husimi Q", fontsize=12)

        draw_sphere_grid(ax, R=1)   


        ax.set_box_aspect([1, 1, 1])
        R = 1.1
        ax.set_xlim([-R, R])
        ax.set_ylim([-R, R])
        ax.set_zlim([-R, R])

        R = 1.7  # axis length

        # X axis (to the right)
        ax.quiver(0, 0, 0, R, 0.3, 0.1,
                color="k", linewidth=2, arrow_length_ratio=0.08)
        ax.text(R+0.1, 0, 0, r"$x$", fontsize=12,
                ha="left", va="center")

        # Y axis (towards you or into the screen depending on view)
        ax.quiver(0, 0, 0, 0, R+0.5, 0,
                color="k", linewidth=2, arrow_length_ratio=0.08)
        ax.text(0, R+0.7, 0, r"$y$", fontsize=12,
                ha="center", va="bottom")

        # Z axis (vertical)
        ax.quiver(0, 0, 0, 0, 0, R,
                color="k", linewidth=2, arrow_length_ratio=0.08)
        ax.text(0, 0, R, r"$z$", fontsize=12,
                ha="center", va="bottom")

        ax.axis("off")
        # look from +x direction (sphere "seen from the right/left" in your figure)
        ax.view_init(elev=20, azim=60)

        plt.show()
