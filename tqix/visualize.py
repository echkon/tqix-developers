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

def husimi_spin_3d(state,theta,phi,N = 100,cmap = 'jet',dirname ="",
                fname = 'fig_husimi_spin_3d.eps',alpha = 0.7,colorbar=True,axis=True,view=None):
    """ to plot Husimi visualization in Bloch sphere
    
    Parameters:
    state: quantum object
        A given quantum state needed to visualize
    theta, phi: array-like or 2D meshgrid
        Either [min, max] bounds or full theta/phi meshgrid arrays.
    cmap: str or Colormap
        A colormap instance or colormap name (default: 'jet') 
    fname: string
        File name  
    colorbar: bool
        Whether to draw the Husimi colorbar (default: True)
    axis: bool
        Whether to draw the coordinate axes (default: True)

    Returns:
    A file with fname
    """

    theta_array = np.asarray(theta)
    phi_array = np.asarray(phi)
    if theta_array.ndim > 1:
        theta_array = np.unique(theta_array)
    if phi_array.ndim > 1:
        phi_array = np.unique(phi_array)

    if theta_array.size == 2 and phi_array.size == 2:
        theta_array = np.linspace(theta_array[0], theta_array[1], N)
        phi_array = np.linspace(phi_array[0], phi_array[1], N)
    elif theta_array.ndim > 1 or phi_array.ndim > 1:
        theta_array = theta_array.flatten()
        phi_array = phi_array.flatten()

    theta_grid, phi_grid = np.meshgrid(theta_array,phi_array)

    #convert to x,y,z
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    d = state.shape[0] if hasattr(state, 'shape') else len(state)
    J = float((d - 1) / 2.0)

    sphere_radius = 5
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')
    plot_husimi_surface(ax, state, J, theta_array, phi_array, x, y, z, sphere_radius, cmap=cmap)

    format_husimi_axes(ax, sphere_radius=sphere_radius, axis=axis)
    if colorbar:
        #add_husimi_colorbar(fig, ax, label=r"$Q(\theta, \phi) = |\langle \theta, \phi | CSS \rangle|^2$")
        add_husimi_colorbar(fig, ax)

    _printout(fname)
    if view is not None:
        elev, azim = view
        ax.view_init(elev=elev, azim=azim)
    fig.savefig(os.path.join(dirname,fname), dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.1)

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

def compute_spin_evolution(H, psi_start, t_list, eigvecs_proj):
    """
    Evolves a state over time and computes probability distributions in a given basis.
    Returns the states, the list of probability arrays, and a unified ymax.
    """
    states = []
    probs_list = []
    ymax_hist = 0.0

    for t in t_list:
        psi_t = evolve_state(H, psi_start, t)
        states.append(psi_t)
        
        # Projecting onto the basis eigenvectors
        psi_t_proj = eigvecs_proj.conj().T @ psi_t
        prob = np.abs(psi_t_proj)**2
        prob /= prob.sum()
        
        probs_list.append(prob)
        ymax_hist = max(ymax_hist, prob.max())
        
    return states, probs_list, ymax_hist * 1.15 # Includes headroom

def plot_husimi_surface(ax_h, psi_t, J, theta_vals, phi_vals, X, Y, Z, sphere_radius=5, cmap='jet'):
    """Calculates the Husimi Q function and plots it on the 3D sphere."""
    theta_vals = np.asarray(theta_vals)
    phi_vals = np.asarray(phi_vals)
    if theta_vals.ndim > 1:
        theta_vals = np.unique(theta_vals)
    if phi_vals.ndim > 1:
        phi_vals = np.unique(phi_vals)

    if hasattr(psi_t, 'shape') and psi_t.ndim == 2 and psi_t.shape[0] == psi_t.shape[1]:
        Q = husimi_spin(psi_t, theta_vals, phi_vals)
    else:
        Q = husimi_Q_spin(psi_t, J, theta_vals, phi_vals)
    
    # Transpose Q so its (theta, phi) indexing aligns with the X, Y, Z meshgrid arrays.
    Q = Q.T 
    Q_plot = Q / np.nanmax(Q)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(Q_plot)

    ax_h.plot_surface(
        X * sphere_radius, Y * sphere_radius, Z * sphere_radius,
        rstride=1, cstride=1, shade=False,
        facecolors=colors, linewidth=0, antialiased=True, alpha=0.7,
    )
    draw_sphere_grid(ax_h, R=sphere_radius)

def format_husimi_axes(ax_h, sphere_radius=5, axis=True):
    """Applies custom OAT/TAT camera angles, axes limits, and specific vector arrows."""
    arrow_length = sphere_radius * 1.5
    label_offset = arrow_length * 1.15

    # Set invisible bounding box
    ax_h.set_xlim(-arrow_length, arrow_length)
    ax_h.set_ylim(-arrow_length, arrow_length)
    z_shift = 2.2
    ax_h.set_zlim(-arrow_length + z_shift, arrow_length + z_shift)

    # Specific camera view
    ax_h.view_init(elev=20, azim=-60)
    ax_h.set_proj_type('ortho')
    ax_h.set_box_aspect([1, 1, 1], zoom=1.8)
    ax_h.set_axis_off()

    if axis:
        # Draw the same stylized Husimi axes as in husimi.ipynb
        ax_h.quiver(0, 0, 0, arrow_length+0.2, -2.0, 0.7, color="k", arrow_length_ratio=0.08, linewidth=1.5)
        ax_h.quiver(0, 0, 0, -0.2, -arrow_length-3.0, -0.1, color="k", arrow_length_ratio=0.08, linewidth=1.5)
        ax_h.quiver(0, 0, 0, 0, 0, arrow_length, color="k", arrow_length_ratio=0.08, linewidth=1.5)

        # Place labels at the same locations as the notebook style
        ax_h.text(label_offset-0.6, 0, 0, r"$x$", color="k", fontsize=12, ha='center', va='center')
        ax_h.text(0, -label_offset-3.0, 0, r"$y$", color="k", fontsize=12, ha='center', va='center')
        ax_h.text(0, 0, label_offset, r"$z$", color="k", fontsize=12, ha='center', va='center')
def add_husimi_colorbar(fig, ax, label="", shrink=0.85, aspect=20, pad=0.05):
    """Adds a colorbar to a Husimi visualization with a custom label."""
    mappable = mpl.cm.ScalarMappable(cmap=plt.cm.jet, norm=mpl.colors.Normalize(vmin=0, vmax=1))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=shrink, aspect=aspect, pad=pad)
    if label:
        cbar.set_label(label, fontsize=14, rotation=90, labelpad=25)
    return cbar

def format_histogram_axes(ax_hist, eigvals, ymax, xlabel=r"$m_x$", ylabel=r"$P(m_x)$", title=None):
    """Formats the 1D probability histogram with standard aesthetics."""
    ax_hist.set_axisbelow(True)
    ax_hist.grid(axis='y', color='gray', alpha=0.3, linestyle='--')
    
    ax_hist.set_xlim(eigvals[0] - 10.0, eigvals[-1] + 10.0)
    ax_hist.set_ylim(0, ymax)
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel(ylabel)
    
    if title:
        ax_hist.set_title(title, fontsize=13)
        
    ax_hist.set_xticks([eigvals[0], 0, eigvals[-1]])
    ax_hist.set_xticklabels([r"$-m$", r"$0$", r"$m$"])


def plot_evolution_sequence(filename, states, probs_list, labels, eigvals_x,
                            ymax_hist=0.2, sphere_radius=5, N=100):
    """
    Top row: Husimi Q distributions.
    Bottom row: P(m_x) histograms.
    """
    n_cols = len(states)
    fig = plt.figure(figsize=(3 * n_cols, 6), dpi=150)

    theta_vals = np.linspace(0, np.pi, N)
    phi_vals = np.linspace(0, 2*np.pi, N)
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)

    X = np.sin(theta_grid) * np.cos(phi_grid)
    Y = np.sin(theta_grid) * np.sin(phi_grid)
    Z = np.cos(theta_grid)

    dim = len(states[0])
    J = (dim - 1) / 2

    for i, (psi_t, prob_x, label) in enumerate(zip(states, probs_list, labels)):
        ax_h = fig.add_subplot(2, n_cols, i + 1, projection="3d")
        plot_husimi_surface(
            ax_h, psi_t, J,
            theta_vals, phi_vals,
            X, Y, Z,
            sphere_radius
        )
        format_husimi_axes(ax_h, sphere_radius)

        ax_hist = fig.add_subplot(2, n_cols, n_cols + i + 1)
        ax_hist.bar(eigvals_x, prob_x, width=0.8)
        format_histogram_axes(ax_hist, eigvals_x, ymax_hist, title=label)

    plt.tight_layout()
    plt.savefig(filename, format="eps", dpi=150)
    plt.show()
    
import matplotlib.animation as animation

def generate_evolution_animation(filename, states, probs_list, titles,
                                 eigvals_x, ymax_hist=0.2,
                                 fps=15, sphere_radius=5, N=100):
    """
    Creates and saves an MP4 animation.

    Top panel: Husimi Q distribution on the sphere.
    Bottom panel: P(m_x) histogram.

    This version infers J and builds theta/phi/X/Y/Z internally.
    """

    # --- Infer spin size from state dimension ---
    dim = states[0].shape[0]
    J = (dim - 1) / 2

    # --- Build Husimi sphere grid internally ---
    theta_vals = np.linspace(0, np.pi, N)
    phi_vals = np.linspace(0, 2*np.pi, N)

    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)

    X = np.sin(theta_grid) * np.cos(phi_grid)
    Y = np.sin(theta_grid) * np.sin(phi_grid)
    Z = np.cos(theta_grid)

    fig = plt.figure(figsize=(4, 7), dpi=150)
    ax_h = fig.add_subplot(2, 1, 1, projection="3d")
    ax_hist = fig.add_subplot(2, 1, 2)

    n_frames = len(states)

    def draw_frame(i):
        ax_h.cla()
        ax_hist.cla()

        # --- Husimi sphere ---
        plot_husimi_surface(
            ax_h,
            states[i],
            J,
            theta_vals,
            phi_vals,
            X,
            Y,
            Z,
            sphere_radius
        )

        format_husimi_axes(ax_h, sphere_radius)
        ax_h.set_title(titles[i], fontsize=13)

        # --- Histogram ---
        ax_hist.bar(eigvals_x, probs_list[i], width=0.8)
        format_histogram_axes(ax_hist, eigvals_x, ymax_hist)

        plt.tight_layout()

    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=n_frames,
        interval=1000 / fps
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        bitrate=2000,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
    )

    ani.save(filename, writer=writer)
    plt.close(fig)

    print(f"Saved {filename}")
