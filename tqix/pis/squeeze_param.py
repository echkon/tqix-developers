from tqix.pis.util import *
from tqix.pis import *
import numpy as np 

__all__ = ['get_xi_2_H','get_xi_2_S','get_xi_2_R','get_xi_2_D','get_xi_2_E','get_xi_2_F']

def get_xi_2_H(alpha,beta,gamma,qc):
    var_alpha = qc.var(alpha)
    mean_beta = qc.expval(beta)
    mean_gamma = qc.expval(gamma)
    return (2*var_alpha)/np.sqrt(mean_beta**2+mean_gamma**2) 

def get_xi_2_S(qc):
    N = qc.N
    mean_x = qc.expval(type="x")
    mean_y = qc.expval(type="y")
    mean_z = qc.expval(type="z")
    mag_mean_J = np.sqrt(np.real(mean_x**2+mean_y**2+mean_z**2))
    theta = np.arccos(mean_z/mag_mean_J)
    if mean_y > 0:
        phi = np.arccos(mean_x/(mag_mean_J*np.sin(theta)))
    else:
        phi = 2*np.pi - np.arccos(mean_x/(mag_mean_J*np.sin(theta)))
    n1 = np.asarray([-np.sin(phi),np.cos(phi),0]).astype(np.complex128)
    n2 = np.asarray([np.cos(theta)*np.cos(phi),0,-np.sin(theta)]).astype(np.complex128)
    cov = qc.expval(type="cov",use_vector=True,n1=n1,n2=n2)/2
    # *(1e-16)
    if -1e-4 < cov < 1e-4:
        cov = 0
    mean_n1n2_minus = qc.expval(type="n1n2_minus",use_vector=True,n1=n1,n2=n2)
    mean_n1n2_plus = qc.expval(type="n1n2_plus",use_vector=True,n1=n1,n2=n2)
    xi_2_S_1 =  2/N*(mean_n1n2_plus+np.sqrt(mean_n1n2_minus**2+4*cov**2))
    xi_2_S_2 =  2/N*(mean_n1n2_plus-np.sqrt(mean_n1n2_minus**2+4*cov**2))
    if xi_2_S_1 < 0 and xi_2_S_2 > 0:
        return xi_2_S_2
    elif xi_2_S_1 > 0 and xi_2_S_2 < 0:
        return xi_2_S_1
    else:
        return min(xi_2_S_1,xi_2_S_2)

def get_xi_2_R(qc):
    xi_2_S = get_xi_2_S(qc)
    N = qc.N
    mean_Jz = qc.expval(type="z")
    return (N**2/(4*np.abs(mean_Jz)**2))*xi_2_S

def get_xi_2_D(qc,n):
    N = qc.N
    var = qc.var(type="xyz",use_vector=True,n=n)
    mean = qc.expval(type="xyz",use_vector=True,n=n)   
    return (N*var)/(N**2/4-mean**2)

def get_xi_2_E(qc,n):
    N = qc.N
    var_Jvec = qc.var(type="xyz",use_vector=True,n=n)
    mean_Jvec = qc.expval(type="xyz",use_vector=True,n=n)
    mean_J_2 = qc.expval(type="J2")
    return (N*var_Jvec)/(mean_J_2-N/2-mean_Jvec**2)

def get_xi_2_F(qc,n1,n2,n3):
    N=qc.N
    var_J_n1 = qc.var(type="xyz",use_vector=True,n=n1)
    mean_J_n2 = qc.expval(type="xyz",use_vector=True,n=n2)
    mean_J_n3 = qc.expval(type="xyz",use_vector=True,n=n3)
    return (N*var_J_n1)/(mean_J_n2**2+mean_J_n3**2)




