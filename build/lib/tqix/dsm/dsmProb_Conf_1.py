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

from numpy import pi,sqrt

import numpy as np
import matplotlib.pyplot as plt

from tqix.qx import *
from tqix.utility import randunit,krondel,randnormal
from tqix.qoper import eyex
from tqix.qtool import dotx
from tqix.backend import cdf
from tqix.qstate import add_random_noise,add_white_noise
from tqix.dsm.util import gtrace,gfide

__all__ = ['execu']

def execu(state,ncdf,nsamp,theta,m,st,pHist='False'):
   # to run Probe-controll Configuration 1 DSM
   dim = state.shape[0]
  
   ntr = np.zeros(nsamp)
   nfi = np.zeros(nsamp)
   
   if typex(state)=='oper':
      for i in range(nsamp):
         #p = randunit()
         p = randnormal(0.1, 0.01)
         while (p < 0) or (p > 1):
            p = randnormal(0.1, 0.01)
         state_new = add_white_noise(state,p)
         post_state = get_post_state(dim,m,st)
         prob = get_prob(state_new,post_state,theta,dim) 
         rho = get_rho(state_new,prob,ncdf,dim)
         restate = get_qurec(post_state,theta,rho,dim)
         
         ntr[i] = gtrace(restate,state)
         nfi[i] = gfide(restate,state)
   else:
      p0 = abs(sum(state))**2/dim
      for i in range(nsamp):
         state_new = add_random_noise(state,m,st)
         post_state = get_post_state(dim,m,st)
         restate = get_pqurec(state_new,post_state,p0,theta,ncdf,dim)

         ntr[i] = gtrace(restate,state)
         nfi[i] = gfide(restate,state)

   # if plot hist
   if pHist:
      plt.hist(nfi,bins=100)
      #plt.show()

   # take average
   atr = sum(ntr)/float(nsamp)
   afi = sum(nfi)/float(nsamp)
   etr,efi = 0.0,0.0

   for i in range(nsamp):
      etr += (ntr[i]-atr)**2
      efi += (nfi[i]-afi)**2

   etr /= float(nsamp)
   efi /= float(nsamp)

   return atr,np.sqrt(etr),afi,np.sqrt(efi)

"""
-------------------
some common modules 
-------------------
"""

# get post_state
def get_post_state(dim,m = 0.0,st = 0.01):
   """
   to generate a post-selected state
   >>> |c0> = 1/√N ∑(1+detal_m)|m>,
   which was used for both pure and mixed states.

   Parameters:
   -----------
   dim : dimension
   detal_m : normal Distribusion 
   st: standard deviation (default 0.01)
   N: normalization factor

   -----------
   Return quantum object |c0>
   qtype = 'bra'
   isnorm = 'true'
   """
   post_state = 1 + randnormal(m,st,dim)
   post_state /= np.linalg.norm(post_state)
   return qx(post_state)

# get propability
def get_prob(state,post_state,theta,dim):
   """
   to get probabilities P0,P1,....PL,PR
   >>> P_k = trace(rho_out * POVM_k) 

   Parameters
   ----------
   state (oper or ket): input state
   post_state (bra): post-selected state
   theta (real): angle
   dim(int): dimesion  
   """
   if typex(state)=='oper':
       e_theta = 1 - np.cos(pi * theta)
       temp_prob = np.zeros((3,dim,dim),dtype=complex)
       rho = np.zeros((4,dim,dim),dtype=complex)
       prob = np.zeros((6,dim,dim),dtype=complex)
       for x in range(dim):
          for p in range(dim):
             # temp_prob_0: sum_{x,y}
             for tx in range(dim):
                for y in range(dim):
                   temp_prob[0,x,p] += state[tx,y]*\
                                       np.exp(1j*2.0*pi*(y-tx)*p/dim)*\
                                       post_state[0,y]*post_state[0,tx]
             # temp_prob_1: sum_y
             for y in range(dim):
                temp_prob[1,x,p] += state[x,y]*\
                                    np.exp(1j*2.0*pi*(y-x)*p/dim)*\
                                    post_state[0,x]*post_state[0,y]
             # temp_prob_2: sum_{x,x}
             temp_prob[2,x,p] = state[x,x] * post_state[0,x]**2
       # rho_00: pointer_state_00
       rho[0,:,:] = (temp_prob[0,:,:] - \
                     e_theta*2*np.real(temp_prob[1,:,:]) + \
                     e_theta**2 * temp_prob[2,:,:]) / 2.0
       # rho_10: pointer_state_10
       rho[1,:,:] = (temp_prob[1,:,:] - \
                     e_theta * temp_prob[2,:,:]) * \
                     np.sin(pi * theta) / 2.0 
       # rho_01
       rho[2,:,:] = np.conj(rho[1,:,:])
       # rho_11
       rho[3,:,:] = temp_prob[2,:,:] * np.sin(pi * theta)**2 /2.0
       #probabilities
       prob[0,:,:] = np.real((rho[0,:,:]+rho[1,:,:]\
                   +rho[2,:,:]+rho[3,:,:])/2.0) #Prob.+
       prob[1,:,:] = (rho[0,:,:]-rho[1,:,:]\
                   -rho[2,:,:]+rho[3,:,:])/2.0 #Prob.-
       prob[2,:,:] = (rho[0,:,:]-1j*rho[1,:,:]\
                   +1j*rho[2,:,:]+rho[3,:,:])/2.0 #Prob.L
       prob[3,:,:] = (rho[0,:,:]+1j*rho[1,:,:]\
                   -1j*rho[2,:,:]+rho[3,:,:])/2.0 #Prob.R
       prob[4,:,:] = rho[0,:,:] #Prob. 0
       prob[5,:,:] = rho[3,:,:] #Prob. 1
       return np.real(prob)
   else: #for pure state
      prob = np.zeros((6,dim),dtype=complex)
        
      rpsi = np.real(state)
      ipsi = np.imag(state)

      psi_til = abs(dotx(post_state,state)[0,0]) #scala
      psi_abs2 = abs(state)**2 #ket: calculate |\psi'_n|^2

      #calculate probabilities
      prob[0,:]= (psi_til**2 - 2.0*post_state[0,:]*psi_til*rpsi[:,0]+\
              post_state[0,:]**2*psi_abs2[:,0])/(2.0) #P(0)
      prob[1,:]= (post_state[0,:]**2*psi_abs2[:,0])/(2.0) #P(1)
      prob[2,:]= (psi_til**2)/(4.0) #P(+)
      prob[3,:]= (psi_til**2-4.0*psi_til*post_state[0,:]*rpsi[:,0]+\
              4.0*post_state[0,:]**2*psi_abs2[:,0])/(4.0) #P(-)
      prob[4,:]= (psi_til**2 - 2.0*psi_til*post_state[0,:]*rpsi[:,0]+\
              2.0*psi_til*post_state[0,:]*ipsi[:,0]+\
              2.0*post_state[0,:]**2*psi_abs2[:,0])/(4.0) #P(L)
      prob[5,:]= (psi_til**2 - 2.0*psi_til*post_state[0,:]*rpsi[:,0]-\
              2.0*psi_til*post_state[0,:]*ipsi[:,0]+\
              2.0*post_state[0,:]**2*psi_abs2[:,0])/(4.0) #P(R)
      return np.real(prob)

def get_bisection(state,prob,dim):
   # to calculate bisection

   if typex(state)=='oper':
      rs = np.zeros((6,dim,dim))
      for i in range(6):
         for j in range(dim):
            for k in range(dim):
               rs[i,j,k] = randunit()/prob[i,j,k]
   else:
      rs = np.zeros((6,dim))
      for i in range(6):
         for j in range(dim):
            rs[i,j] = randunit()/prob[i,j]
   return rs

def get_rho(state,prob,niter,dim):
   # to calculate rho10
   # just for mixed state

   N = np.int(niter/3) #we measure 3 bases {+,-}, {L,R}, {0,1}
   rho = np.zeros((2,dim,dim),dtype=complex)
   temp_rs = np.zeros((N,6,dim,dim))

   for i in range(N):
      temp_rs[i,:,:,:] = get_bisection(state,prob,dim)

   ave_rs = np.zeros((6,dim,dim))
   for i in range(6):
      for j in range(dim):
         for k in range(dim):
            ave_rs[i,j,k] = cdf(temp_rs[:,i,j,k])
   for x in range(dim):
      for p in range(dim):
          #rho10
          rho[0,x,p] = 1/2.0*(ave_rs[0,x,p]-ave_rs[1,x,p]+\
                   1j*(ave_rs[2,x,p]-ave_rs[3,x,p]))
          #rho11
          rho[1,x,p] = ave_rs[5,x,p]
   return rho

def get_qurec(post_state,theta,rho,dim):
   # to calculate the reconstructed state
   # just for mixed state
   restate = np.zeros((dim,dim),dtype=complex)
   for x in range(dim):
      for y in range(dim):
         for p in range(dim):
            restate[x,y] += np.exp(1j*2*pi*(x-y)*p/dim)*\
                            rho[0,x,p]
         restate[x,y] += dim * np.tan(theta * pi / 2.0) * \
                         krondel(x,y) * rho[1,x,y]
         restate[x,y] /= post_state[0,x] * post_state[0,y]
         restate[x,y] *= (2 / (dim * np.sin(theta * pi))) 
   #generate physical restate
   restate = normx(restate)
   return restate

def get_pqurec(state,post_state,p0,theta,niter,dim):
   """
   to get pure quantum state 
   Eqs. (16,17) in the main text
   >>>             (P+ - P- + 2P1)
       Re[psi]_n = ---------------   (16)
                   psi_til(1+delta_n) 

   >>>                PL - PR
       Im[psi]_n = ---------------   (17)
                   psi_til(1+deta_n)
   """

   #p0 = abs(sum(state))**2/dim
   niter = int(niter*p0/3.)
   temp_rs = np.zeros((niter,6,dim))
   psi_til = abs(np.sum(dotx(post_state,state)))
   prob = get_prob(state,post_state,theta,dim)

   for i in range(niter):
      temp_rs[i,:,:] = get_bisection(state,prob,dim)
      
   ave_rs = np.zeros((6,dim))  
   for i in range(6):
      for j in range(dim):
         ave_rs[i,j] = cdf(temp_rs[:,i,j])
   
   rtemp = np.zeros(dim)
   itemp = np.zeros(dim)
   rnorm = 0.0
   for i in range(dim):
      rtemp[i] = (ave_rs[2,i]-ave_rs[3,i]+2*ave_rs[1,i])/\
                 (psi_til*post_state[0,i])
      itemp[i] = (ave_rs[4,i]-ave_rs[5,i])/\
                 (psi_til*post_state[0,i])
      rnorm += rtemp[i]**2+itemp[i]**2

   rtemp /= sqrt(rnorm)
   itemp /= sqrt(rnorm)
   
   restate = np.zeros((dim,dim),dtype=complex)
   for i in range(dim):
      for j in range(dim):
         restate[i,j] = (rtemp[i]+1j*itemp[i])*\
                     (rtemp[j]-1j*itemp[j])
   return restate
