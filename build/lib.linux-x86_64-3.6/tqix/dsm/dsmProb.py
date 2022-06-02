"""
>>> this file is a part of tqix: a Toolbox for Quantum in X
                              x: quantum measurement, quantum metrology, 
                                 quantum tomography, and more.
________________________________
>>> copyright (c) 2019 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""

from numpy import sin, sqrt, tan, pi

import numpy as np
import matplotlib.pyplot as plt

from tqix.qx import *
from tqix.utility import randunit
from tqix.backend import cdf
from tqix.dsm.util import gtrace,gfide

__all__ = ['execu']

def execu(state,niter,nsamp,theta,pHist='False'):
   # to run Probe-controll DSM
    
   dim  = state.shape[0]
   prob = get_prob(state,theta,dim)

   ntr = np.zeros(nsamp)
   nfi = np.zeros(nsamp)
   
   if typex(state)=='oper':
      for i in range(nsamp):
         rho10 = get_rho10(state,prob,niter,dim)
         restate = get_qurec(state,theta,rho10,dim)
         ntr[i] = gtrace(restate,state)
         nfi[i] = gfide(restate,state)
   else:
      for i in range(nsamp):
         restate = get_pqurec(state,theta,niter,prob,dim)
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
# get propability
def get_prob(state,theta,dim):
   # to get probability

   if typex(state)=='oper':
      temp_prob = np.zeros((4,dim,dim),dtype=complex)
      prob = np.zeros((4,dim,dim),dtype=complex)
      for x in range(dim):
         for p in range(dim):
            # temp_prob_00
            for tx in range(dim):
               for y in range(dim):
                  temp_prob[0,x,p] += \
                  state[tx,y]*np.exp(1j*2.0*pi*(y-tx)*p/dim)
            temp_prob[0,x,p] /= (2.0*dim)

            # temp_prob_10
            for y in range(dim):
               temp_prob[1,x,p] += state[x,y]*np.exp(1j*2.0*pi*(y-x)*p/dim)

            temp_prob[1,x,p] /= (2.0*dim)
            
            # temp_prob_01
            temp_prob[2,x,p] = np.conj(temp_prob[1,x,p])

            # temp_prob_11
            temp_prob[3,x,p] = state[x,x]/(2.0*dim)

      prob[0,:,:] = np.real((temp_prob[0,:,:]+temp_prob[1,:,:]\
                 +temp_prob[2,:,:]+temp_prob[3,:,:])/2.0) #Prob.+
      prob[1,:,:] = (temp_prob[0,:,:]-temp_prob[1,:,:]\
                  -temp_prob[2,:,:]+temp_prob[3,:,:])/2.0 #Prob.-
      prob[2,:,:] = (temp_prob[0,:,:]-1j*temp_prob[1,:,:]\
                  +1j*temp_prob[2,:,:]+temp_prob[3,:,:])/2.0 #Prob.L
      prob[3,:,:] = (temp_prob[0,:,:]+1j*temp_prob[1,:,:]\
                  -1j*temp_prob[2,:,:]+temp_prob[3,:,:])/2.0 #Prob.R
      return np.real(prob)
   else: #for pure state
      rpsi = np.zeros((dim))
      ipsi = np.zeros((dim))
      prob = np.zeros((4,dim),dtype=complex)
        
      rpsi[:] = np.real(state[:,0])
      ipsi[:] = np.imag(state[:,0])
      psit = abs(state[:,0])

      prob[0,:]= (psit**2 + 2.0*psit*rpsi[:]+\
                 (rpsi[:]**2+ipsi[:]**2))/(4.0*dim) #P(+)
      prob[1,:]= (psit**2 - 2.0*psit*rpsi[:]+\
                 (rpsi[:]**2+ipsi[:]**2))/(4.0*dim) #P(-)
      prob[2,:]= (psit**2 + 2.0*psit*ipsi[:]+\
                 (rpsi[:]**2+ipsi[:]**2))/(4.0*dim) #P(L)
      prob[3,:]= (psit**2 - 2.0*psit*ipsi[:]+\
                 (rpsi[:]**2+ipsi[:]**2))/(4.0*dim) #P(R)
      return np.real(prob)

def get_bisection(state,prob,dim):
   # to calculate bisection

   if typex(state)=='oper':
      rs = np.zeros((4,dim,dim))
      for i in range(4):
         for j in range(dim):
            for k in range(dim):
               rs[i,j,k] = randunit()/prob[i,j,k]
   else:
      rs = np.zeros((4,dim))
      for i in range(4):
         for j in range(dim):
            rs[i,j] = randunit()/prob[i,j]
   return rs

def get_rho10(state,prob,niter,dim):
   # to calculate rho10
   # just for mixed state

   N = np.int(niter/2) #we just meas. 2 bases {+,-} and {L,R}
   rho10 = np.zeros((dim,dim),dtype=complex)
   temp_rs = np.zeros((N,4,dim,dim))

   for i in range(N):
      temp_rs[i,:,:,:] = get_bisection(state,prob,dim)

   ave_rs = np.zeros((4,dim,dim))
   for i in range(4):
      for j in range(dim):
         for k in range(dim):
            ave_rs[i,j,k] = cdf(temp_rs[:,i,j,k])
   for x in range(dim):
      for p in range(dim):
         rho10[x,p] = 1/2.0*(ave_rs[0,x,p]-ave_rs[1,x,p]+\
                         1j*(ave_rs[2,x,p]-ave_rs[3,x,p]))
   return rho10

def get_qurec(state,theta,rho10,dim):
   # to calculate the reconstructed state
   # just for mixed state
    
   restate = np.zeros((dim,dim),dtype=complex)
   for x in range(dim):
      for y in range(dim):
         for p in range(dim):
            restate[x,y] += np.exp(1j*2*pi*(x-y)*p/dim)*\
                            rho10[x,p]
   #restate /= np.linalg.norm(restate)
   restate *= 2.0*dim
   restate = normx(restate)
   return restate

def get_pqurec(state,theta,niter,prob,dim):
   # to get pure quantum state
  
   sin_tt  = sin(pi*theta)
   rpsi = np.real(state)
   ipsi = np.imag(state)
   psit = sqrt(sum(rpsi)**2+sum(ipsi)**2)
   p0 = (sum(rpsi)**2+sum(ipsi)**2)/dim

   N = np.int(niter*p0/2)
   temp_rs = np.zeros((N,4,dim))

   for i in range(N):
      temp_rs[i,:,:] = get_bisection(state,prob,dim)

   ave_rs = np.zeros((4,dim))  
   for i in range(4):
      for j in range(dim):
         ave_rs[i,j] = cdf(temp_rs[:,i,j])

   rtemp = np.zeros(dim)
   itemp = np.zeros(dim)
   rnorm = 0.0
   for i in range(dim):
      rtemp[i] = (ave_rs[0,i]-ave_rs[1,i])*dim/psit
      itemp[i] = (ave_rs[2,i]-ave_rs[3,i])*dim/psit
      rnorm += rtemp[i]**2+itemp[i]**2
   rtemp /= sqrt(rnorm)
   itemp /= sqrt(rnorm)

   restate = np.zeros((dim,dim),dtype=complex)
   for i in range(dim):
      for j in range(dim):
         restate[i,j] = (rtemp[i]+1j*itemp[i])*\
                     (rtemp[j]-1j*itemp[j])
   return restate
