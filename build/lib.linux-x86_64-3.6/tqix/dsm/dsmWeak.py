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

from numpy import sin,pi,tan,sqrt

import numpy as np
import matplotlib.pyplot as plt

from tqix.qx import *
from tqix.utility import randunit
from tqix.backend import cdf
from tqix.dsm.util import gtrace,gfide

__all__ = ['execu']

def execu(state,niter,nsamp,theta,pHist='False'):
   # to run weak DSM
    
   dim  = state.shape[0]
   prob = get_prob(state,theta,dim)

   ntr = np.zeros(nsamp)
   nfi = np.zeros(nsamp)
   
   if typex(state)=='oper':
      for i in range(nsamp):
         rho10,rho11 = get_rho10_rho11(state,prob,niter,dim)
         restate = get_qurec(state,theta,rho10,rho11,dim)
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

   sin_tt  = sin(pi*theta)
   sin_2tt = sin(pi*theta/2.0)**2
   epsi_tt = 2.0*sin_2tt

   if typex(state)=='oper':
      temp_prob = np.zeros((4,dim,dim),dtype=complex)
      prob = np.zeros((5,dim,dim),dtype=complex)
      for x in range(dim):
         for p in range(dim):
            # temp_prob_00
            #temp_prob[0,x,p] = 0.0
            for tx in range(dim):
               for y in range(dim):
                  temp_prob[0,x,p] += \
                  state[tx,y]*np.exp(1j*2.0*pi*(y-tx)*p/dim)
            for y in range(dim):
               temp_prob[0,x,p] -= 2.0*sin_2tt*(\
               state[x,y]*np.exp(1j*2.0*pi*(y-x)*p/dim)+\
               np.conj(state[x,y])*np.exp(-1j*2.0*pi*(y-x)*p/dim))

            temp_prob[0,x,p] += 4*sin_2tt**2*state[x,x]
            temp_prob[0,x,p] /= dim

            # temp_prob_10
            for y in range(dim):
               temp_prob[1,x,p] += state[x,y]*np.exp(1j*2.0*pi*(y-x)*p/dim)
            temp_prob[1,x,p] -= 2*sin_2tt*state[x,x]
            temp_prob[1,x,p] = temp_prob[1,x,p]*sin_tt/dim
            
            # temp_prob_01
            temp_prob[2,x,p] = np.conj(temp_prob[1,x,p])

            # temp_prob_11
            temp_prob[3,x,p] = sin_tt**2*state[x,x]/dim

      prob[0,:,:] = np.real((temp_prob[0,:,:]+temp_prob[1,:,:]\
                 +temp_prob[2,:,:]+temp_prob[3,:,:])/2.0) #Prob.+
      prob[1,:,:] = (temp_prob[0,:,:]-temp_prob[1,:,:]\
                  -temp_prob[2,:,:]+temp_prob[3,:,:])/2.0 #Prob.-
      prob[2,:,:] = (temp_prob[0,:,:]-1j*temp_prob[1,:,:]\
                  +1j*temp_prob[2,:,:]+temp_prob[3,:,:])/2.0 #Prob.L
      prob[3,:,:] = (temp_prob[0,:,:]+1j*temp_prob[1,:,:]\
                  -1j*temp_prob[2,:,:]+temp_prob[3,:,:])/2.0 #Prob.R
      prob[4,:,:] = temp_prob[3,:,:] #Prob.1
      return np.real(prob)
   else: #for pure state
      rpsi = np.zeros((dim))
      ipsi = np.zeros((dim))
      prob = np.zeros((5,dim))
        
      rpsi[:] = np.real(state[:,0])
      ipsi[:] = np.imag(state[:,0])
      psit = sqrt(sum(rpsi)**2+sum(ipsi)**2)

      prob[0,:]= psit/dim*(psit/2.0+pi*theta*rpsi[:]) #P(+)
      prob[1,:]= psit/dim*(psit/2.0-pi*theta*rpsi[:]) #P(-)
      prob[2,:]= psit/dim*(psit/2.0+pi*theta*ipsi[:]) #P(L)
      prob[3,:]= psit/dim*(psit/2.0-pi*theta*ipsi[:]) #P(R)
      prob[4,:]= 0.0 #P(1)
      return np.real(prob)

def get_bisection(state,prob,dim):
   # to calculate bisection

   if typex(state)=='oper':
      rs = np.zeros((5,dim,dim))
      for i in range(5):
         for j in range(dim):
            for k in range(dim):
               rs[i,j,k] = randunit()/prob[i,j,k]
   else:
      rs = np.zeros((5,dim))
      for i in range(5):
         for j in range(dim):
            rs[i,j] = randunit()/prob[i,j]
   return rs

def get_rho10_rho11(state,prob,niter,dim):
   # to calculate rho10 and rho11
   # just for mixed state

   N = np.int(niter/2)
   rho10 = np.zeros((dim,dim),dtype=complex)
   rho11 = np.zeros((dim,dim),dtype=complex)
   temp_rs = np.zeros((N,5,dim,dim))

   for i in range(N):
      temp_rs[i,:,:,:] = get_bisection(state,prob,dim)

   ave_rs = np.zeros((5,dim,dim))
   for i in range(5):
      for j in range(dim):
         for k in range(dim):
            ave_rs[i,j,k] = cdf(temp_rs[:,i,j,k])
   for x in range(dim):
      for p in range(dim):
         rho10[x,p] = 1/2.0*(ave_rs[0,x,p]-ave_rs[1,x,p]+\
                         1j*(ave_rs[2,x,p]-ave_rs[3,x,p]))
         rho11[x,p] = ave_rs[4,x,p]
   return rho10,rho11

def get_qurec(state,theta,rho10,rho11,dim):
   # to calculate the reconstructed state
   # just for mixed state
    
   restate = np.zeros((dim,dim),dtype=complex)
   for x in range(dim):
      for y in range(dim):
         for p in range(dim):
            restate[x,y] += np.exp(1j*2*pi*(x-y)*p/dim)*\
                            rho10[x,p] #S37
   
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
   temp_rs = np.zeros((N,5,dim))

   for i in range(N):
      temp_rs[i,:,:] = get_bisection(state,prob,dim)

   ave_rs = np.zeros((5,dim))  
   for i in range(5):
      for j in range(dim):
         ave_rs[i,j] = cdf(temp_rs[:,i,j])

   rtemp = np.zeros(dim)
   itemp = np.zeros(dim)
   rnorm = 0.0
   for i in range(dim):
      rtemp[i] = (ave_rs[0,i]-ave_rs[1,i]+2.0*tan(pi*theta/2.0)*\
             ave_rs[4,i])*dim/(2*psit*sin_tt)
      itemp[i] = (ave_rs[2,i]-ave_rs[3,i])*dim/(2*psit*sin_tt)
      rnorm += rtemp[i]**2+itemp[i]**2
   rtemp /= sqrt(rnorm)
   itemp /= sqrt(rnorm)

   restate = np.zeros((dim,dim),dtype=complex)
   for i in range(dim):
      for j in range(dim):
         restate[i,j] = (rtemp[i]+1j*itemp[i])*\
                     (rtemp[j]-1j*itemp[j])
   return restate
