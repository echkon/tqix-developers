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

import numpy as np
from tqix.dsm.dsmStrong import execu as exeStr
from tqix.dsm.dsmWeak import execu as exeWeak
from tqix.dsm.dsmProb import execu as exeProb
from tqix.dsm.dsmProb_Conf_1 import execu as exeProb_1
from tqix.dsm.dsmProb_Conf_2 import execu as exeProb_2

class execute:
   # to run the program
   
   def __init__(self,state,niter,nsamp,theta,m=0.,st=0.,pHist=False):
      self.state = state
      self.niter = niter
      self.nsamp = nsamp
      self.theta = theta
      self.m = m
      self.st = st
      self.pHist = pHist

   def job(self,name):
      if name == 'strong':
         model = exeStr(self.state,\
                 self.niter,self.nsamp,\
                 self.theta,self.pHist)
      if name == 'weak':
         model = exeWeak(self.state,\
                 self.niter,self.nsamp,\
                 self.theta,self.pHist)
      if name == 'prob':
         model = exeProb(self.state,\
                 self.niter,self.nsamp,\
                 self.theta,self.pHist)
      if name == 'prob_1':
         model = exeProb_1(self.state,\
                 self.niter,self.nsamp,\
                 self.theta,self.m,self.st,self.pHist)
      if name == 'prob_2':
         model = exeProb_2(self.state,\
                 self.niter,self.nsamp,\
                 self.theta,self.m,self.st,self.pHist)
      return model

   def info(self):
      print("This is to run a DSM code")
