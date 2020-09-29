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

__all__ = ['qmeas','qevo','qsim']

import time
from numpy import sqrt,exp,pi
import numpy as np
from tqix.utility import randunit
from tqix.qoper import *
from tqix.qtool import dotx, tensorx
from tqix.qstate import *
from tqix.backend import *
from tqix.qx import *

from tqix.povm.povm import _pauli,_stoke,_mub,_sic

class qmeas:
    """
    quantum measurement
    """
    def __init__(self,state,*args):
        self.state = state
        self.args = args
    
    def probability(self):
        pr, dt = qmeas._pro_time_(self)
        return pr

    def mtime(self):
        pr, dt = qmeas._pro_time_(self)
        return dt 
    
    def _pro_time_(self):
        if self.state is None:
           msg = 'please input somthing'
           raise TypeError(msg)
        else:
           # turn state into oper
           if typex(self.state) != 'oper':
              instate = operx(self.state)
           else:
              instate = self.state
          
           dims = shapex(instate)[0]
           # check args
           if self.args == None:
              return tracex(dotx(instate,self.args))
           elif self.args[0] == 'Pauli':
              pov = _pauli(int(np.log2(dims)))
           elif self.args[0] == 'Stoke':
              pov = _stoke(int(np.log2(dims)))
           elif self.args[0] == 'MUB':
              pov = _mub(dims)
           elif self.args[0] == 'SIC':
              pov = _sic(dims)
           else: #arbitary POVMs
              if len(self.args) == 1 and \
                 isinstance(self.args[0],(list,np.ndarray)):
                 # e.g. args = [m1,m2,...mn]
                 pov = self.args[0]
              else:
                 pov = self.args

           # now calculate probability
           pr = []
           start = time.time() # starting time
           for i in range (len(pov)):
              pr.append(tracex(dotx(instate,pov[i]))) 
           end = time.time() # stoping time
           dtime = (end - start) # time need to calcualte all proba.   
           return pr, dtime
    
    def poststate(self):
        return self.state #not yet

class qevo:
    # to get the evolution of quantum system 

    def __init__(self,state=None,evol=None,post=None,obser=None):
        self.state = state
        self.evol = evol
        self.post = post
        self.obser = obser

    def output(self):
        # to get output state
        if self.state is None: 
           msg = 'please input somthing'
           raise TypeError(msg)
        else:
           dim = self.state.shape[0]
           if self.evol is None:
              self.evol = eyex(dim)
           if typex(self.state) == 'bra':
              msg = 'state should not a bra vector'
              raise TypeError(msg)
           elif typex(self.state) == 'ket':
              if self.post is None:
                 return(dotx(self.evol,self.state))
              else: # with post-selected
                 if typex(self.post) != 'bra':
                    msg = 'post state must be a bra vector'
                    raise TypeError(msg)
                 else:
                    return(dotx(self.post,self.evol,self.state))
           else: # state is an oper
              if self.post is None:
                 return(dotx(self.evol,self.state,daggx(self.evol)))
              else: # with post-selected
                 if typex(self.post) != 'bra':
                    msg = 'post state must be a bra vector'
                    raise TypeError(msg)
                 else:
                    pp = dotx(self.evol,self.state,dagg(self.evol))
                    return(dotx(self.post,pp,dagg(self.post)))

    def info(self):
       print('to get measurement results: outcome state,expectation')

class qsim:
    # to simulate a measuremet
    def __init__(self,qmeas,niter = 1000,backend='cdf'):
        self.qmeas = qmeas.probability()
        self.niter = niter
        self.backend = backend
 
    def get_qmeas_simul(self):
        res = []
        for i in range(len(self.qmeas)):
            if self.backend == 'mc':
               res.append(mc(np.real(self.qmeas[i]),self.niter))
            elif self.backend == 'cdf':
               temp = []
               for j in range(self.niter):
                   temp.append(randunit()/np.real(self.qmeas[i]))
               res.append(cdf(temp))
            else:
               raise TypeError('no ', backend) 
        return res    

