#This is an example of strong DSM

from numpy import sqrt

import numpy as np
from tqix import *
from tqix.dsm.execute import *

N = 3
exi = 1

state = random(N)
print(typex(state),' => ')
print(state)
print('is normalize = ', isnormx(state))
#print('get_module = ', get_module(state))

niter = 1000
nsamp = 100 #fixed
theta = 0.5

m = 0.0
st = 0.1

model = execute(state,niter,nsamp,theta,m,st)

"""
>>> below we call model.job('...')
>>> '...' can be: weak, strong, prob, prob_1,prob_2
"""
job = 'prob_1'
result = model.job(job)
print('\n *** Result for', job, '*** \n')
print(result)
