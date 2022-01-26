import time
#import os
#import psutil
import numpy as np
from tqix import *
import matplotlib.pyplot as plt

N = 100

# For pauli and Stoke
dim_p,time_p = [],[]
dim_s,time_s = [],[]
for n in range (1,4):
    #n: number of quibts
    dtime_p = 0.0
    dtime_s = 0.0
    for i in range(N):
        state = random(2**n)
        ###
        model = qmeas(state,'Pauli')
        #pr = model.probability()
        dtime = model.mtime()
        dtime_p += dtime

        ###
        model = qmeas(state,'Stoke')
        #pr = model.probability()
        dtime = model.mtime()
        dtime_s += dtime

    dtime_p /= float(N)
    dtime_s /= float(N)
    ###
    dim_p.append(2**n)
    time_p.append(dtime_p)
    time_s.append(dtime_s)

# For MUB
dim_mub,time_mub = [],[]
for d in (2,3,4,5,7):
    dtime_mub = 0.0
    for i in range(N):
        state = random(d)
        ###
        model = qmeas(state,'MUB')
        #pr = model.probability()
        dtime = model.mtime()
        dtime_mub += dtime
    dtime_mub /= float(N)
    ###
    dim_mub.append(d)
    time_mub.append(dtime_mub)

# For SIC
dim_sic, time_sic = [],[]
for d in range(2,9):
    dtime_sic = 0.0
    for i in range(N):
        state = random(d)
        ###
        model = qmeas(state,'SIC')
        #pr = model.probability()
        dtime = model.mtime()
        dtime_sic += dtime
    dtime_sic /= float(N)
    ###
    dim_sic.append(d)
    time_sic.append(dtime_sic)

# Plot figure
fig, ax1 = plt.subplots(figsize=(12,8)) 
ax1.plot(dim_p, time_p, marker = 'o')
ax1.plot(dim_p, time_s, marker = '^')
ax1.plot(dim_mub, time_mub, marker = 's') 
ax1.plot(dim_sic, time_sic, marker = 'v') 
ax1.legend(('pauli','stoke','mub','sic')) 
ax1.set_xlabel('d') 
ax1.set_ylabel('time (s)')
plt.ylim(0,0.0006)
plt.savefig('time_povm.eps')
plt.show()
