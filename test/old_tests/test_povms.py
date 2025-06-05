import time
import os
#import psutil
import numpy as np
from tqix import *
import matplotlib.pyplot as plt

#Parameters
#name = 'MUB.dat'#file name
#f = open(name,'w')
#f.write("#Varied quantity: Dimensions"+'\n')
#f.write("#Output: "+'\n')

da,ti_mub,ti_sic = [],[],[]
N = 10000

for d in range(2,5):
    dtime_mub = 0.0
    dtime_sic = 0.0
    for i in range(N):
        state = random(d)
        ###
        model = qmeas(state,'MUB')
        start = time.time()
        pr = model.probability()
        end = time.time()
        dtime_mub += (end - start)

        ###
        model = qmeas(state,'SIC')
        start = time.time()
        pr = model.probability()
        end = time.time()
        dtime_sic += (end - start)
    
    dtime_mub /= float(N)
    dtime_sic /= float(N)
    ###
    da.append(d)
    ti_mub.append(dtime_mub)
    ti_sic.append(dtime_sic)

dp,ti_pauli = [],[]
for n in range (1,3):
    #n: number of quibts
    dtime_pauli = 0.0
    for i in range(N):
        state = random(2**n)
        ###
        model = qmeas(state,'Pauli')
        start = time.time()
        pr = model.probability()
        end = time.time()
        dtime_pauli += (end - start)
    
    dtime_pauli /= float(N)
    dp.append(2**n)
    ti_pauli.append(dtime_pauli)

# Plot figure
fig, ax1 = plt.subplots(figsize=(12,6)) 
ax1.plot(dp, ti_pauli, 'r')
ax1.plot(da, ti_mub, 'b--') 
ax1.plot(da, ti_sic,'g:') 
ax1.legend(('pauli','mub','sic')) 
ax1.set_xlabel('d') 
ax1.set_ylabel('time (s)')
plt.savefig('time_povm.eps')
plt.show()
