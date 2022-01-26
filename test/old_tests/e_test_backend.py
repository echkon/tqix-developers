from datetime import datetime
from tqix import *
import numpy as np
import matplotlib.pyplot as plt

def func(n):
   fx = []
   x = np.linspace(0,5,n)
   for i in range(n):
       fx.append(np.exp(-x[i]))
   return fx

samp = 1000 #number of sample
ite  = 1000 #number interation

fx = func(samp)
mc_sim = []
cdf_sim = []

mc_start = datetime.now()
for i in fx:
    mc_sim.append(mc(i,ite))
mc_stop = datetime.now()
mc_time = mc_stop - mc_start

cdf_start = datetime.now()
for i in fx:
    temp = []
    for j in range(ite):
        temp.append(randunit()/i)
    cdf_sim.append(cdf(temp))
cdf_stop = datetime.now()
cdf_time = cdf_stop - cdf_start

x = np.linspace(0,5,samp)
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(x,mc_sim,'b+')
ax.plot(x,cdf_sim,'g.')
ax.plot(x, fx, 'r')
ax.legend(("mc","cdf","fx"))
ax.set_xlabel('x')
ax.set_ylabel('exponential');
plt.savefig('fig1.eps')
plt.show()

#for i in range(samp):
#  print(x[i],mc_sim[i],cdf_sim[i])

#print(ite,mc_time.total_seconds(),mc_time.total_seconds())


# time check
mc_times = []
cdf_times = []
ites = []
for ite in range(1000,11000,1000):
    mc_start = datetime.now()
    for i in fx:
        mc_sim.append(mc(i,ite))
    mc_stop = datetime.now()
    delt = mc_stop - mc_start
    delt = delt.total_seconds()
    mc_times.append(delt)

    cdf_start = datetime.now()
    for i in fx:
        temp = []
        for j in range(ite):
            temp.append(randunit()/i)
        cdf_sim.append(cdf(temp))
    cdf_stop = datetime.now()
    delt = cdf_stop - cdf_start
    delt = delt.total_seconds()
    cdf_times.append(delt)

    ites.append(ite)
    ite *= 10
    #print(ite,mc_time.total_seconds(),cdf_time.total_seconds())

fig, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(ites, mc_times, 'r')
ax1.plot(ites, cdf_times,'b--')
ax1.legend(("mc","cdf"))
ax1.set_xlabel('iteration')
ax1.set_ylabel('exponential');
plt.savefig('fig2.eps')
plt.show


