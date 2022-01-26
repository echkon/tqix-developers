import time
import os
import psutil
import numpy as np
from tqix import *

#Parameters
name = 'SIC.dat'#file name
f = open(name,'w')
f.write("#Varied quantity: Dimensions"+'\n')
f.write("#Output: "+'\n')

d = 4
N = 10
state = random(d)
###
model = qmeas(state,'SIC')
a = time.time()
for i in range(N):
    pr = model.probability()

b = time.time()
###
f.write(str(N)+'  ')
f.write(str(b - a)+'\n')
print(b-a)

N = 100
state = random(d)
###
model = qmeas(state,'SIC')
a = time.time()
for i in range(N):
    pr = model.probability()

b = time.time()
###
f.write(str(N)+'  ')
f.write(str(b - a)+'\n')
print(b-a)

N = 1000
state = random(d)
###
model = qmeas(state,'SIC')
a = time.time()
for i in range(N):
    pr = model.probability()

b = time.time()
###
f.write(str(N)+'  ')
f.write(str(b - a)+'\n')
print(b-a)

N = 10000
state = random(d)
###
model = qmeas(state,'SIC')
a = time.time()
for i in range(N):
    pr = model.probability()

b = time.time()
###
f.write(str(N)+'  ')
f.write(str(b - a)+'\n')
print(b-a)

f.close()
