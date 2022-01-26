from tqix import random, l2normx, randnormal
from tqix import *
import numpy as np
d = 8
m = 0.0
st = 0.1
state = random(d)
norm = l2normx(state)

#print(prime)
#print(norm)
N = 10000000
array = np.zeros(N)
for i in range(N):
    per = [randnormal(m,st,d) + 1j*randnormal(m,st,d)]
    per = daggx(per)
    prime = per + state
    norm = l2normx(prime)
    array[i] = np.real(norm)
#print(array)
#frequency, bins = np.histogram(array, bins=20, range=[0,2])
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,5.4), 'figure.dpi':100})

# Plot Histogram on x
plt.hist(array, bins=200, range=[0,3],histtype='step',density=True)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
#plt.show()
plt.savefig('QFI.pdf')

