import numpy as np
from tqix import *

d = 2
state = random(d)
m = 0.0,
st = 0.1

qq = []
for i in range(10):
   per = randnormal(m,st,1)+1j*randnormal(m,st,1)
   news = state + per

   #chuan hoa boi A
   A = l2normx(news)
   qq.append(4/A*(d-A)/(4*(d-1)))
  
tb = np.mean(qq)
print(tb)

# Rotation
print(soper(1,'y'))
print(lowering(4))
print(raising(4))

