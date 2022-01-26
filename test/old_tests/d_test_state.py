from tqix import *
from numpy import pi

j = 17/2
theta = 0.1
phi = 0.2
e = random(8)
print(isnormx(e))
print(e)
print(coherent(5, 0.25j))


H = tensorx(sigmaz(), sigmaz(), sigmaz()) 
print(H)
H2 = ptracex(H,0)
print(H2)

