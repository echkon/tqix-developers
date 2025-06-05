from tqix import *
import numpy as np

#check basesx state
print('*************************')
print('check basesx')
b = obasis(5,0)
print(b)

# check random state
print('*************************')
print('check random')
b = random(2)
print(b)


#check ghz state
print('*************************')
print('check ghz')
g = ghz(3)
print(g)

#check w state
print('*************************')
print('check w')
g = w(3)
print(g)

#check dick state
print('*************************')
print('check dick')
g = dicke(3,2)
print(g)

#check qtool
print('*************************')
print('check tensor')
print(tensorx(sigmax(),sigmaz()))
print(tensorx(sigmax(),sigmay(),sigmaz()))


