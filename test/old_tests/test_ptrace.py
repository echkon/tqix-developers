from tqix import *
import numpy as np

H = tensorx(sigmax(),eyex(2))
rho = ptracex(H,[2])
print(H)
print(rho)


