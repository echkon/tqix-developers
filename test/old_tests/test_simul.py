from tqix import dotx, daggx
import numpy as np
from numpy import sqrt
"""
state = random(3)
model = qmeas(state,'MUB')
simul = qsim(model,1000)
var = simul.get_qmeas_simul()
print(var)
c5 = np.sqrt(5)
i = np.array([1/c5,2/c5,0,0])
j = np.array([1/c5,2/(5+c5),sqrt(0.5+0.5/c5),0])
dot = dotx(daggx(j),j)
#print(dot)
print((abs(dot))**2)
"""
a = _sic_(4)
