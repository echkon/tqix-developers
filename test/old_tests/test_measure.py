#This is a test measure function
  
import numpy as np
from tqix import *

state = ghz(1)
print(state)
print(typex(state))

# check arbitary POVMs
print('check arbitaty POVMS')
model = qmeas(state,[sigmax(),sigmay(),sigmaz()])
print(model.probability())


# check Pauli POVMs
print('*****************')
print('check Pauli POVMs')
state = random(2)
model = qmeas(state,'Pauli')
pr = model.probability()
print(pr)
print(sum(pr))

# check Pauli POVMs
print('*****************')
print('check Stoke POVMs')
state = random(4)
model = qmeas(state,'Stoke')
pr = model.probability()
print(pr)
print(sum(pr))

# check MUB POVMs
print('*****************')
print('check MUB POVMs inside')
state = random(2)
model = qmeas(state,'MUB')
pr = model.probability()
print(pr)
print(sum(pr))

# check SIC POVMs
print('*****************')
print('check SIC POVMs')
state = random(2)
model = qmeas(state,'SIC')
pr = model.probability()
print(pr)
print(sum(pr),'tongne')


"""
# check evolution
print('*****************')
print('check evolution')
state = obasis(2,0) # up
model = qevo(state,sigmax(),None,None)
"""

#print('\get out come state')
#print(model.output())

#print('\get measurement')
#print(model.get_qmeas())

nloop = 10000
simul = qsim(model,niter = nloop, backend = 'cdf')
print(simul.get_qmeas_simul())

