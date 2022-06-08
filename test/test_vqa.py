from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt

from tqix.pis.optimizers import ADAM

N=100


def cost_function(theta):
    qc = circuit(N)
    qc.RN(np.pi/2,0)
    qc.OAT(theta[0],"Z")
    qc.TNT(theta[1],"ZX")
    qc.TAT(theta[2],"ZY")
    loss = np.real(get_xi_2_S(qc))
    return loss

objective_function = lambda params: cost_function(params) 
route = (("RN2",),("OAT","Z"),("TNT","ZX"),("TAT","ZY"))
# optimizer = GD(lr=0.01,eps=1e-10,maxiter=5000,tol=1e-19,use_qng=True,route=route,N=N)
optimizer = ADAM(lr=0.001,eps=1e-10,amsgrad=True,maxiter=150)
np.random.seed(3)
# init_params = np.random.uniform(low=0,high=0.12,size=(3))
init_params = [0.00195902, 0.14166777, 0.01656466]
opt_params, loss,loss_hist = fit(objective_function,optimizer,init_params,return_loss_hist=True)
print(opt_params,loss)
plt.figure()
ax = plt.gca() 
ax.plot(list(range(len(loss_hist))),loss_hist)
ax.set_xlabel("iteration")
ax.set_ylabel("loss")
plt.show()
plt.savefig("adam_loss.png")

