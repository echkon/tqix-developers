from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqix.pis.optimizers import ADAM
import torch 

N=50


def cost_function(theta,use_tensor=False,device=None):
    qc = circuit(N,use_tensor=use_tensor,device=device)
    qc.RN(np.pi/2,0)
    qc.OAT(theta[0],"Z",noise=0.05) 
    qc.TNT(theta[1],omega=theta[1],gate_type="ZX")
    qc.TAT(theta[2],"ZY")
    if use_tensor:
        loss = torch.real(get_xi_2_S(qc,use_tensor=use_tensor))
    else:
        loss = np.real(get_xi_2_S(qc,use_tensor=use_tensor))
    return  loss
init_params = torch.tensor([0.00195902, 0.14166777, 0.01656466]).requires_grad_()

objective_function = lambda params: cost_function(params,use_tensor=True,device='cuda') 
route = (("RN2",),("OAT","Z"),("TNT","ZX"),("TAT","ZY"))
optimizer = GD(lr=0.01,eps=1e-10,maxiter=200,use_qng=True,route=route,tol=1e-19,N=N)
# optimizer = ADAM(lr=0.001,eps=1e-10,amsgrad=True,maxiter=550)
# init_params = [0.00195902, 0.14166777, 0.01656466]
opt_params, loss,loss_hist = fit(objective_function,optimizer,init_params,return_loss_hist=True)
# print(opt_params,loss)
plt.figure()
ax = plt.gca() 
ax.plot(list(range(len(loss_hist))),loss_hist)
ax.set_xlabel("iteration")
ax.set_ylabel("loss")
plt.show()
plt.savefig("adam_loss.png")

