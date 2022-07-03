from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqix.pis.optimizers import ADAM
import torch 
import pickle 

N=100
loss_dict = {}

def cost_function(theta,use_tensor=False):
    qc = circuit(N,use_tensor=use_tensor)
    qc.RN(np.pi/2,0)
    qc.OAT(theta[0],"Z") 
    qc.TNT(theta[1],omega=theta[1],gate_type="ZX")
    qc.TAT(theta[2],"ZY")
    if use_tensor:
        loss = torch.real(get_xi_2_S(qc))
    else:
        loss = np.real(get_xi_2_S(qc))
    return  loss

route = (("RN2",),("OAT","Z"),("TNT","ZX"),("TAT","ZY"))

#sparse
def sparse(optimizer,loss_dict,mode):
    objective_function = lambda params: cost_function(params) 
    init_params = [0.00195902, 0.14166777, 0.01656466]
    opt_params, loss,_,loss_hist,time_iters = fit(objective_function,optimizer,init_params,return_loss_hist=True,return_time_iters = True)
    loss_dict[mode] = loss_hist
    return loss_dict,time_iters
#tensor
def tensor(optimizer,loss_dict,mode):
    objective_function = lambda params: cost_function(params,use_tensor=True) 
    init_params = torch.tensor([0.00195902, 0.14166777, 0.01656466]).to('cuda').requires_grad_()
    opt_params, loss,_,loss_hist,time_iters = fit(objective_function,optimizer,init_params,return_loss_hist=True,return_time_iters = True)
    loss_dict[mode] = loss_hist
    return loss_dict,time_iters

optimizer = GD(lr=0.1,eps=1e-10,maxiter=200,use_qng=True,route=route,tol=1e-19,N=N)
loss_dict,_ = sparse(optimizer,loss_dict,"sparse_qng")
print(loss_dict)
# optimizer = GD(lr=0.03,eps=1e-10,maxiter=200,use_qng=True,route=route,tol=1e-19,N=N)
# loss_dict,_ = tensor(optimizer,loss_dict,"tensor_qng")

# optimizer = GD(lr=0.0001,eps=1e-10,maxiter=200,tol=1e-19,N=N)
# loss_dict,_ = sparse(optimizer,loss_dict,"sparse_gd")
# optimizer = GD(lr=0.0001,eps=1e-10,maxiter=200,tol=1e-19,N=N)
# loss_dict,_ = tensor(optimizer,loss_dict,"tensor_gd")

# optimizer = ADAM(lr=0.01,eps=1e-10,amsgrad=False,maxiter=200)
# loss_dict,sparse_times = sparse(optimizer,loss_dict,"sparse_adam_non_amsgrad")
# print(loss_dict)

# optimizer = ADAM(lr=0.001,eps=1e-10,amsgrad=False,maxiter=200)
# loss_dict,tensor_times = tensor(optimizer,loss_dict,"tensor_adam_non_amsgrad")

# optimizer = ADAM(lr=0.001,eps=1e-10,amsgrad=True,maxiter=200)
# loss_dict,_ = sparse(optimizer,loss_dict,"sparse_adam_amsgrad")
# optimizer = ADAM(lr=0.001,eps=1e-10,amsgrad=True,maxiter=200)
# loss_dict,_ = tensor(optimizer,loss_dict,"tensor_adam_amsgrad")

# time_results={"sparse_time":sparse_times,"tensor_time":tensor_times}

# print("losses:",loss_dict)
# print("time:",time_results)

# with open('vqa_loss_data.pickle', 'wb') as handle:
#     pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vqa_time_data.pickle', 'wb') as handle:
#     pickle.dump(time_results, handle, protocol=pickle.HIGHEST_PROTOCOL)






# print(opt_params,loss)
# plt.figure()
# ax = plt.gca() 
# ax.plot(list(range(len(loss_hist))),loss_hist)
# ax.set_xlabel("iteration")
# ax.set_ylabel("loss")
# plt.show()
# plt.savefig("adam_loss.png")