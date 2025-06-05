from tqix.pis import *
from tqix import *
import numpy as np
from matplotlib import pyplot as plt
import torch 
import numpy as np 

def cost_function(theta,use_gpu=False):
    qc = circuit(N,use_gpu=use_gpu)
    qc.RN(np.pi/2,0)
    qc.OAT(theta[0],"Z") 
    qc.TNT(theta[1],omega=theta[1],gate_type="ZX")
    qc.TAT(theta[2],"ZY")
    if use_gpu:
        loss = torch.real(get_xi_2_S(qc))
    else:
        loss = np.real(get_xi_2_S(qc))
    return  loss

N=100 #number of qubits
route = (("RN2",),("OAT","Z"),("TNT","ZX"),("TAT","ZY")) #define layers for QNG algorithm
loss_dict = {}

#function to optimize circuit of sparse array
def sparse(optimizer,loss_dict,mode):
    objective_function = lambda params: cost_function(params) 
    init_params = [0.00195902, 0.14166777, 0.01656466] #random init parameters for circuit
    _,_,_,loss_hist,time_iters = fit(objective_function,optimizer,init_params,return_loss_hist=True,return_time_iters = True)
    loss_dict[mode] = loss_hist
    return loss_dict,time_iters

#function to optimize circuit of tensor
def tensor(optimizer,loss_dict,mode):
    objective_function = lambda params: cost_function(params,use_gpu=True) 
    init_params = [0.00195902, 0.14166777, 0.01656466] #random init parameters for circuit
    init_params = torch.tensor(init_params).to('cuda').requires_grad_()
    _, _,_,loss_hist,time_iters = fit(objective_function,optimizer,init_params,return_loss_hist=True,return_time_iters = True)
    loss_dict[mode] = loss_hist
    return loss_dict,time_iters

#QNG
optimizer = GD(lr=0.03,eps=1e-10,maxiter=200,use_qng=True,route=route,tol=1e-19,N=N)
loss_dict,_ = tensor(optimizer,loss_dict,"tensor_qng")
print(loss_dict)
#GD
optimizer = GD(lr=0.0001,eps=1e-10,maxiter=200,tol=1e-19,N=N)
loss_dict,_ = tensor(optimizer,loss_dict,"tensor_gd")

#ADAM 
optimizer = ADAM(lr=0.01,eps=1e-10,amsgrad=False,maxiter=200)
loss_dict,sparse_times = sparse(optimizer,loss_dict,"sparse_adam")

optimizer = ADAM(lr=0.001,eps=1e-10,amsgrad=False,maxiter=200)
loss_dict,tensor_times = tensor(optimizer,loss_dict,"tensor_adam")

#plot loss values with respect to iterations
ax = plt.gca() 
ax.plot(range(len(loss_dict["tensor_qng"] )),loss_dict["tensor_qng"] ,'c-*',label=r'$QNG;\eta=0.1$')
ax.plot(range(len(loss_dict["tensor_gd"])),loss_dict["tensor_gd"],'y-^',label=r'$GD;\eta=0.03$')
ax.plot(range(len(loss_dict["tensor_adam"])),loss_dict["tensor_adam"],'g-o',label=r'$Adam;\eta=0.01$')


ax.set_xlabel("iterations")
ax.set_ylabel(r"$C(\theta)$")
ax.set_xlim(0,130)
ax.set_ylim(0,15)
lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.25))
plt.savefig("./loss_bm.eps", bbox_extra_artists=(lgd,), bbox_inches='tight')

#for compare running tume between using tensor and sparse array structure, we plot ADAM as an example
cumsum_vqa_time_res_sparse = np.cumsum(loss_dict['sparse_adam'])
cumsum_vqa_time_res_tensor = np.cumsum(loss_dict['tensor_adam'])
from matplotlib import pyplot as plt
ax = plt.gca() 
ax.plot(range(len(cumsum_vqa_time_res_sparse)),cumsum_vqa_time_res_sparse,'g--',label=r'$CPU$')
ax.plot(range(len(cumsum_vqa_time_res_tensor)),cumsum_vqa_time_res_tensor,'r-',label=r'$GPU$')

ax.set_xlabel("iterations")
ax.set_ylabel("time(s)")
ax.set_ylim(0,100)
lgd = ax.legend(loc='upper right')
plt.savefig("./timetensorsparse.eps", bbox_extra_artists=(lgd,), bbox_inches='tight')
