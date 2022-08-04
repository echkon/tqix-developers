import numpy as np
from tqix.qx import *
from tqix.pis.util import *
from tqix.pis import *
from scipy.sparse import csc_matrix,lil_matrix
import multiprocessing 
import functools 
import torch 

__all__ =['add_noise','calc_rho_0']

def calc_rho_0(rho_0,iks,jmm1,state,all_iks,j_min,j_max,N_in,d_dicke,use_gpu=False,result_queue=None):
    if use_gpu:
        accumulate_states = rho_0
        for ik in iks:
            i,k = ik
            j,m,m1 = jmm1[i,k]
            first_term = 0
            second_term = 0
            third_term = 0
            gamma_1 = 0; gamma_2 = 0; gamma_3 = 0; gamma_4 = 0; gamma_5 = 0; gamma_6 = 0
            gamma_7 = 0; gamma_8 = 0; gamma_9 = 0
            p_jmm1 = state[i,k]

            if j >= j_min and j <= j_max and m > -j and m1 > -j and m < j and m1 < j:
                Lambda_a = get_Lambda(N_in,j,type="a")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j,m,m1]] = 1 
                gamma_1 = m*m1*Lambda_a*dicke_bx_m

            if (j-1) >= j_min and (j-1) <= j_max and (m >= -(j-1) and m1 >= -(j-1)) and (m <= (j-1) and m1 <= (j-1)):
                B_jm_z = get_B(j,m,type="z")
                B_jm1_z = get_B(j,m1,type="z")
                Lambda_b = get_Lambda(N_in,j,type="b")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j-1,m,m1]] = 1 
                gamma_2 = B_jm_z*B_jm1_z*Lambda_b*dicke_bx_m

            if (j+1) <= j_max and (j+1) >= j_min and (m >= -(j+1) and m1 >= -(j+1)) and (m <= (j+1) and m1 <= (j+1)):
                D_jm_z = get_D(j,m,type="z")
                D_jm1_z = get_D(j,m1,type="z")
                Lambda_d = get_Lambda(N_in,j,type="d")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j+1,m,m1]] = 1
                gamma_3 = D_jm_z*D_jm1_z*Lambda_d*dicke_bx_m
            
            first_term = (gamma_1+gamma_2+gamma_3)
            
            if j >= j_min and j <= j_max  and (m-1) >= -j and (m1-1) >= -j and (m-1) <= j and (m1-1) <= j:
                A_jm_minus = get_A(j,m,type="-")
                A_jm1_minus = get_A(j,m1,type="-")
                Lambda_a = get_Lambda(N_in,j,type="a")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j,m-1,m1-1]] = 1
                gamma_4 = A_jm_minus*A_jm1_minus*Lambda_a*dicke_bx_m
            
            if (j-1) >= j_min and (j-1) <= j_max and (m-1) >= -(j-1) and (m1-1) >= -(j-1) and (m-1) <= (j-1) and  (m1-1) <= (j-1):
                B_jm_minus = get_B(j,m,type="-")
                B_jm1_minus = get_B(j,m1,type="-")
                Lambda_b = get_Lambda(N_in,j,type="b")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j-1,m-1,m1-1]] = 1
                gamma_5 = B_jm_minus*B_jm1_minus*Lambda_b*dicke_bx_m
            
            if  (j+1) >= j_min and (j+1) <= j_max and (m-1) >= -(j+1) and (m1-1) >= -(j+1) and (m-1) <= (j+1) and (m1-1)  <= (j+1):
                D_jm_minus = get_D(j,m,type="-")
                D_jm1_minus = get_D(j,m1,type="-")
                Lambda_d = get_Lambda(N_in,j,type="d")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j+1,m-1,m1-1]] = 1
                gamma_6 = D_jm_minus*D_jm1_minus*Lambda_d*dicke_bx_m
            second_term = (gamma_4+gamma_5+gamma_6)/2
            
            if j >= j_min and j <= j_max and (m+1) <= j and (m1+1) <= j and (m+1) >= -j and (m1+1) >= -j:
                A_jm_plus = get_A(j,m,type="+")
                A_jm1_plus = get_A(j,m1,type="+")
                Lambda_a = get_Lambda(N_in,j,type="a")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j,m+1,m1+1]] = 1
                gamma_7 = A_jm_plus*A_jm1_plus*Lambda_a*dicke_bx_m
            
            if (j-1) >= j_min and (j-1) <= j_max and (m+1) >= -(j-1) and (m1+1) >= -(j-1) and (m+1) <= (j-1) and (m1+1) <= (j-1):
                B_jm_plus = get_B(j,m,type="+")
                B_jm1_plus = get_B(j,m1,type="+")
                Lambda_b = get_Lambda(N_in,j,type="b")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j-1,m+1,m1+1]] = 1
                gamma_8 = B_jm_plus*B_jm1_plus*Lambda_b*dicke_bx_m
            
            if (j+1) >= j_min and (j+1) <= j_max and  (m+1) >= -(j+1) and (m1+1) >= -(j+1) and (m+1) <= (j+1) and (m1+1) <= (j+1):
                D_jm_plus = get_D(j,m,type="+")
                D_jm1_plus = get_D(j,m1,type="+")
                Lambda_d = get_Lambda(N_in,j,type="d")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j+1,m+1,m1+1]] = 1
                gamma_9 = D_jm_plus*D_jm1_plus*Lambda_d*dicke_bx_m

        third_term = (gamma_7+gamma_8+gamma_9)/2
        accumulate_states += p_jmm1*(first_term+second_term+third_term)
        if result_queue is not None:
            result_queue.put(accumulate_states)
            result_queue.join()
        else:
            return accumulate_states
    else:
        accumulate_states = csc_matrix((d_dicke, d_dicke), dtype=np.complex)
        for ik in iks:
            i,k = ik 
            j,m,m1 = jmm1[i,k]
            first_term = 0
            second_term = 0
            third_term = 0
            gamma_1 = 0; gamma_2 = 0; gamma_3 = 0; gamma_4 = 0; gamma_5 = 0; gamma_6 = 0
            gamma_7 = 0; gamma_8 = 0; gamma_9 = 0
            p_jmm1 = state[i,k]

            if j >= j_min and j <= j_max and m > -j and m1 > -j and m < j and m1 < j:
                Lambda_a = get_Lambda(N_in,j,type="a")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j,m,m1]] = 1 
                gamma_1 = m*m1*Lambda_a*dicke_bx_m

            if (j-1) >= j_min and (j-1) <= j_max and (m >= -(j-1) and m1 >= -(j-1)) and (m <= (j-1) and m1 <= (j-1)):
                B_jm_z = get_B(j,m,type="z")
                B_jm1_z = get_B(j,m1,type="z")
                Lambda_b = get_Lambda(N_in,j,type="b")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j-1,m,m1]] = 1
                gamma_2 = B_jm_z*B_jm1_z*Lambda_b*dicke_bx_m

            if (j+1) <= j_max and (j+1) >= j_min and (m >= -(j+1) and m1 >= -(j+1)) and (m <= (j+1) and m1 <= (j+1)):
                D_jm_z = get_D(j,m,type="z")
                D_jm1_z = get_D(j,m1,type="z")
                Lambda_d = get_Lambda(N_in,j,type="d")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j+1,m,m1]] = 1               
                gamma_3 = D_jm_z*D_jm1_z*Lambda_d*dicke_bx_m
            
            first_term = (gamma_1+gamma_2+gamma_3)
            
            if j >= j_min and j <= j_max  and (m-1) >= -j and (m1-1) >= -j and (m-1) <= j and (m1-1) <= j:
                A_jm_minus = get_A(j,m,type="-")
                A_jm1_minus = get_A(j,m1,type="-")
                Lambda_a = get_Lambda(N_in,j,type="a")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j,m-1,m1-1]] = 1 
                gamma_4 = A_jm_minus*A_jm1_minus*Lambda_a*dicke_bx_m
            
            if (j-1) >= j_min and (j-1) <= j_max and (m-1) >= -(j-1) and (m1-1) >= -(j-1) and (m-1) <= (j-1) and  (m1-1) <= (j-1):
                B_jm_minus = get_B(j,m,type="-")
                B_jm1_minus = get_B(j,m1,type="-")
                Lambda_b = get_Lambda(N_in,j,type="b")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j-1,m-1,m1-1]] = 1          
                gamma_5 = B_jm_minus*B_jm1_minus*Lambda_b*dicke_bx_m
            
            if  (j+1) >= j_min and (j+1) <= j_max and (m-1) >= -(j+1) and (m1-1) >= -(j+1) and (m-1) <= (j+1) and (m1-1)  <= (j+1):
                D_jm_minus = get_D(j,m,type="-")
                D_jm1_minus = get_D(j,m1,type="-")
                Lambda_d = get_Lambda(N_in,j,type="d")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j+1,m-1,m1-1]] = 1             
                gamma_6 = D_jm_minus*D_jm1_minus*Lambda_d*dicke_bx_m
            
            second_term = (gamma_4+gamma_5+gamma_6)/2
            
            if j >= j_min and j <= j_max and (m+1) <= j and (m1+1) <= j and (m+1) >= -j and (m1+1) >= -j:
                A_jm_plus = get_A(j,m,type="+")
                A_jm1_plus = get_A(j,m1,type="+")
                Lambda_a = get_Lambda(N_in,j,type="a")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j,m+1,m1+1]] = 1  
                gamma_7 = A_jm_plus*A_jm1_plus*Lambda_a*dicke_bx_m
            
            if (j-1) >= j_min and (j-1) <= j_max and (m+1) >= -(j-1) and (m1+1) >= -(j-1) and (m+1) <= (j-1) and (m1+1) <= (j-1):
                B_jm_plus = get_B(j,m,type="+")
                B_jm1_plus = get_B(j,m1,type="+")
                Lambda_b = get_Lambda(N_in,j,type="b")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j-1,m+1,m1+1]] = 1
                gamma_8 = B_jm_plus*B_jm1_plus*Lambda_b*dicke_bx_m
            
            if (j+1) >= j_min and (j+1) <= j_max and  (m+1) >= -(j+1) and (m1+1) >= -(j+1) and (m+1) <= (j+1) and (m1+1) <= (j+1):
                D_jm_plus = get_D(j,m,type="+")
                D_jm1_plus = get_D(j,m1,type="+")
                Lambda_d = get_Lambda(N_in,j,type="d")
                dicke_bx_m = rho_0
                dicke_bx_m[all_iks[j+1,m+1,m1+1]] = 1
                gamma_9 = D_jm_plus*D_jm1_plus*Lambda_d*dicke_bx_m

            third_term = (gamma_7+gamma_8+gamma_9)/2
            accumulate_states += p_jmm1*(first_term+second_term+third_term)

        return accumulate_states

def add_noise(qc,noise=0.3,num_process=None,use_gpu=False,device=None):
    state = qc.state
    d_in = shapex(state)[0]
    N_in = qc.N
    d_dicke = get_dim(N_in)

    if d_in != d_dicke:
        if use_gpu:
            state = torch.nn.functional.pad(state,(0,d_dicke-d_in,0,d_dicke-d_in)) 
        else:
            state = lil_matrix(np.pad(state.toarray(),((0,d_dicke-d_in),(0,d_dicke-d_in))))
    new_Nds = get_Nds(shapex(state)[0])

    assert N_in == new_Nds, "not full block"
    if use_gpu:
        non_zero_arrays = state.nonzero(as_tuple=True)
        iks = list(zip(non_zero_arrays[0].detach().cpu().tolist(),non_zero_arrays[1].detach().cpu().tolist()))
    else:
        non_zero_arrays = state.nonzero()   
        iks = list(zip(non_zero_arrays[0],non_zero_arrays[1]))
    
    jmm1,all_iks  = get_jmm1_idx(N_in)
    if use_gpu:
        rho_0 = torch.zeros(d_dicke, d_dicke).type(torch.complex128).to(device)
        if num_process is not None:
            rho = state.detach().cpu()
        else:
            rho = state
    else:
        rho_0 = csc_matrix((d_dicke, d_dicke), dtype=np.complex)
        rho = state
    j_min = get_jmin(N_in)
    j_max = N_in/2

    run_arguments = []
    len_iks = len(iks)
    if num_process is None:
        pass
    else:
        for i in range(num_process):    
            begin_idx = round(len_iks*i/num_process)
            end_idx = round(len_iks*(i+1)/num_process)
            if use_gpu:
                run_arguments.append((rho_0.detach().cpu(),iks[begin_idx:end_idx],jmm1,state.detach().cpu(),all_iks,j_min,j_max,N_in,d_dicke,use_gpu))
            else:
                run_arguments.append((rho_0,iks[begin_idx:end_idx],jmm1,state,all_iks,j_min,j_max,N_in,d_dicke))

    if use_gpu:
        if num_process is None:
            rho_0 = calc_rho_0(rho_0,iks,jmm1,state,all_iks,j_min,j_max,N_in,d_dicke,use_gpu)
            rho_0 = 4/(3*N_in)*rho_0
            normalized_rho_0 = daggx(rho_0) @ (rho_0)/((daggx(rho_0) @ (rho_0)).diagonal().sum())
            new_state = (1-noise)*rho + noise*normalized_rho_0 
            return new_state
        else:
            import torch.multiprocessing as mp
            processes = []
            accumulate_states=[]
            result_queue = mp.JoinableQueue(1)
            for rank in range(num_process):
                arg_list = list(run_arguments[rank])
                arg_list.append(result_queue)
                run_arguments[rank] = tuple(arg_list)
                p = mp.Process(target=calc_rho_0, args=run_arguments[rank])
                p.start()
                temp_result = result_queue.get()
                result_queue.task_done()
                accumulate_states.append(temp_result)
                del temp_result
                processes.append(p)
            for p in processes:
                p.join()
            rho_0 = functools.reduce(lambda x,y: x+y,accumulate_states)
            rho_0 = 4/(3*N_in)*rho_0
            normalized_rho_0 = daggx(rho_0) @ (rho_0)/((daggx(rho_0) @ (rho_0)).diagonal().sum())
            new_state = (1-noise)*rho + noise*normalized_rho_0 
            return new_state.to(device)
    else:
        if num_process is None:
            rho_0 = calc_rho_0(rho_0,iks,jmm1,state,all_iks,j_min,j_max,N_in,d_dicke,use_gpu)
            rho_0 = 4/(3*N_in)*rho_0
            normalized_rho_0 = daggx(rho_0).dot(rho_0)/((daggx(rho_0).dot(rho_0)).diagonal().sum())
            new_state = (1-noise)*rho + noise*normalized_rho_0             
            return new_state

        else:
            pool = multiprocessing.Pool(processes=num_process)
            accumulate_states = pool.starmap(calc_rho_0,run_arguments)
            pool.close()
            pool.join()    
            rho_0 = functools.reduce(lambda x,y: x+y,accumulate_states)
            rho_0 = 4/(3*N_in)*rho_0
            normalized_rho_0 = daggx(rho_0).dot(rho_0)/((daggx(rho_0).dot(rho_0)).diagonal().sum())
            new_state = (1-noise)*rho + noise*normalized_rho_0             
            return new_state
