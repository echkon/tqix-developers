"""
>>> this file is a part of tqix: a Toolbox for Quantum in X
                              x: quantum measurement, quantum metrology, 
                                 quantum tomography, and more.
________________________________
>>> copyright (c) 2019 and later
>>> authors: Binho Le
>>> contributors:
>>> all rights reserved
________________________________
"""

#from numpy import
from functools import partial
import numpy as np
import random
from tqix.qx import *
from tqix.qtool import dotx
from tqix.pis.util import *
from tqix.pis import *
from scipy.sparse import block_diag,csr_matrix
from scipy.sparse.linalg import expm
from functools import *
import torch 

__all__ = ['Gates']

class Gates(object):
    """_summary_
    Class contains gate operations of tqix.pis
    """       

    def __init__(self):
        self.state = None        
        self.theta = None

    def RX(self,theta=None,*args, **kwargs):
        """_summary_

        :param theta: angle, defaults to None
        :type theta: float, optional
        :return: new state after being acted upon by RX  
        :rtype: ndarray, tensor, sparse
        """        
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Rx",noise=noise,num_processes=processes)
    
    def RY(self,theta=None,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by RY  
        """        
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta}
        self.check_input_param(params) 
        return self.gates("Ry",noise=noise,num_processes=processes)
    
    def RZ(self,theta=None,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by RZ
        """        
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Rz",noise=noise,num_processes=processes)
    
    def OAT(self,theta,gate_type,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by OAT
        """        
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta,"gate_type":gate_type}
        self.check_input_param(params)
        gate_type = gate_type.lower()
        if gate_type =="x":
            return self.RX2(theta,noise=noise,num_processes=processes)
        if gate_type =="y":
            return self.RY2(theta,noise=noise,num_processes=processes)
        if gate_type =="z":
            return self.RZ2(theta,noise=noise,num_processes=processes)                    
    
    def TAT(self,theta,gate_type,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by TAT 
        """        
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta,"gate_type":gate_type}
        self.check_input_param(params)
        return self.gates(type=gate_type+"TAT",noise=noise,num_processes=processes)
    
    def TNT(self,theta,gate_type,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by TNT
        """  
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        omega = kwargs.pop('omega', None)
        params = {"theta":theta,"gate_type":gate_type,"omega":omega}
        self.check_input_param(params)
        return self.gates(type=gate_type+"TNT",omega=omega,noise=noise,num_processes=processes)

    def RX2(self,theta=None,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by RX2
        """  
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Rx2",noise=noise,num_processes=processes)
    
    def RY2(self,theta=None,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by RY2
        """  
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Ry2",noise=noise,num_processes=processes)
    
    def RZ2(self,theta=None,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by RZ2 
        """  
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("Rz2",noise=noise,num_processes=processes)
    
    def R_plus(self,theta=None,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by R_plus
        """  
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("R+",noise=noise,num_processes=processes)
    
    def R_minus(self,theta=None,*args, **kwargs):
        """_summary_

        Args:
            theta (float): angle.

        Returns:
            state : state after being acted upon by R_minus
        """  
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta}
        self.check_input_param(params)
        return self.gates("R-",noise=noise,num_processes=processes)
    
    def GMS(self,theta,phi,*args, **kwargs):
        """_summary_

        Args:
            theta (float)
            phi (float)

        Returns:
            state : state after being acted upon by GMS
        """        
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta,"phi":phi}
        self.check_input_param(params)
        return self.gates(type="GMS",phi=phi,noise=noise,num_processes=processes)

    def RN(self,theta,phi,*args, **kwargs):
        """_summary_

        Args:
            theta (float)
            phi (float)

        Returns:
            state : state after being acted upon by RN
        """        
        noise = kwargs.pop('noise', None)
        processes = kwargs.pop('num_processes', None)
        params = {"theta":theta,"phi":phi}
        self.check_input_param(params)
        return self.gates(type="RN",phi=phi,noise=noise,num_processes=processes)

    def check_input_param(self,params):
        """_summary_
        Check input value not None 

        Args:
            params (dict)

        Raises:
            ValueError: inform variable is None
        """        
        self.theta = params["theta"]
        for param,value in params.items():
            if value == None:
                raise ValueError(f"{param} is None")
    
    def get_N_d_d_dicked(self,state):
        """_summary_

        Args:
            state : state of qubits

        Returns:
            d_in (int): dimension of state 
            N_int (int): number of qubits
            d_dicke (int): dimension of dicke basis 
        """        
        d_in = shapex(state)[0]
        N_in = self.N
        d_dicke = get_dim(N_in)
        return d_in,N_in,d_dicke

    def get_J(self,N_in,d_in,d_dicke,type):
        """_summary_

        :param N_in: number of qubits
        :type N_in: int
        :param d_in: dimension of state
        :type d_in: int
        :param d_dicke: dimension of dicke basis 
        :type d_dicke: int
        :param type: type of J operator
        :type type: str
        :return: operator matrix representation 
        :rtype: ndarray, tensor, sparse
        """                        
          
        if "x" in type:
            S = partial(Sx)
        
        elif "y" in type:
            S = partial(Sy)
        
        elif "z" in type:
            S = partial(Sz)
        
        elif "+" in type:
            S = partial(S_plus)
        
        elif "-" in type:
            S = partial(S_minus)

        if d_in != d_dicke:
            if "2" in type:
                if self.use_gpu:
                    J = S(N_in/2,self.use_gpu,self.device).mm(S(N_in/2,self.use_gpu,self.device))
                else:
                    J = S(N_in/2).dot(S(N_in/2))
            else:
                J = S(N_in/2,self.use_gpu,self.device)
        else:
            j_array = get_jarray(N_in)[::-1]
            blocks = []
            for j in j_array:
                if self.use_gpu:
                    blocks.append(S(j,self.use_gpu,self.device))
                else:
                    blocks.append(S(j))
            if not self.use_gpu:
                J = block_diag(blocks,format="csc")
            else:
                J = torch.block_diag(*blocks)
            if "2" in type:
                if self.use_gpu:
                    J = J.mm(J)
                else:
                    J = J.dot(J)     
        return J
    
    def Jx(self):
        """_summary_

        Returns:
            Jx
        """        
        d_in,N_in,d_dicke = self.get_N_d_d_dicked(self.state)
        get_J = partial(self.get_J,N_in,d_in,d_dicke)
        return get_J("x")
    
    def Jy(self):
        """_summary_

        Returns:
            Jy
        """        
        d_in,N_in,d_dicke = self.get_N_d_d_dicked(self.state)
        get_J = partial(self.get_J,N_in,d_in,d_dicke)
        return get_J("y")
    
    def Jz(self):
        """_summary_

        Returns:
            Jz 
        """        
        d_in,N_in,d_dicke = self.get_N_d_d_dicked(self.state)
        get_J = partial(self.get_J,N_in,d_in,d_dicke)
        return get_J("z")
    
    def J_plus(self):
        """_summary_

        Returns:
            J_plus
        """        
        d_in,N_in,d_dicke = self.get_N_d_d_dicked(self.state)
        get_J = partial(self.get_J,N_in,d_in,d_dicke)
        return get_J("+")
    
    def J_minus(self):
        """_summary_

        Returns:
            J_minus
        """        
        d_in,N_in,d_dicke = self.get_N_d_d_dicked(self.state)
        get_J = partial(self.get_J,N_in,d_in,d_dicke)
        return get_J("-")
    
    def var(self,type="",*args, **kwargs):
        """_summary_

        Args:
            type (str): type of variance

        Returns:
            var : variance of state 
        """        
        use_vector = kwargs.pop('use_vector', None)
        n = kwargs.pop('n', None)
        if use_vector:
            exp_val_J_2 = self.expval(type=type+"2",n=n,use_vector=use_vector)
            exp_val_J = self.expval(type=type,n=n,use_vector=use_vector)
        else:
            if "theta" in type:
                exp_val_J_2 = self.expval(type=type+"2")
                exp_val_J = self.expval(type=type)
            else:
                exp_val_J_2 = self.expval(type=type+"2")
                exp_val_J = self.expval(type=type)
        return exp_val_J_2-exp_val_J**2
    
    def expval(self,type="",*args, **kwargs):
        """_summary_

        Args:
            type (str): type of expectation

        Returns:
            expval : expected value of state 
        """        
        state = self.state
        observable = kwargs.pop('observable', None)
        if observable is not None:
            if self.use_gpu:
                return observable.mm(state).diagonal().sum()
            return observable.dot(state).diagonal().sum()
        d_in,N_in,d_dicke = self.get_N_d_d_dicked(state)
        get_J = partial(self.get_J,N_in,d_in,d_dicke)
        type = type.lower()
        count_ops = 0
        for ops in ["x","y","z","+","-"]:
            if ops in type:
                count_ops += 1
        if "j" in type:
                count_ops += 3
        if"cov" in type:
            count_ops += 6
        if "theta" in type or "n1n2_minus" in type or "n1n2_plus" in type:
            count_ops += 2
        
        if "oat'" in type:
            J = get_J(type[4]+'2')
            if '2' in type:
                if self.use_gpu:
                    J = J.mm(J)
                else:
                    J = J.dot(J)
        elif "tnt'" in type:
            J_2 = get_J(type[4]+"2")
            J_prime = get_J(type[5])
            J = (J_2 - J_prime*N_in/2)
            if '2' in type:
                if self.use_gpu:
                    J = J.mm(J)
                else:
                    J = J.dot(J)
        elif "tat'" in type:
            J_2 = get_J(type[4]+"2")
            J_2_prime = get_J(type[5]+"2")
            J = (J_2-J_2_prime)
            if '2' in type:
                if self.use_gpu:
                    J = J.mm(J)
                else:
                    J = J.dot(J)
        else:
            if count_ops == 1:
                J = get_J(type)
            elif count_ops > 1:
                use_vector = kwargs.pop('use_vector', None)
                n = kwargs.pop('n', None)
                if use_vector:
                    order = {"x":0,"y":1,"z":2}

                    if "cov" in type or "n1n2_minus" in type or "n1n2_plus" in type:
                        n1 = kwargs.pop('n1', None)
                        n2 = kwargs.pop('n2', None)
                        if self.use_gpu:
                            list_J_1 = ([get_J(type_J)*(n1[order[type_J]]) for type_J in "xyz"])
                            list_J_2 = ([get_J(type_J)*(n2[order[type_J]]) for type_J in "xyz"])
                        else:
                            list_J_1 = ([get_J(type_J).dot(n1[order[type_J]]) for type_J in "xyz"])
                            list_J_2 = ([get_J(type_J).dot(n2[order[type_J]]) for type_J in "xyz"])
                        J_1 = reduce(lambda x,y:x+y,list_J_1)
                        J_2 = reduce(lambda x,y:x+y,list_J_2)
                        if "cov" in type:
                            if self.use_gpu:
                                J = J_1.mm(J_2) + J_2.mm(J_1)
                            else:
                                J = J_1.dot(J_2) + J_2.dot(J_1)
                        elif "n1n2_minus" in type:
                            if self.use_gpu:
                                J_1 = J_1.mm(J_1)
                                J_2 = J_2.mm(J_2)
                            else:
                                J_1 = J_1.dot(J_1)
                                J_2 = J_2.dot(J_2)
                            J = J_1 - J_2
                        elif "n1n2_plus" in type:
                            if self.use_gpu:
                                J_1 = J_1.mm(J_1)
                                J_2 = J_2.mm(J_2)
                            else:
                                J_1 = J_1.dot(J_1)
                                J_2 = J_2.dot(J_2)
                            J = J_1 + J_2
                    else:
                        if self.use_gpu:
                            list_J = ([get_J(type_J)*(n[order[type_J]]) for type_J in type if type_J != "2"])
                        else:
                            list_J = ([get_J(type_J).dot(n[order[type_J]]) for type_J in type if type_J != "2"])
                        J = reduce(lambda x,y:x+y,list_J)
                        if "2" in type:
                            if self.use_gpu:
                                J = J.mm(J)
                            else:
                                J = J.dot(J)
                elif "theta" in type:
                    J = np.cos(self.theta)*get_J("y")+np.sin(self.theta)*get_J("z")
                    if "2" in type:
                        if self.use_gpu:
                            J = J.mm(J)
                        else:
                            J = J.dot(J)
                else:
                    if "j" in type and "2" in type:
                        list_J = ([get_J(type_J+"2") for type_J in "xyz"])
                    elif "2" in type:
                        list_J = ([get_J(type_J+"2") for type_J in type if type_J != "2"])
                    J = reduce(lambda x,y:x+y,list_J)
        if self.use_gpu:
            return (J.type(torch.complex128) @ state.type(torch.complex128)).diagonal().sum()
        else:
            return J.dot(state).diagonal().sum()


    def gates(self,type="",*args, **kwargs):
        """_summary_

        Args:
            type (str): type of gate

        Returns:
            state : state after being acted upon by gate operation 
        """        
        state = self.state
        d_in,N_in,d_dicke = self.get_N_d_d_dicked(state)
        get_J = partial(self.get_J,N_in,d_in,d_dicke)
        
        type = type.lower()
        count_ops = 0
        for ops in ["x","y","z","+","-"]:
            if ops in type:
                count_ops += 1
        if any([True for gate in ["rn","gms"] if gate in type]):
            count_ops += 2
        if count_ops == 1:
            J = get_J(type)
            if self.use_gpu:
                expJ = torch.matrix_exp(-1j*self.theta*J)
            else:
                expJ = expm(-1j*self.theta*J)
        elif count_ops == 2:
            if "tat" in type:
                J_2 = get_J(type[0]+"2")
                J_2_prime = get_J(type[1]+"2")
                if self.use_gpu:
                    expJ = torch.matrix_exp(-1j*self.theta*(J_2-J_2_prime))
                else:
                    expJ = expm(-1j*self.theta*(J_2-J_2_prime))
            elif "tnt" in type:
                J_2 = get_J(type[0]+"2")
                J_prime = get_J(type[1])
                omega = kwargs.pop('omega', None)
                if self.use_gpu:
                    expJ = torch.matrix_exp(-1j*(self.theta*J_2 - omega*J_prime))
                else:
                    expJ = expm(-1j*(self.theta*J_2 - omega*J_prime))
            elif "gms" in type:
                phi = kwargs.pop('phi', None)
                J = get_J("x")
                J_prime = get_J("y")
                S_phi = 2*(J*np.cos(phi)+J_prime*np.sin(phi))
                if self.use_gpu:
                    expJ = torch.matrix_exp(-1j*self.theta*(S_phi@S_phi)/4)
                else:
                    expJ = expm(-1j*self.theta*S_phi.dot(S_phi)/4)
            elif "rn" in type:
                phi = kwargs.pop('phi', None)
                J = get_J("x")
                J_prime = get_J("y")
                if self.use_gpu:
                    expJ = torch.matrix_exp(1j*self.theta*(J*np.sin(phi)-J_prime*np.cos(phi)))
                else:
                    expJ = expm(1j*self.theta*(J*np.sin(phi)-J_prime*np.cos(phi)))

        expJ_conj = daggx(expJ)
        if self.use_gpu:
            new_state = expJ.type(torch.complex128) @ self.state.type(torch.complex128) @ expJ_conj.type(torch.complex128)
        else:
            new_state = expJ.dot(self.state).dot(expJ_conj)
        self.state = new_state

        noise = kwargs.pop('noise', None)
        
        if noise is not None:
            if not self.use_gpu:
                if self.num_process is not None:
                    new_state = add_noise(self,noise,num_process=self.num_process)
                else:
                    new_state = add_noise(self,noise)
            else:
                if self.num_process is not None:
                    new_state = add_noise(self,noise,num_process=self.num_process,use_gpu=self.use_gpu,device=self.device)
                else:
                    new_state = add_noise(self,noise,use_gpu=self.use_gpu,device=self.device)
            self.state = new_state

        return self
    
    def measure(self,num_shots = None):
        """_summary_

        Args:
            num_shots (int): number of shots for measurement

        Returns:
            result : resulted measurement 
        """        
        state = self.state 
        device = self.device
        return_tensor = False
        if torch.is_tensor(state):
            if state.is_cuda:
                state = state.detach().cpu().numpy()
            else:
                state = state.numpy()
            return_tensor = True 
        t_prob = np.real(csr_matrix(state).diagonal())
        result = np.zeros_like(t_prob)
        mask_zeros = t_prob != 0
        for _ in range(num_shots):    
            rand_prob = np.array([random.uniform(0,1)]*t_prob.shape[0])
            mask_prob_ge = t_prob > rand_prob
            result[mask_zeros & mask_prob_ge] += 1
        result /= num_shots
        result[-1] = 1 - np.sum(result[:-1])
        if result[-1] < 0:
            result[-1] = 0
        if return_tensor:
            return torch.tensor(result).to(device)
        return result   
