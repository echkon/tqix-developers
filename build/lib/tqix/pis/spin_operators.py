import numpy as np
from scipy.sparse import csc_matrix
import torch 

__all__ = ['Sx','Sy','Sz','S_plus','S_minus','S_2']

def Sx(S,use_gpu = False,device='cuda'):
    dim = int(2*S+1)
    if use_gpu:
        m,m_prime = torch.arange(-S,S+1).flip([0]).to(device),torch.arange(-S,S+1).flip([0]).to(device)
        m_ind,m_prime_ind = torch.meshgrid(torch.arange(dim).to(device), torch.arange(dim).to(device),indexing='ij')
        m_prime_ind_plus = m_prime_ind+1
        m_ind_plus = m_ind+1
        non_zero_row_inds,non_zero_col_inds =  torch.nonzero((m_ind_plus == m_prime_ind).type(torch.int64)+(m_ind == m_prime_ind_plus).type(torch.int64),as_tuple=True)
        coor = torch.stack([non_zero_row_inds,non_zero_col_inds])
        non_zero_values = 1/2*torch.sqrt(S*(S+1)-torch.mul(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
        opr = torch.sparse_coo_tensor(coor, non_zero_values, (dim, dim)).to_dense()
    else: 
        m,m_prime = np.arange(-S,S+1)[::-1],np.arange(-S,S+1)[::-1]
        m_ind,m_prime_ind = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
        m_prime_ind_plus = m_prime_ind+1
        m_ind_plus = m_ind+1
        non_zero_row_inds,non_zero_col_inds =  np.nonzero((m_ind_plus == m_prime_ind).astype(np.int64)+(m_ind == m_prime_ind_plus).astype(np.int64))
        non_zero_values = 1/2*np.sqrt(S*(S+1)-np.multiply(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
        opr = csc_matrix((non_zero_values, (non_zero_row_inds,non_zero_col_inds)), shape=(dim, dim))
    return opr

def Sy(S,use_gpu = False,device='cuda'):
    dim = int(2*S+1)
    if use_gpu:
        m,m_prime = torch.arange(-S,S+1).flip([0]).to(device),torch.arange(-S,S+1).flip([0]).to(device)
        m_ind,m_prime_ind = torch.meshgrid(torch.arange(dim).to(device), torch.arange(dim).to(device),indexing='ij')
        m_prime_ind_plus = m_prime_ind+1
        m_ind_plus = m_ind+1
        kron_delta = (m_prime_ind == m_ind_plus).type(torch.int64) - (m_prime_ind_plus == m_ind).type(torch.int64) 
        non_zero_row_inds,non_zero_col_inds = torch.nonzero(kron_delta,as_tuple=True)
        coor = torch.stack([non_zero_row_inds,non_zero_col_inds])
        non_zero_kron_delta = kron_delta[non_zero_row_inds,non_zero_col_inds]
        non_zero_values = non_zero_kron_delta*1/(2j)*torch.sqrt(S*(S+1)-torch.mul(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
        opr = torch.sparse_coo_tensor(coor, non_zero_values, (dim, dim)).to_dense()
    else:
        m,m_prime = np.arange(-S,S+1)[::-1],np.arange(-S,S+1)[::-1]
        m_ind,m_prime_ind = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
        m_prime_ind_plus = m_prime_ind+1
        m_ind_plus = m_ind+1
        kron_delta = (m_prime_ind == m_ind_plus).astype(np.int64) - (m_prime_ind_plus == m_ind).astype(np.int64) 
        non_zero_row_inds,non_zero_col_inds = np.nonzero(kron_delta)
        non_zero_kron_delta = kron_delta[non_zero_row_inds,non_zero_col_inds]
        non_zero_values = non_zero_kron_delta*np.array(1/(2j))*np.sqrt(S*(S+1)-np.multiply(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
        opr = csc_matrix((non_zero_values, (non_zero_row_inds,non_zero_col_inds)), shape=(dim, dim))
    return opr

def Sz(S,use_gpu = False,device='cuda'):
    dim = int(2*S+1)
    if use_gpu:
        m,_ = torch.arange(-S,S+1).flip([0]).to(device),torch.arange(-S,S+1).flip([0]).to(device)
        m_ind,m_prime_ind = torch.meshgrid(torch.arange(dim).to(device), torch.arange(dim).to(device), indexing='ij')
        kron_delta = (m_ind == m_prime_ind).type(torch.int64) 
        non_zero_row_inds,non_zero_col_inds = torch.nonzero(kron_delta,as_tuple=True)
        non_zero_kron_delta = kron_delta[non_zero_row_inds,non_zero_col_inds]
        non_zero_values = non_zero_kron_delta*m[non_zero_row_inds]
        coor = torch.stack([non_zero_row_inds,non_zero_col_inds])
        opr = torch.sparse_coo_tensor(coor, non_zero_values, (dim, dim)).to_dense()
    else:
        m,_ = np.arange(-S,S+1)[::-1],np.arange(-S,S+1)[::-1]
        m_ind,m_prime_ind = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
        kron_delta = (m_ind == m_prime_ind).astype(np.int64) 
        non_zero_row_inds,non_zero_col_inds = np.nonzero(kron_delta)
        non_zero_kron_delta = kron_delta[non_zero_row_inds,non_zero_col_inds]
        non_zero_values = non_zero_kron_delta*m[non_zero_row_inds]
        opr = csc_matrix((non_zero_values, (non_zero_row_inds,non_zero_col_inds)), shape=(dim, dim))
    return opr

def S_minus(S,use_gpu = False,device='cuda'):
    dim = int(2*S+1)
    if use_gpu:
        m,m_prime = torch.arange(-S,S+1).flip([0]).to(device),torch.arange(-S,S+1).flip([0]).to(device)
        m_ind,m_prime_ind = torch.meshgrid(torch.arange(dim).to(device), torch.arange(dim).to(device), indexing='ij')
        m_prime_ind_plus = m_prime_ind+1
        non_zero_row_inds,non_zero_col_inds = torch.nonzero((m_prime_ind_plus==m_ind).type(torch.int64),as_tuple=True)
        non_zero_values = torch.sqrt(S*(S+1)-torch.mul(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
        coor = torch.stack([non_zero_row_inds,non_zero_col_inds])
        opr = torch.sparse_coo_tensor(coor, non_zero_values, (dim, dim)).to_dense()
    else:
        m,m_prime = np.arange(-S,S+1)[::-1],np.arange(-S,S+1)[::-1]
        m_ind,m_prime_ind = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
        m_prime_ind_plus = m_prime_ind+1
        non_zero_row_inds,non_zero_col_inds = np.nonzero((m_prime_ind_plus==m_ind).astype(np.int64))
        non_zero_values = np.sqrt(S*(S+1)-np.multiply(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
        opr = csc_matrix((non_zero_values, (non_zero_row_inds,non_zero_col_inds)), shape=(dim, dim))
    return opr

def S_plus(S,use_gpu = False,device='cuda'):
    dim = int(2*S+1)
    if use_gpu:
        m,m_prime = torch.arange(-S,S+1).flip([0]).to(device),torch.arange(-S,S+1).flip([0]).to(device)
        m_ind,m_prime_ind = torch.meshgrid(torch.arange(dim).to(device), torch.arange(dim).to(device), indexing='ij')
        m_ind_plus = m_ind+1
        non_zero_row_inds,non_zero_col_inds =  torch.nonzero((m_prime_ind==m_ind_plus).type(torch.int64),as_tuple=True)
        non_zero_values = torch.sqrt(S*(S+1)-torch.mul(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
        coor = torch.stack([non_zero_row_inds,non_zero_col_inds])
        opr = torch.sparse_coo_tensor(coor, non_zero_values, (dim, dim)).to_dense()
    else:
        m,m_prime = np.arange(-S,S+1)[::-1],np.arange(-S,S+1)[::-1]
        m_ind,m_prime_ind = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
        m_ind_plus = m_ind+1
        non_zero_row_inds,non_zero_col_inds = np.nonzero((m_prime_ind == m_ind_plus).astype(np.int64))
        non_zero_values = np.sqrt(S*(S+1)-np.multiply(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
        opr = csc_matrix((non_zero_values, (non_zero_row_inds,non_zero_col_inds)), shape=(dim, dim))
    return opr

def S_2(S):
    dim = int(2*S+1)
    m_ind,m_prime_ind = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
    non_zero_row_inds,non_zero_col_inds = np.nonzero((m_ind == m_prime_ind).astype(np.int64))
    non_zero_values = np.array([S*(S+1)]*len(non_zero_row_inds))
    opr = csc_matrix((non_zero_values, (non_zero_row_inds,non_zero_col_inds)), shape=(dim, dim))
    return opr
 