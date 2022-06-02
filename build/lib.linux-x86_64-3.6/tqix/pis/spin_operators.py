import numpy as np
from scipy.sparse import csc_matrix


__all__ = ['Sx','Sy','Sz','S_plus','S_minus','S_2']

def Sx(S):
    dim = int(2*S+1)
    m,m_prime = np.arange(-S,S+1)[::-1],np.arange(-S,S+1)[::-1]
    m_ind,m_prime_ind = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
    m_prime_ind_plus = m_prime_ind+1
    m_ind_plus = m_ind+1
    non_zero_row_inds,non_zero_col_inds =  np.nonzero((m_ind_plus == m_prime_ind).astype(np.int64)+(m_ind == m_prime_ind_plus).astype(np.int64))
    non_zero_values = 1/2*np.sqrt(S*(S+1)-np.multiply(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
    opr = csc_matrix((non_zero_values, (non_zero_row_inds,non_zero_col_inds)), shape=(dim, dim))
    return opr

def Sy(S):
    dim = int(2*S+1)
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

def Sz(S):
    dim = int(2*S+1)
    m,_ = np.arange(-S,S+1)[::-1],np.arange(-S,S+1)[::-1]
    m_ind,m_prime_ind = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
    kron_delta = (m_ind == m_prime_ind).astype(np.int64) 
    non_zero_row_inds,non_zero_col_inds = np.nonzero(kron_delta)
    non_zero_kron_delta = kron_delta[non_zero_row_inds,non_zero_col_inds]
    non_zero_values = non_zero_kron_delta*m[non_zero_row_inds]
    opr = csc_matrix((non_zero_values, (non_zero_row_inds,non_zero_col_inds)), shape=(dim, dim))
    return opr

def S_minus(S):
    dim = int(2*S+1)
    m,m_prime = np.arange(-S,S+1)[::-1],np.arange(-S,S+1)[::-1]
    m_ind,m_prime_ind = np.meshgrid(np.arange(dim), np.arange(dim), indexing='ij')
    m_prime_ind_plus = m_prime_ind+1
    non_zero_row_inds,non_zero_col_inds = np.nonzero((m_prime_ind_plus==m_ind).astype(np.int64))
    non_zero_values = np.sqrt(S*(S+1)-np.multiply(m[non_zero_row_inds],m_prime[non_zero_col_inds]))
    opr = csc_matrix((non_zero_values, (non_zero_row_inds,non_zero_col_inds)), shape=(dim, dim))
    return opr

def S_plus(S):
    dim = int(2*S+1)
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