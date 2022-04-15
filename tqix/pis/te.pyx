import numpy as np
cimport numpy as np 
from numpy cimport ndarray 
cimport cython 
np.import_array()

ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
cdef get_num_block(int N):    
    return <int>(N/2 + 1 -1/2*(N % 2))

@cython.boundscheck(False)
cdef get_jmin(int N):
    if N % 2 == 0:
        return 0
    else:
        return 0.5


@cython.boundscheck(False)
cdef get_jarray(int N):
    """ to get array of j from N

    Paramater:
    -------------
    N: number of particle

    Return:
    -------------
    jarray: a array of j"""

    cdef np.ndarray jarray = np.arange(get_jmin(N),N/2 + 1,1)
    return jarray

@cython.boundscheck(False)
cdef get_array_block(int N):
    # an array of block
    num_block = get_num_block(N)
    cdef np.ndarray array_block =  (np.arange(1,num_block+1)*(N+2-np.arange(1,num_block+1))).astype(np.int32)
    # [i * (N+2-i) for i in range(1, num_block+1)]
    return array_block

@cython.boundscheck(False)
cdef get_jmm1_idx(int N):
    # get index i,k of density matrix from j,m,m1
    # and revert j,m,m1 from i,k
    # ref. qutip
    
    ik = {}
    jmm1 = {}
    
    block = get_array_block(N)
    j_vary = get_jarray(N)
    for j in j_vary:
        m_vary = get_marray(j)
        for m in m_vary:
            for m1 in m_vary:
                i, k = get_midx(N,j,m,m1,block)
                jmm1[(i, k)] = (j, m, m1)
                ik[(j, m, m1)] = (i, k)
    return [jmm1,ik]

@cython.boundscheck(False)
cdef get_midx(int N,int j,int m,int m1,np.ndarray[int, ndim=1] block):
    # to get index in density matrix
    # ref. qutip
    k = <int>(j-m1)
    kp = <int>(j-m)
    block_num = <int>(N/2 - j) #0,1,2,...
    offset = 0
    
    if block_num > 0:
        offset = block[block_num - 1]
    i = kp + offset
    k = k + offset
    return (i,k)

@cython.boundscheck(False)
cdef get_marray(int j):
    cdef np.ndarray marray = np.arange(-j,j+1,1)
    return marray

@cython.boundscheck(False)
cdef get_A(int j,int m,str type=""):
    if type == "+":
        return np.array(np.sqrt((j-m)*(j+m+1))).astype(complex)
    elif type == "-":
        return np.array(np.sqrt((j+m)*(j-m+1))).astype(complex)
    else:
        return np.array(m).astype(complex)

@cython.boundscheck(False)
cdef get_Lambda(int N,int j,str type=""):
    if type == "a":
        return np.array((N/2+1)/(2*j*(j+1))).astype(complex)
    elif type == "b":
        return np.array((N/2+j+1)/(2*j*(2*j+1))).astype(complex)
    else:
        return np.array((N/2-j)/(2*(j+1)*(2*j+1))).astype(complex)

@cython.boundscheck(False)
cdef get_dim(int N):
    d = (N/2 + 1)**2 - (N % 2)/4
    return <int>d
    
@cython.boundscheck(False)
cdef dicke_bx(int N,jmm1):
    dim = get_dim(N)
    cdef np.ndarray rho = np.zeros((dim,dim),dtype = complex)
    ik = get_jmm1_idx(N)[1] # return i,k from jmm1
    for key in jmm1:
        i,k = ik[key]
        rho[i,k] = jmm1[key]
    return rho

@cython.boundscheck(False)
cpdef cacl(np.ndarray[int, ndim=2] iks,jmm1,state,int j_min,int j_max,int N_in,int d_dicke):
    for ik in iks:
        ik =  tuple(ik)
        j,m,m1 = jmm1[ik]
        i,k = ik
        if j >= j_min and j <= j_max  and (m-1) >= -j and (m1-1) >= -j and (m-1) <= j and (m1-1) <= j:
            A_jm_minus = get_A(j,m,type="-")
            A_jm1_minus = get_A(j,m1,type="-")
            Lambda_a = get_Lambda(N_in,j,type="a")
            print(j,m,m1,N_in)
            print( A_jm_minus,A_jm1_minus,Lambda_a,dicke_bx(N_in,{(j,m-1,m1-1):1}))
            gamma_4 = A_jm_minus*A_jm1_minus*Lambda_a*dicke_bx(N_in,{(j,m-1,m1-1):1})
    return gamma_4