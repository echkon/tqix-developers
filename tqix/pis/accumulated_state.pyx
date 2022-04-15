from re import M
import numpy as np
cimport numpy as np 
from numpy cimport ndarray 
cimport cython 
np.import_array()
from scipy.sparse import csc_matrix
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
cdef get_dim(int N):
    """ to get dimension from number of particle N
    Paramater:
    -------------
    N: number of particle
    
    Return:
    -------------
    d: dimension """
    
    d = (N/2 + 1)**2 - (N % 2)/4
    return <int>d

@cython.boundscheck(False)
cdef get_num_block(int N):
    """ to get number of block
    
    Paramater:
    -------------
    N: number of particle
    
    Return:
    -------------
    Nbj: Number of block j """
    
    return <int>(N/2 + 1 -1/2*(N % 2))

@cython.boundscheck(False)
cdef get_array_block(int N):
    # an array of block
    num_block = get_num_block(N)
    cdef np.ndarray array_block =  (np.arange(1,num_block+1)*(N+2-np.arange(1,num_block+1))).astype(np.int32)
    # [i * (N+2-i) for i in range(1, num_block+1)]
    return array_block

@cython.boundscheck(False)
cdef get_jmin(int N):
    """ to get j min

    Paramater:
    -------------
    N: number of particle

    Return:
    -------------
    jmin: min of j """

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
cdef get_marray(int j):
    """ to get array of m from j

    Paramater:
    -------------
    j: number of j

    Return:
    -------------
    marray: a array of j"""

    cdef np.ndarray marray = np.arange(-j,j+1,1)
    return marray

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
cdef get_A(int j,int m,str type=""):
    if type == "+":
        return np.array(np.sqrt((j-m)*(j+m+1))).astype(complex)
    elif type == "-":
        return np.array(np.sqrt((j+m)*(j-m+1))).astype(complex)
    else:
        return np.array(m).astype(complex)

@cython.boundscheck(False)
cdef get_B(int j,int m,str type=""):
    if type == "+":
        return np.sqrt((j-m)*(j-m-1)).astype(complex)
    elif type == "-":
        return (-np.sqrt((j+m)*(j+m-1))).astype(complex)
    else:
        return np.sqrt((j+m)*(j-m)).astype(complex)

@cython.boundscheck(False)
cdef get_D(int j,int m,str type=""):
    if type == "+":
        return (-np.sqrt((j+m+1)*(j+m+2))).astype(complex)
    if type == "-":
        return np.sqrt((j-m+1)*(j-m+2)).astype(complex)
    else:
        return np.sqrt((j+m+1)*(j-m+1)).astype(complex)

@cython.boundscheck(False)
cdef get_Lambda(int N,int j,str type=""):
    if type == "a":
        return np.array((N/2+1)/(2*j*(j+1))).astype(complex)
    elif type == "b":
        return np.array((N/2+j+1)/(2*j*(2*j+1))).astype(complex)
    else:
        return np.array((N/2-j)/(2*(j+1)*(2*j+1))).astype(complex)

@cython.boundscheck(False)
cdef dicke_bx(int N,jmm1):
    # create a dicke basis follow jmm1
    # jmm1 as {(j,m,m1):p}
    
    dim = get_dim(N)
    cdef np.ndarray rho = np.zeros((dim,dim),dtype = complex)
    ik = get_jmm1_idx(N)[1] # return i,k from jmm1
    for key in jmm1:
        i,k = ik[key]
        rho[i,k] = jmm1[key]
    return rho


@cython.boundscheck(False)
cpdef calc_rho_0(np.ndarray[int, ndim=2] iks,jmm1,state,int j_min,int j_max,int N_in,int d_dicke):
    cdef np.ndarray accumulate_states = np.zeros((d_dicke, d_dicke), dtype=np.complex)
    cdef DTYPE_t first_term
    cdef DTYPE_t second_term
    cdef DTYPE_t third_term
    cdef DTYPE_t gamma_1
    cdef DTYPE_t gamma_2
    cdef DTYPE_t gamma_3
    cdef DTYPE_t gamma_4
    cdef DTYPE_t gamma_5
    cdef DTYPE_t gamma_6
    cdef DTYPE_t gamma_7
    cdef DTYPE_t gamma_8
    cdef DTYPE_t gamma_9
    
    for ik in iks:
        ik =  tuple(ik)
        j,m,m1 = jmm1[ik]
        j = <int> j 
        m = <int> m 
        m1 = <int> m1
        i,k = ik
        first_term = 0
        second_term = 0
        third_term = 0
        gamma_1 = 0;  gamma_2 = 0; gamma_3 = 0; gamma_4 = 0; gamma_5 = 0; gamma_6 = 0
        gamma_7 = 0; gamma_8 = 0; gamma_9 = 0
        p_jmm1 = state[i,k]
        print(j,j_min,j_max,m,m1)
        if j >= j_min and j <= j_max and m > -j and m1 > -j and m < j and m1 < j:
            Lambda_a = get_Lambda(N_in,j,type="a")
            # print("gamm1:",np.array(m),np.array(m1),Lambda_a,dicke_bx(N_in,{(j,m,m1):1}))
            gamma_1 = np.array(m)*np.array(m1)*Lambda_a*dicke_bx(N_in,{(j,m,m1):1})

        if (j-1) >= j_min and (j-1) <= j_max and (m >= -(j-1) and m1 >= -(j-1)) and (m <= (j-1) and m1 <= (j-1)):
            B_jm_z = get_B(j,m,type="z")
            B_jm1_z = get_B(j,m1,type="z")
            Lambda_b = get_Lambda(N_in,j,type="b")
            gamma_2 = B_jm_z*B_jm1_z*Lambda_b*dicke_bx(N_in,{(j-1,m,m1):1}) 

        if (j+1) <= j_max and (j+1) >= j_min and (m >= -(j+1) and m1 >= -(j+1)) and (m <= (j+1) and m1 <= (j+1)):
            D_jm_z = get_D(j,m,type="z")
            D_jm1_z = get_D(j,m1,type="z")
            Lambda_d = get_Lambda(N_in,j,type="d")
            gamma_3 = D_jm_z*D_jm1_z*Lambda_d*dicke_bx(N_in,{(j+1,m,m1):1})
        
        first_term = (gamma_1+gamma_2+gamma_3)
        
        if j >= j_min and j <= j_max  and (m-1) >= -j and (m1-1) >= -j and (m-1) <= j and (m1-1) <= j:
            A_jm_minus = get_A(j,m,type="-")
            A_jm1_minus = get_A(j,m1,type="-")
            Lambda_a = get_Lambda(N_in,j,type="a")
            matrix = dicke_bx(N_in,{(j,m-1,m1-1):1})
            gamma_4 = A_jm_minus*A_jm1_minus*Lambda_a*matrix
        
        if (j-1) >= j_min and (j-1) <= j_max and (m-1) >= -(j-1) and (m1-1) >= -(j-1) and (m-1) <= (j-1) and  (m1-1) <= (j-1):
            B_jm_minus = get_B(j,m,type="-")
            B_jm1_minus = get_B(j,m1,type="-")
            Lambda_b = get_Lambda(N_in,j,type="b")
            gamma_5 = B_jm_minus*B_jm1_minus*Lambda_b*dicke_bx(N_in,{(j-1,m-1,m1-1):1})
        
        if  (j+1) >= j_min and (j+1) <= j_max and (m-1) >= -(j+1) and (m1-1) >= -(j+1) and (m-1) <= (j+1) and (m1-1)  <= (j+1):
            D_jm_minus = get_D(j,m,type="-")
            D_jm1_minus = get_D(j,m1,type="-")
            Lambda_d = get_Lambda(N_in,j,type="d")
            gamma_6 = D_jm_minus*D_jm1_minus*Lambda_d*dicke_bx(N_in,{(j+1,m-1,m1-1):1})
        
        second_term = (gamma_4+gamma_5+gamma_6)/2
        
        if j >= j_min and j <= j_max and (m+1) <= j and (m1+1) <= j and (m+1) >= -j and (m1+1) >= -j:
            A_jm_plus = get_A(j,m,type="+")
            A_jm1_plus = get_A(j,m1,type="+")
            Lambda_a = get_Lambda(N_in,j,type="a")
            gamma_7 = A_jm_plus*A_jm1_plus*Lambda_a*dicke_bx(N_in,{(j,m+1,m1+1):1})
        
        if (j-1) >= j_min and (j-1) <= j_max and (m+1) >= -(j-1) and (m1+1) >= -(j-1) and (m+1) <= (j-1) and (m1+1) <= (j-1):
            B_jm_plus = get_B(j,m,type="+")
            B_jm1_plus = get_B(j,m1,type="+")
            Lambda_b = get_Lambda(N_in,j,type="b")
            gamma_8 = B_jm_plus*B_jm1_plus*Lambda_b*dicke_bx(N_in,{(j-1,m+1,m1+1):1})
        
        if (j+1) >= j_min and (j+1) <= j_max and  (m+1) >= -(j+1) and (m1+1) >= -(j+1) and (m+1) <= (j+1) and (m1+1) <= (j+1):
            D_jm_plus = get_D(j,m,type="+")
            D_jm1_plus = get_D(j,m1,type="+")
            Lambda_d = get_Lambda(N_in,j,type="d")
            gamma_9 = D_jm_plus*D_jm1_plus*Lambda_d*dicke_bx(N_in,{(j+1,m+1,m1+1):1})

        third_term = (gamma_7+gamma_8+gamma_9)/2
        accumulate_states += 0

    return accumulate_states