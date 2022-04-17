from re import M
import numpy as np
cimport numpy as np
from numpy cimport ndarray 
cimport cython
np.import_array()
from scipy.sparse import csc_matrix

@cython.exceptval(-2, check=True)
@cython.boundscheck(False)
cdef get_A(double j,double m,str type=""):
    if type == "+":
        return np.sqrt((j-m)*(j+m+1))
    elif type == "-":
        return np.sqrt((j+m)*(j-m+1))
    else:
        return m

@cython.exceptval(-2, check=True)
@cython.boundscheck(False)
cdef get_B(double j,double m,str type=""):
    if type == "+":
        return np.sqrt((j-m)*(j-m-1))
    elif type == "-":
        return -np.sqrt((j+m)*(j+m-1))
    else:
        return np.sqrt((j+m)*(j-m))

@cython.exceptval(-2, check=True)
@cython.boundscheck(False)
cdef get_D(double j,double m,str type=""):
    if type == "+":
        return -np.sqrt((j+m+1)*(j+m+2))
    if type == "-":
        return np.sqrt((j-m+1)*(j-m+2))
    else:
        return (j+m+1)*(j-m+1)

@cython.exceptval(-2, check=True)
@cython.boundscheck(False)
cdef get_Lambda(int N, double j,str type=""):
    if type == "a":
        return (N/2+1)/(2*j*(j+1))
    elif type == "b":
        return (N/2+j+1)/(2*j*(2*j+1))
    else:
        return (N/2-j)/(2*(j+1)*(2*j+1))

@cython.exceptval(-2, check=True)
@cython.boundscheck(False)
cdef dicke_bx(int N,dict jmm1,ik,dim):
    # create a dicke basis follow jmm1
    # jmm1 as {(j,m,m1):p}
    cdef int i,k 
    cdef np.ndarray[np.complex128_t, ndim=2] rho = np.zeros((dim,dim),dtype = np.complex128)
    for key in jmm1:
        i,k = ik[key]
        rho[i,k] = jmm1[key]
    return csc_matrix(rho)

@cython.exceptval(-2, check=True)
@cython.boundscheck(False)
def calc_rho_0(iks,jmm1,state,all_iks,j_min,j_max,N_in,d_dicke):

    accumulate_states = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
    cdef int i,k 
    cdef double j,m,m1,Lambda_a,Lambda_b,B_jm_z,B_jm1_z,D_jm_z,D_jm1_z,Lambda_d,A_jm_plus,A_jm1_minus,B_jm_minus,B_jm1_minus,D_jm_plus,D_jm1_plus
    cdef complex p_jmm1 
    
    for ik in iks:
        j,m,m1 = jmm1[ik]
        i,k = ik
        first_term = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
        second_term = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
        third_term = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
        gamma_1 = csc_matrix((d_dicke, d_dicke), dtype=np.complex128) 
        gamma_2 = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
        gamma_3 = csc_matrix((d_dicke, d_dicke), dtype=np.complex128) 
        gamma_4 = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
        gamma_5 = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
        gamma_6 = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
        gamma_7 = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
        gamma_8 = csc_matrix((d_dicke, d_dicke), dtype=np.complex128) 
        gamma_9 = csc_matrix((d_dicke, d_dicke), dtype=np.complex128)
        p_jmm1 = state[i,k]

        if j >= j_min and j <= j_max and m > -j and m1 > -j and m < j and m1 < j:
            Lambda_a = get_Lambda(N_in,j,type="a")
            gamma_1 = m*m1*Lambda_a*dicke_bx(N_in,{(j,m,m1):1},all_iks,d_dicke)

        if (j-1) >= j_min and (j-1) <= j_max and (m >= -(j-1) and m1 >= -(j-1)) and (m <= (j-1) and m1 <= (j-1)):
            B_jm_z = get_B(j,m,type="z")
            B_jm1_z = get_B(j,m1,type="z")
            Lambda_b = get_Lambda(N_in,j,type="b")
            gamma_2 = B_jm_z*B_jm1_z*Lambda_b*dicke_bx(N_in,{(j-1,m,m1):1},all_iks,d_dicke) 

        if (j+1) <= j_max and (j+1) >= j_min and (m >= -(j+1) and m1 >= -(j+1)) and (m <= (j+1) and m1 <= (j+1)):
            D_jm_z = get_D(j,m,type="z")
            D_jm1_z = get_D(j,m1,type="z")
            Lambda_d = get_Lambda(N_in,j,type="d")
            gamma_3 = D_jm_z*D_jm1_z*Lambda_d*dicke_bx(N_in,{(j+1,m,m1):1},all_iks,d_dicke)
        
        if j >= j_min and j <= j_max  and (m-1) >= -j and (m1-1) >= -j and (m-1) <= j and (m1-1) <= j:
            A_jm_minus = get_A(j,m,type="-")
            A_jm1_minus = get_A(j,m1,type="-")
            Lambda_a = get_Lambda(N_in,j,type="a")
            gamma_4 = A_jm_minus*A_jm1_minus*Lambda_a*dicke_bx(N_in,{(j,m-1,m1-1):1},all_iks,d_dicke)
        
        if (j-1) >= j_min and (j-1) <= j_max and (m-1) >= -(j-1) and (m1-1) >= -(j-1) and (m-1) <= (j-1) and  (m1-1) <= (j-1):
            B_jm_minus = get_B(j,m,type="-")
            B_jm1_minus = get_B(j,m1,type="-")
            Lambda_b = get_Lambda(N_in,j,type="b")
            gamma_5 = B_jm_minus*B_jm1_minus*Lambda_b*dicke_bx(N_in,{(j-1,m-1,m1-1):1},all_iks,d_dicke)
        
        if  (j+1) >= j_min and (j+1) <= j_max and (m-1) >= -(j+1) and (m1-1) >= -(j+1) and (m-1) <= (j+1) and (m1-1)  <= (j+1):
            D_jm_minus = get_D(j,m,type="-")
            D_jm1_minus = get_D(j,m1,type="-")
            Lambda_d = get_Lambda(N_in,j,type="d")
            gamma_6 = D_jm_minus*D_jm1_minus*Lambda_d*dicke_bx(N_in,{(j+1,m-1,m1-1):1},all_iks,d_dicke)
        
        if j >= j_min and j <= j_max and (m+1) <= j and (m1+1) <= j and (m+1) >= -j and (m1+1) >= -j:
            A_jm_plus = get_A(j,m,type="+")
            A_jm1_plus = get_A(j,m1,type="+")
            Lambda_a = get_Lambda(N_in,j,type="a")
            gamma_7 = A_jm_plus*A_jm1_plus*Lambda_a*dicke_bx(N_in,{(j,m+1,m1+1):1},all_iks,d_dicke)
        
        if (j-1) >= j_min and (j-1) <= j_max and (m+1) >= -(j-1) and (m1+1) >= -(j-1) and (m+1) <= (j-1) and (m1+1) <= (j-1):
            B_jm_plus = get_B(j,m,type="+")
            B_jm1_plus = get_B(j,m1,type="+")
            Lambda_b = get_Lambda(N_in,j,type="b")
            gamma_8 = B_jm_plus*B_jm1_plus*Lambda_b*dicke_bx(N_in,{(j-1,m+1,m1+1):1},all_iks,d_dicke)
        
        if (j+1) >= j_min and (j+1) <= j_max and  (m+1) >= -(j+1) and (m1+1) >= -(j+1) and (m+1) <= (j+1) and (m1+1) <= (j+1):
            D_jm_plus = get_D(j,m,type="+")
            D_jm1_plus = get_D(j,m1,type="+")
            Lambda_d = get_Lambda(N_in,j,type="d")
            gamma_9 = D_jm_plus*D_jm1_plus*Lambda_d*dicke_bx(N_in,{(j+1,m+1,m1+1):1},all_iks,d_dicke)

        accumulate_states += p_jmm1*((gamma_1+gamma_2+gamma_3)+(gamma_4+gamma_5+gamma_6)/2+(gamma_7+gamma_8+gamma_9)/2)

    return accumulate_states