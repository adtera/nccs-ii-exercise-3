import numpy as np
from scipy.sparse import diags

def build_tridiag_matrix(n, D, dt=1, dx=1):
    s = (dt/(dx**2))/2
    s *= D
    b = (-s)*np.ones(n-1)
    a = (1+2*s)*np.ones(n)
    c = (-s)*np.ones(n-1)
    b[-1] = 2 * b[-1]
    k = [b,a,c]
    offset = [-1,0,1]
    return diags(k,offset).toarray()

def build_tridiag(n, D, dt=1, dx=1):
    s = dt/(dx**2)
    s *= D
    b = (-s)*np.ones(n)
    a = (1+2*s)*np.ones(n)
    c = (-s)*np.ones(n)
    b[0] = 0
    c[-1] = 0
    return a,b,c
    
def build_tridiag_with_inverse(n, D, dt=1, dx=1):
    s = (dt/(dx**2))/2
    s *= D

    T = build_tridiag_matrix(n, D, dt, dx)

    k = [(s)*np.ones(n-1),(1-2*s)*np.ones(n),(s)*np.ones(n-1)]
    offset = [-1,0,1]
    diag = diags(k,offset).toarray()
    inverse = np.linalg.inv(diag)

    T_bar = np.matmul(inverse, T)

    a_inverse = np.diag(T_bar,0)
    b_inverse = np.diag(T_bar,-1)
    c_inverse = np.diag(T_bar,1)

    b_inverse = np.insert(b_inverse,0,0)
    c_inverse = np.append(c_inverse,0)

    return a_inverse, b_inverse, c_inverse


def thomas_algorithm(n, a, b, c, C_old):
    alpha = np.zeros((n,))
    gamma = np.zeros((n,))
    v = np.zeros((n,))
    C_new = np.zeros((n,))

    #Solver
    alpha[0] = a[0]
    gamma[0] = c[0]/(a[0])
    v[0] = C_old[0]/(a[0])

    for i in range(1,n):
        alpha[i] = a[i]-b[i]*gamma[i-1]
        gamma[i] = c[i]/alpha[i]
        v[i] = (C_old[i]-b[i]*v[i-1])/(alpha[i])
#    print(f"v[:]: {v[:]}")

    C_new[n-1] = v[n-1]
    
    for i in range(0,n-1):
        C_new[i]=v[i]-gamma[i]*C_new[i+1]

#    print(f"C_new[:]: {C_new[:]}")
    return C_new[:]
