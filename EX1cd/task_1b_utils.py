import numpy as np
from scipy.sparse import diags
from tqdm import tqdm


def build_tridiag_matrix(n, s):
    b = (-s)*np.ones(n-1)
    a = (1+2*s)*np.ones(n)
    c = (-s)*np.ones(n-1)
    b[-1] = 2*b[-1] # neumann B.C.
    k = [a,b,c]
    offset = [0,-1,1]
    return diags(k,offset).toarray()

def first_order_tridiags(n, s):
    b = (-s)*np.ones(n)
    a = (1+2*s)*np.ones(n)
    c = (-s)*np.ones(n)
    b[0] = 0
    b[-1] = 2*b[-1]
    c[-1] = 0
    return a,b,c
    
def second_order_tridiags(n, s):
    T = build_tridiag_matrix(n, s)

    b = (s)*np.ones(n-1)
    a = (1-2*s)*np.ones(n)
    c = (s)*np.ones(n-1)
    b[-1] = 2*b[-1]
    k = [a,b,c]
    offset = [0,-1,1]
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
    y = np.zeros((n,))
    C_new = np.zeros((n,))
    #Solver
    alpha[0] = a[0]
    gamma[0] = c[0]/a[0]
    y[0] = C_old[0]/a[0]

    for i in range(1,n):
        alpha[i] = a[i] - b[i]*gamma[i-1]
        gamma[i] = c[i] / alpha[i]
        y[i] = (C_old[i] - b[i]*y[i-1]) / alpha[i]

    C_new[n-1] = y[n-1]
    for i in reversed(range(0,n-1)):
        C_new[i] = y[i] - gamma[i]*C_new[i+1]

    return C_new[:]


def analytical_result(x, t, D, N):
    C = [formula(x[i], t, D, N) for i in range (len(x))]
    return C

def formula(x, t, D, N):
    N = int(N)
    parts = [(((-1)**j) / ((j+(1/2))*np.pi)
    *np.cos((j+1/2)*np.pi*x)
    *np.exp((-1)*(((j+1/2)*np.pi)**2)*D*t)) for j in range(N)]
    C_analytic = 1 - 2*sum(parts)
    return C_analytic

# Compute analytic solutions


def formula_depr(k,iters,place, delta_t, D):

    summand = np.array([(((-1)**j)/((j+1/2)*np.pi)*np.cos(((j+1/2)*np.pi)*place)*np.exp((-1)*(((j+1/2)*np.pi)**2)*k*delta_t*D)) for j in range(iters)]).sum()
    C_analytic = 1-2*summand
    return C_analytic

# Compute analytic solutions
