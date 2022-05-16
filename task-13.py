#%%
from scipy.sparse import diags
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.animation import FuncAnimation



def build_tridiag_matrix(n, D, dt=1, dx=1):
    s = dt/(dx**2)
    s *= D
    k = [(-s)*np.ones(n-1),(1+2*s)*np.ones(n),(-s)*np.ones(n-1)]
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




t = 2000000 # timesteps

n = 10
D = 10**(-6)
#T = build_tridiag_matrix(n, D)
a,b,c = build_tridiag(n,D,dt=1, dx = 0.02)
b[-1] = 2 * b[-1] # neumann B.C.


C = np.zeros((n,t))

C[0,:] = 1


for _t in tqdm(range(1,t)):
#    print(f" C[:,_t-1] : { C[:,_t-1]}")
    C[:,_t] = thomas_algorithm(n, a, b, c, C[:,_t-1])
    C[0,_t] = 1

#np.delete(C, slice(None, None, 100000))

#%%
ax = sns.heatmap(C[:,:2000], cmap = 'Greens')

# %%

fig = plt.figure()
sns.heatmap(C[:,0], cmap = 'Greens')

def init():
    sns.heatmap(C[:,0])

def animate(i):
    sns.heatmap(C[:,i], cbar=False)

anim = FuncAnimation(fig, animate, init_func=init, frames=20, repeat=False)

plt.show()
# %%
