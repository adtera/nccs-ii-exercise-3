#%%
from scipy.sparse import diags
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

from task_1b_utils import build_tridiag, build_tridiag_with_inverse, thomas_algorithm

t = 2000000 # timesteps

n = 5
D = 10**(-6)
a,b,c = build_tridiag_with_inverse(n,D,dt=1, dx = 0.02)


C = np.zeros((n,t))
C[0,:] = 1
for _t in tqdm(range(1,t)):
#    print(f" C[:,_t-1] : { C[:,_t-1]}")
    C[:,_t] = thomas_algorithm(n, a, b, c, C[:,_t-1])
    C[0,_t] = 1

#np.delete(C, slice(None, None, 100000))

#%%

ax = sns.heatmap(C[:,:2000], cmap = 'Greens').set(title='second order')


# %%

fig = plt.figure()
sns.heatmap(C[:,0], cmap = 'Greens')

#%%

def init():
    sns.heatmap(C[:,0])

def animate(i):
    sns.heatmap(C[:,i], cbar=False)

anim = FuncAnimation(fig, animate, init_func=init, frames=20, repeat=False)

plt.show()
# %%
