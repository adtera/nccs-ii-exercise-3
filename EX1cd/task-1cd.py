#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

from task_1b_utils import analytical_result, first_order_tridiags, thomas_algorithm, second_order_tridiags

N_t = 50000 # timesteps
N_x = 11
dt = 0.01
dx = 1/(N_x - 1)
D = 10**(-6)
x = np.arange(0,1+dx,dx)
t = np.arange(0,N_t * dt,dt)

#Task 1c
C = np.zeros((N_t,N_x))
C[0,0] = 1 # B.C.
s = dt/(dx**2)
s *= D
a,b,c = first_order_tridiags(N_x,s)

#Task 1d
C_inv = np.zeros((N_t,N_x))
C_inv[0,0] = 1
s_inv = (dt/(dx**2))/2
s_inv *= D
a_inv,b_inv,c_inv = second_order_tridiags(N_x,s_inv)

C_an = np.zeros((N_t,N_x))
C_an[0,:] = np.flip(analytical_result(x, 0, D, 1000))


for _t in tqdm(range(1,N_t)):
    C[_t,:] = thomas_algorithm(N_x, a, b, c, C[_t-1,:])
    C[_t,0] = 1

    C_inv[_t,:] = thomas_algorithm(N_x, a_inv, b_inv, c_inv, C_inv[_t-1,:])
    C_inv[_t,0] = 1

    C_an[_t,:] = np.flip(analytical_result(x, _t*dt, D, 1000))
 

diff = np.abs(C - C_an)
diff_avg = np.mean(diff, axis = 1)

diff_inv = np.abs(C_inv - C_an)
diff_inv_avg = np.mean(diff_inv, axis = 1)


#%%
fig, ax = plt.subplots(figsize=(12,5))
sns.lineplot(y=x,x=C_an[-1,:], lw=1)
sns.scatterplot(y=x,x=C[-1,:], marker = '1', color = 'black', s = 100)
sns.scatterplot(y=x,x=C_inv[-1,:], marker = '2', color = 'red', s = 100)
plt.legend(loc='upper left', labels=['Analytic', 'Implicit 1st order', 'Implicit 2nd order'])
plt.title('Implicit Method for 10000 0.002 timesteps (adim)')
plt.savefig('Implicit_Method_10000')
#%%
fig, ax = plt.subplots(figsize=(12,5))

sns.lineplot(y=x,x=C_an[6000,:], lw=1)
sns.scatterplot(y=x,x=C[6000,:], marker = '1', color = 'black', s = 100)
sns.scatterplot(y=x,x=C_inv[6000,:], marker = '2', color = 'red', s = 100)
plt.legend(loc='upper left', labels=['Analytic', 'Implicit 1st order', 'Implicit 2nd order'])
plt.title('Implicit Method for 6000 0.002 timesteps (adim)')
plt.savefig('Implicit_Method_6000')

#%%
fig, ax = plt.subplots(figsize=(12,5))
sns.scatterplot(y=x,x=diff[-1,:], marker='1', s=100)
sns.scatterplot(y=x,x=diff_inv[-1,:], marker='2', s=100)
plt.legend(loc='upper left', labels=['1st order', '2nd order'])
plt.title('Absolute Difference between Discrete and Analytic Solution of Implicit Method for 9000 0.2s timesteps')
plt.show

#%%
fig, ax = plt.subplots(figsize=(12,5))
sns.scatterplot(y=x,x=diff[600,:], marker='1', s=100)
sns.scatterplot(y=x,x=diff_inv[600,:], marker='2', s=100)
plt.legend(loc='upper left', labels=['1st order', '2nd order'])
plt.title('Absolute Difference between Discrete and Analytic Solution of Implicit Method for 6000 0.2s timesteps')
plt.show

#%%
fig, ax = plt.subplots(figsize=(12,5))
sns.scatterplot(x=t,y=diff_avg[:], marker='1', s=50)
sns.scatterplot(x=t,y=diff_inv_avg[:], marker='2', s=50)
plt.legend(loc='upper left', labels=['1st order', '2nd order'])
plt.title('Absolute Difference between Discrete and Analytic Solution of Implicit Method for 6000 0.2s timesteps')
plt.show
#%%
ax = sns.heatmap(C[:,:], cmap = 'Greens').set(title='first order')
# %%

# %%

fig = plt.figure()
ax = plt.axes(xlim=(0, x[-1]), ylim=(0, 1))
line, = ax.plot([], [], lw=3)
def init():
    line.set_data([], [])
    return line,

def animate(i):
    fig.suptitle("Concentration over time")
    plt.xlabel("x")
    plt.ylabel("Concetration")
    x_plot = x
    y_plot = C[i,:]
    line.set_data(x_plot, y_plot)
    return line,

anim = FuncAnimation(fig, animate, init_func=init, frames=20000, blit=True)

anim.save('C.gif')
# %%
plt.show
# %%
