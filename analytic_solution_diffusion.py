import numpy as np
N = 100        #number of steps
delta_t = 1000  #time step length
delta_x = 0.01
x = np.arange(0,1+delta_x,delta_x)     #domain
D = 10**(-6)

#Initialize
C = np.zeros((N,len(x)))

def formula(k,iters,place):
    sum = 0
    for j in range(iters):
        a = (-1)**j
        b = (j+1/2)*np.pi
        sum += (a/b*np.cos(b*place)*np.exp((-1)*(b**2)*k*delta_t*D))
    C_analytic = 1-2*sum
    return C_analytic

# Compute analytic solutions
for j in range(0,N):
    for i in range (1, len(x)):
        C[j,i] = formula(j,100,x[i])
        print ("done C at" + "[" + str(j) + ","+ str(i) + "]" )
        
   
print (C[0,-1])

## Plot
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

fig = plt.figure()
ax = plt.axes(xlim=(0, x[-1]), ylim=(0, 1))
line, = ax.plot([], [], lw=3)
def init():
    line.set_data([], [])
    return line,

def animate(i):
    fig.suptitle("Diffusion, analytic solution")
    plt.xlabel("Concentration")
    plt.ylabel("Distance from source")
    x_plot = C[i,:]
    y_plot = 1-x
    line.set_data(x_plot, y_plot)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=N, interval=10, blit=True)

anim.save('C_analytic_diffusion.gif')

print (C)