import numpy as np
import argparse

## Input arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('ts', nargs='?', const=1, default=0.01)
    parser.add_argument('iters', nargs='?', const=1, default=300)
    return parser.parse_args()

args = parse_arguments()

## Time step length and iterations
delta_t = float(args.ts)
N = int(args.iters)


## Space discretization 
x = np.arange(0,2,0.01)

##Given data
U = 1
delta_x = 0.01
C1 = 1 #0.1<x<0.3
C2 = np.exp(-10*(4*x-1)**2)

## Initial conditions
C_SqP = np.zeros((N,len(x)))
C_Gauss = np.zeros((N,len(x)))
C_Gauss_exact = np.zeros((N,len(x)))
C_SqP_exact = np.zeros((N,len(x)))

# Square impuls
for i in range(len(x)):
    if x[i] >0.1 and x[i] <0.3:
        C_SqP[0,i] = C1
        C_SqP_exact[0,i] = C1

# Gaussian initial condition
C_Gauss[0,:] = C2
C_Gauss_exact[0,:] = C2


## SOLVE:
# dC_i/dt = U*dC_i/dx
# C_i_n+1 = C_i_n * (1-Co)*C_i-1_n

# Courant number
Co = U*delta_t /delta_x

# Iteration
for j in range(0,N-1):
    for i in range (1, len(x)):
        C_SqP[j+1,i] = C_SqP[j,i] * (1-Co) + Co * C_SqP[j,i-1]
        C_Gauss[j+1,i] = C_Gauss[j,i] * (1-Co) + Co * C_Gauss[j,i-1]

## Analytical solution
for j in range(0,N-1):
    for i in range(len(x)):
        # Square impuls
        if (x[i] - U * (j+1) * delta_t) >0.1 and (x[i]-U * (j+1) * delta_t) <0.3:
            C_SqP_exact[j+1,i] = C1
        # Gaussian
        C_Gauss_exact[j+1,i] = np.exp(-10*(4*(x[i]-U*delta_t*(j+1))-1)**2)


## Plot
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

fig = plt.figure()
ax = plt.axes(xlim=(0, x[-1]), ylim=(-1, 2))
line, = ax.plot([], [], lw=3)
def init():
    line.set_data([], [])
    return line,

def animate(i):
    fig.suptitle("Advection with initial square impuls and Co = " + str(Co))
    plt.xlabel("x")
    plt.ylabel("Wave amplitude")
    x_plot = x
    y_plot = C_SqP[i,:]
    y_plot_exact = C_SqP_exact[i,:]
    line.set_data(x_plot, y_plot)
    return line,

def animate2(i):
    fig.suptitle("Advection with initial Gaussian impuls and Co = " + str(Co))
    plt.xlabel("x")
    plt.ylabel("Wave amplitude")
    x_plot = x
    y_plot = C_Gauss[i,:]
    line.set_data(x_plot, y_plot)
    return line,

def animate_exact(i):
    fig.suptitle("Advection with initial square impuls, analytical solution")
    plt.xlabel("x")
    plt.ylabel("Wave amplitude")
    x_plot = x
    y_plot_exact = C_SqP_exact[i,:]
    line.set_data(x_plot, y_plot_exact)
    return line,

def animate_exact2(i):
    fig.suptitle("Advection with initial Gaussian impuls, analytical solution")
    plt.xlabel("x")
    plt.ylabel("Wave amplitude")
    x_plot = x
    y_plot_exact = C_Gauss_exact[i,:]
    line.set_data(x_plot, y_plot_exact)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=N, interval=20, blit=True)
anim2 = FuncAnimation(fig, animate2, init_func=init,
                               frames=N, interval=20, blit=True)
anim_exact = FuncAnimation(fig, animate_exact, init_func=init,
                               frames=N, interval=20, blit=True)
anim_exact2 = FuncAnimation(fig, animate_exact2, init_func=init,
                               frames=N, interval=20, blit=True)


## Save plots
anim.save('C_SqP_Co=' + str(Co)+ '.gif')
anim2.save('C_Gauss_Co=' + str(Co)+ '.gif')
anim_exact.save('C_SqP_exact.gif')
anim_exact2.save('C_Gauss_exact.gif')