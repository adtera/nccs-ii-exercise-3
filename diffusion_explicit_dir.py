# imports n stuff
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# Solve 1D Diffusion Equation explicitly for t timesteps and Nx gridpoints 

# Number of timesteps 
Nt =  2 * (10 ** 3)

# Number of Gridpoints 
Nx = 2 * (10 ** 2)

# Diffusion coefficient
D =  10 ** (-6)

# Length of domain
h = 1

# Duration of Diffusion
time = 10000


# Space grid
x_grid = np.linspace(0,h,Nx+1)

# Time Grid
t_grid = np.linspace(0,h,Nt+1)

# Space discretizaton
delta_x = h/Nx


# Time discretication
delta_t = time/Nt

# Combine all constants
d = delta_t * D / (delta_x ** 2)
print(f"d = {d}")

# C at timestep n 
C_nprev = np.zeros(Nx+1)
# C at timestep n-1
C_n = np.zeros(Nx+1)

for n in range(Nt):
    C_n[1:Nx] = d * C_nprev[0:Nx-1] + (1-2*d) * C_nprev[1:Nx] + d * C_nprev[2:Nx+1]
    
    # Dirichlet BC
    C_n[0],C_n[Nx] = 1,0

    # Update C_n 
    C_nprev, C_n = C_n, C_nprev

    print(C_nprev)
plt.plot(C_nprev, x_grid)
plt.title(f"Explicit Solution for 1D Diffusion Problem, with Dirichtlet/Dirichlet BC and d = {d} ")
plt.xlabel('Concentration')
plt.ylabel('Distance from source')
plt.show()




