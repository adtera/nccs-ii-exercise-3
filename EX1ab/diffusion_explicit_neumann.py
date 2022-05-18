# imports n stuff
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# Solve 1D Diffusion Equation explicitly for t timesteps and Nx gridpoints 

# Diffusion coefficient
D =  10 ** (-6)

# Length of domain
h = 1

# Duration of Diffusion
time = 500000


# Space discretizaton
delta_x = 10 ** (-3)

# Number of Gridpoints 
Nx = int(pow(delta_x,-1))

# Space grid
x_grid = np.linspace(0,h,Nx+1)


# Time discretication
delta_t =  10 ** (-2) * 49



# Combine all constants
d = (delta_t * D) / (delta_x ** 2)
print(f"d = {d}")

# C at timestep n 
C_nprev = np.zeros(Nx+1)
# C at timestep n-1
C_n = np.zeros(Nx+1)

for n in range(time):
    for i in range(1,Nx):
        C_n[i] = C_nprev[i] + d * (C_nprev[i-1] - 2 * C_nprev[i] + C_nprev[i+1])
    
    # Von Neumann BC with dC/dx = 0 at x = Nx requires to solve different equation
    C_n[Nx] = d * (2 * C_nprev[Nx-1]) + (1 - 2 * d) * C_nprev[Nx]
    # Dirichlet BC
    C_n[0] = 1
    

    # Update C_n 
    C_nprev, C_n = C_n, C_nprev

    #print(C_nprev)
plt.plot(C_nprev, x_grid)
plt.title(f"Explicit Solution,"
    f"with Dirichtlet/Neumann BC, d = {round(d,2)}," 
    f"dx = {round(delta_x,4)}; dt = {round(delta_t,4)}; time = {time}")
plt.xlabel('Concentration')
plt.ylabel('Distance from source')
plt.savefig(f'Explicit_Diffusion_n{d}.png')
plt.show()




