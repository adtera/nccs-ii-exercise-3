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
time = 100000



# Space discretizaton
delta_x = 10 ** (-3)

# Number of Gridpoints 
Nx = int(pow(delta_x,-1))

# Space grid
x_grid = np.linspace(0,h,Nx+1)


# Time discretication
delta_t =  10 ** (-2) * 49

# Number of timesteps s
#Nt =  time * delta_t

# Time Grid
#t_grid = np.linspace(0,h,Nt+1)


# Combine all constants
d = (delta_t * D) / (delta_x ** 2)
print(f"d = {d}")


# Make calculations for Dirichlet/Dirichlet 
# C at timestep n 
DC_nprev = np.zeros(Nx+1)
# C at timestep n-1
DC_n = np.zeros(Nx+1)

# Make calculations for Neumann/Dirichlet
# C at timestep n 
NC_nprev = np.zeros(Nx+1)
# C at timestep n-1
NC_n = np.zeros(Nx+1)

for n in range(time):
    for i in range(1,Nx):
        # Dir/Dir
        DC_n[i] = d * DC_nprev[i-1] + (1-2*d) * DC_nprev[i] + d * DC_nprev[i+1]
        # Dir/Neumann
        NC_n[i] = d * NC_nprev[i-1] + (1-2 * d) * NC_nprev[i] + d * NC_nprev[i+1]

    # Dirichlet BC
    DC_n[0],DC_n[Nx] = 1,0

    # Update DC_n 
    DC_nprev, DC_n = DC_n, DC_nprev

    # Von Neumann BC with dC/dx = 0 at x = Nx requires to solve different equation
    NC_n[Nx] = d * (2 * NC_nprev[Nx-1]) + (1 - 2 * d) * NC_nprev[Nx] 
    print(NC_n[Nx])
    # Dirichlet BC
    NC_n[0] = 1
    
    # Update NC_n 
    NC_nprev, NC_n = NC_n, NC_nprev




plt.plot(DC_nprev, x_grid,label="Dir/Dir")
plt.plot(NC_nprev,x_grid,label="Neu/Dir")
plt.legend()
plt.title(f"Explicit Solution"
    f" BC, d = {round(d,2)}," 
    f"dx = {round(delta_x,4)}; dt = {round(delta_t,4)}; time = {time}")
plt.xlabel('Concentration')
plt.ylabel('Distance from source')
plt.savefig(f'Explicit_Diffusion_tinf_d{d}.png')
plt.show()




