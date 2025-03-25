import numpy as np
import matplotlib.pyplot as plt

def air_density(h):
    if h < 11000:
        T = 15.04 - 0.00649*h
        p = 101.29 * ((T + 273.1)/288.08)**5.256
    elif h >= 11000 and h <= 25000:
        T = -56.46
        p = 22.65 * np.exp(1.73 - .000157 * h)
    else:
        T = -131.21 + .00299 * h
        p = 2.488 * ((T + 273.1)/ 216.6)**-11.388
    return p / (0.2869 * (T + 273.1))

def ambient_temp(h):
    if h < 11000:
        T = 15.04 - 0.00649*h
    elif h >= 11000 and h <= 25000:
        T = -56.46
    else:
        T = -131.21 + .00299 * h
    return (T + 273.1)

def sutton_groves(v, h):
    q_dot = abs(1.83e-4 * ((air_density(h) / R) ** 0.5) * (v ** 3))  # Heat flux (W/m²)
    return q_dot

# Velocity and height data
file_name='Data.txt'
data=np.genfromtxt(file_name,comments='%')
times, velocities, heights = np.loadtxt(file_name, unpack=True)


# Simulation Parameters
nx, ny, nz = 50, 50, 50      # Grid points in x, y and z
dx, dy, dz = 0.001, 0.001, 0.001  # Grid spacing (m)
dt = 0.01            # Time step (s)
time_steps = len(times)  # Number of time steps

# Material & Environment Properties
k = 0.45          # Thermal conductivity (W/mK)
rho = 1050        # Density (kg/m³)
c_p = 3500        # Specific heat capacity (J/kgK)
alpha = k / (rho * c_p)  # Thermal diffusivity (m²/s)
Initial_temp = 283.0

# Sutton-Graves Parameters
R = 0.015          # Nose radius (m)

# Radiation Cooling Parameters
epsilon = 0.8    # Emissivity of surface
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m²K⁴)
T_inf = 300      # Ambient temperature (K)

# Initialize Temperature Field
T = np.full((nx, ny, nz), Initial_temp)


# Apply Sutton-Graves heating to all boundaries
q_dot_0 = sutton_groves(velocities[0], heights[0]) * dx / k
T[0, :, :] += q_dot_0  # Left
T[-1, :, :] += q_dot_0  # Right
T[:, 0, :] += q_dot_0  # Bottom
T[:, -1, :] += q_dot_0  # Top
T[:, :, 0] += q_dot_0 # Front
T[:, :, -1] += q_dot_0 # Back

correction = -q_dot_0
T[0,0,:] += correction
T[0,-1,:] += correction
T[-1,0,:] += correction
T[-1,-1,:] += correction
T[:,0,0] += correction
T[:,0,-1] += correction
T[:,-1,0] += correction
T[:,-1,-1] += correction
T[0,:,0] += correction
T[0,:,-1] += correction
T[-1,:,0] += correction
T[-1,:,-1] += correction

T_central_max = 0

# Time-stepping loop
for t in range(time_steps):
    T_new = T.copy()

    # Finite Difference Scheme (Explicit Method)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1,nz-1):
                T_new[i, j, k] = T[i, j, k] + alpha * dt * (
                    (T[i+1, j, k] - 2*T[i, j, k] + T[i-1, j, k]) / dx**2 +
                    (T[i, j+1, k] - 2*T[i, j, k] + T[i, j-1, k]) / dy**2 +
                    (T[i, j, k+1] - 2*T[i, j, k] + T[i, j, k-1]) / dz**2
                )

    
    
    if T[nx//2, ny//2, nz//2] > T_central_max:
        T_central_max = T[nx//2, ny//2, nz//2]
        step = t
        
    # Apply radiative cooling and heating
    rad_cooling = (epsilon * sigma * (T[:,0,0]**4 - ambient_temp(heights[t])**4) * dt) / (rho * c_p) # Similar on all edges
    q_dot_t = sutton_groves(velocities[t], heights[t]) * dx / k
    heat_transfer = rad_cooling+q_dot_t
    T_new[0, :, :] += heat_transfer  # Left
    T_new[-1, :, :] += heat_transfer  # Right
    T_new[:, 0, :] += heat_transfer  # Bottom
    T_new[:, -1, :] += heat_transfer  # Top
    T_new[:, :, 0] += heat_transfer # Front
    T_new[:, :, -1] += heat_transfer # Back
    
    # So that edges do not double
    correction = rad_cooling[0]-q_dot_t
    T[0,0,:] += correction
    T[0,-1,:] += correction
    T[-1,0,:] += correction
    T[-1,-1,:] += correction
    T[:,0,0] += correction
    T[:,0,-1] += correction
    T[:,-1,0] += correction
    T[:,-1,-1] += correction
    T[0,:,0] += correction
    T[0,:,-1] += correction
    T[-1,:,0] += correction
    T[-1,:,-1] += correction

    # Update temperature field
    T = T_new.copy()

# Plot Results
print(f"Max central temperature: {T_central_max}")
print(f"Step: {step}")
plt.figure(figsize=(6, 5))
plt.imshow(T[:,:,nz//2], cmap='hot', origin='lower', extent=[0, nx*dx, 0, ny*dy])
plt.colorbar(label='Temperature (K)')
plt.title('2D Steak cross-section')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()
