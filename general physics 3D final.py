import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

# Velocity and height data
file_name="Data.txt"
data=np.genfromtxt(file_name,comments="%")
times, velocities, heights = np.loadtxt(file_name, unpack=True)


# Simulation Parameters
nx, ny, nz = 25, 50, 50      # Grid points in x, y and z
dx, dy, dz = 0.001, 0.001, 0.001  # Grid spacing (m)
dt = 0.05            # Time step (s)
time_steps = len(times)  # Number of time steps

# Material & Environment Properties
k = 0.45          # Thermal conductivity (W/mK)
rho = 1050        # Density (kg/m³)
c_p = 3500        # Specific heat capacity (J/kgK)
alpha = k / (rho * c_p)  # Thermal diffusivity (m²/s)
Initial_temp = 283.0  # Temperature (K)
R_steak = 8.314
M_steak = 0.02897
kappa = 1000

# Radiation Cooling Parameters
epsilon = 0.95    # Emissivity of surface
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m²K⁴)

# Atmospheric density model
R_earth = 6371000  # Earth's radius in meters
M_earth = 5.972e24  # Earth's mass (kg)
R_air = 287.0  # Specific gas constant for dry air (J/kg·K)
m_air = 4.8e-26  # Approximate mass of air molecule (kg)
gamma = 1.4 # Ratio of specific heat for air

def gravity(h):
    return sc.constants.G * M_earth / (R_earth + h) ** 2

def air_density(h):
    """Computes air density (kg/m³) at altitude h (meters) using NASA model."""
    G = gravity(h)

    if h < 11000:
        Temp = 288.15 - 0.0065 * h
        P = 101325 * (Temp / 288.15) ** 5.256

    elif h < 20000:
        Temp = 216.65
        P = 22631 * np.exp(-0.000157 * (h - 11000))

    elif h < 32000:
        Temp = 216.65 + 0.001 * (h - 20000)
        P = 5509 * (216.65 / Temp) ** 34.16

    elif h < 47000:
        Temp = 228.65 + 0.0028 * (h - 32000)
        P = 873.58 * (228.65 / Temp) ** 12.2

    elif h < 51000:
        Temp = 270.65
        P = 111.64 * np.exp(-0.000126 * (h - 47000))

    elif h < 71000:
        Temp = 270.65 - 0.0028 * (h - 51000)
        P = 67.443 * (270.65 / Temp) ** -12.2

    elif h < 86000:
        Temp = 214.65 - 0.002 * (h - 71000)
        P = 3.987 * (214.65 / Temp) ** -17.08

    elif h < 91000:
        Temp = 186.8673
        P = 0.374 * np.exp(-G * (h - 86000) / (R_air * Temp))

    elif h < 110000:
        Temp = 263.1905 - 76.32321 * (np.sqrt(1 - ((h - 91000) / 19942.9)**2))
        P = 0.374 * np.exp(-G * (h - 86000) / (R_air * Temp))

    elif h < 120000:
        Temp = 240 + 0.012 * (h - 110000)
        P = 0.374 * np.exp(-G * (h - 86000) / (R_air * Temp))

    else:
        Temp = 1000 - 640 * np.exp(-0.00001875 * (h - 120000) * ((R_earth + 120000) / (R_earth + h)))
        P = 0.374 * np.exp(-G * (h - 86000) / (R_air * Temp))

    # Compute air density using the ideal gas law
    return  P / (R_air * Temp)

def ambient_temp(h):
    """Computes ambient temperature (K) at altitude h (meters) using NASA model."""
    
    if h < 11000:  # Troposphere (0 - 11 km)
        Temp = 288.15 - 0.0065 * h

    elif h < 20000:  # Tropopause (11 - 20 km)
        Temp = 216.65

    elif h < 32000:  # Stratosphere Lower (20 - 32 km)
        Temp = 216.65 + 0.001 * (h - 20000)

    elif h < 47000:  # Stratosphere Upper (32 - 47 km)
        Temp = 228.65 + 0.0028 * (h - 32000)

    elif h < 51000:  # Stratopause (47 - 51 km)
        Temp = 270.65

    elif h < 71000:  # Mesosphere Lower (51 - 71 km)
        Temp = 270.65 - 0.0028 * (h - 51000)

    elif h < 86000:  # Mesosphere Upper (71 - 86 km)
        Temp = 214.65 - 0.002 * (h - 71000)

    elif h < 91000:  # Thermosphere (86 - 91 km)
        Temp = 186.8673

    elif h < 110000:  # Thermosphere (91 - 110 km)
        Temp = 263.1905 - 76.32321 * (np.sqrt(1 - ((h - 91000) / 19942.9)**2))

    elif h < 120000:  # Thermosphere (110 - 120 km)
        Temp = 240 + 0.012 * (h - 110000)

    else:  # Exosphere (120 - 1000 km)
        RE = 6371000  # Earth's radius in meters
        Temp = 1000 - 640 * np.exp(-0.00001875 * (h - 120000) * ((RE + 120000) / (RE + h)))

    return Temp

def heat_coefficient(v, h):
    rho_air = air_density(h)
    return 6.7*rho_air**(4/5)*abs(v)**(4/5)

def heat(T, v, h):
    kappa = heat_coefficient(v, h)
    q = kappa*(ambient_temp(h)*(1+(v/(np.sqrt(gamma*R_steak*ambient_temp(h)/M_steak)))**2*(gamma-1)/2)-T)
    return q

def rad_cooling(T, h):
    q = -epsilon*sigma*(T**4-ambient_temp(h)**4)
    return q

# Initialize Temperature Field
T = np.full((nx, ny, nz), Initial_temp)

# Apply boundary conditions. Specific indices so that edges are not double counted.
# Apply heating to only one face.
q_dot_0 = heat(T[0,:,:], velocities[0], heights[0]) * dt / (dz*rho * c_p)
T[0, :, :] += q_dot_0 +(rad_cooling(T[0,:,:], heights[0]) * dt) / (dz*rho * c_p)  # Left
T[-1, :, :] += (rad_cooling(T[-1,:,:], heights[0]) * dt) / (dz*rho * c_p)  # Right
T[1:-1, 0, :] += (rad_cooling(T[1:-1,0,:], heights[0]) * dt) / (dz*rho * c_p)  # Bottom
T[1:-1, -1, :] += (rad_cooling(T[1:-1, -1,:], heights[0]) * dt) / (dz*rho * c_p)  # Top
T[1:-1, 1:-1, 0] += (rad_cooling(T[1:-1,1:-1,0], heights[0]) * dt) / (dz*rho * c_p) # Front
T[1:-1, 1:-1, -1] += (rad_cooling(T[1:-1,1:-1,-1], heights[0]) * dt) / (dz*rho * c_p) # Back

T_max_arr = T
counter = 0

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
    
    # Find at what step max internal temp is reached
    if T_new[nx//2, ny//2, nz//2] > T_max_arr[nx//2, ny//2, nz//2]:
        step = t
    
    # Update max temperature array
    np.maximum(T, T_max_arr, out=T_max_arr)
    
    # Quarter through the fall. Plots in all three orientations
    if t==time_steps//4: 
        T_quarter = T_new.copy()
        print("25%")
        plt.figure(figsize=(6, 5))
        plt.imshow(T_quarter[:,:,nz//2], cmap="hot", origin="lower", extent=[0,ny*dy,0,nx*dx])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (25%)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
        plt.figure(figsize=(6, 5))
        plt.imshow(T_quarter[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (25%)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
        plt.figure(figsize=(6, 5))
        plt.imshow(T_quarter[:,ny//2,:], cmap="hot", origin="lower", extent=[0,nz*dz,0,nx*dx])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (25%)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
     
    # halfway through the fall
    if t==time_steps//2: 
        T_half = T_new.copy()
        print("50%")
        plt.figure(figsize=(6, 5))
        plt.imshow(T_half[:,:,nz//2], cmap="hot", origin="lower", extent=[0,ny*dy,0,nx*dx])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (50%)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
        plt.figure(figsize=(6, 5))
        plt.imshow(T_half[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (50%)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
        plt.figure(figsize=(6, 5))
        plt.imshow(T_half[:,ny//2,:], cmap="hot", origin="lower", extent=[0,nz*dz,0,nx*dx])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (50%)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
    
    # 75%
    if t==3*time_steps//4: 
        T_3_quarter = T_new.copy()
        print("75%")
        plt.figure(figsize=(6, 5))
        plt.imshow(T_3_quarter[:,:,nz//2], cmap="hot", origin="lower", extent=[0,ny*dy,0,nx*dx])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (75%)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
        plt.figure(figsize=(6, 5))
        plt.imshow(T_3_quarter[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (75%)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
        plt.figure(figsize=(6, 5))
        plt.imshow(T_3_quarter[:,ny//2,:], cmap="hot", origin="lower", extent=[0,nz*dz,0,nx*dx])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (75%)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
    
    
    # Rare
    if counter==0 and T[nx//2, ny//2, nz//2]>323:
        print("Rare")
        print(f"Step: {t}, Height: {heights[t]}")
        counter+=1
        plt.figure(figsize=(6, 5))
        plt.imshow(T[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (Rare)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
      
    # Med-Rare
    if counter==1 and T[nx//2, ny//2, nz//2]>328:
        print("Med Rare")
        print(f"Step: {t}, Height: {heights[t]}")
        counter+=1
        plt.figure(figsize=(6, 5))
        plt.imshow(T[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (Medium Rare)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
      
    # Med
    if counter==2 and T[nx//2, ny//2, nz//2]>333:
        print("Medium")
        print(f"Step: {t}, Height: {heights[t]}")
        counter+=1
        plt.figure(figsize=(6, 5))
        plt.imshow(T[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (Medium)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
      
    # Med-well
    if counter==3 and T[nx//2, ny//2, nz//2]>338:
        print("Med Well")
        print(f"Step: {t}, Height: {heights[t]}")
        counter+=1
        plt.figure(figsize=(6, 5))
        plt.imshow(T[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (Medium Well)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()

    # Well done
    if counter==4 and T[nx//2, ny//2, nz//2]>343:
        print("Well done")
        print(f"Step: {t}, Height: {heights[t]}")
        counter+=1
        plt.figure(figsize=(6, 5))
        plt.imshow(T[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
        plt.colorbar(label="Temperature (K)")
        plt.title("2D Steak cross-section (Well done)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
        
    # Heat transfers
    q_dot_t = heat(T[0,:,:],velocities[t], heights[t]) * dt/ (dz * rho * c_p)
    T_new[0, :, :] += q_dot_t +(rad_cooling(T[0,:,:], heights[t]) * dt) / (dz*rho * c_p)  # Left
    T_new[-1, :, :] += (rad_cooling(T[-1,:,:], heights[t]) * dt) / (dz*rho * c_p)  # Right
    T_new[1:-1, 0, :] += (rad_cooling(T[1:-1,0,:], heights[t]) * dt) / (dz*rho * c_p)  # Bottom
    T_new[1:-1, -1, :] += (rad_cooling(T[1:-1, -1,:], heights[t]) * dt) / (dz*rho * c_p)  # Top
    T_new[1:-1, 1:-1, 0] += (rad_cooling(T[1:-1,1:-1,0], heights[t]) * dt) / (dz*rho * c_p) # Front
    T_new[1:-1, 1:-1, -1] += (rad_cooling(T[1:-1,1:-1,-1], heights[t]) * dt) / (dz*rho * c_p) # Back
    

    # Update temperature field
    T = T_new.copy()


print(f"Total steps: {time_steps}")
print(f"Max central temperature: {T_max_arr[nx//2, ny//2, nz//2]}")
print(f"Step: {step}")

# Plot Final Results
plt.figure(figsize=(6, 5))
plt.imshow(T[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
plt.colorbar(label="Temperature (K)")
plt.title("2D Steak cross-section")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()
plt.figure(figsize=(6, 5))
plt.imshow(T[:,:,nz//2], cmap="hot", origin="lower", extent=[0,ny*dy,0,nx*dx])
plt.colorbar(label="Temperature (K)")
plt.title("2D Steak cross-section")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()
plt.imshow(T[:,ny//2,:], cmap="hot", origin="lower", extent=[0,nz*dz,0,nx*dx])
plt.colorbar(label="Temperature (K)")
plt.title("2D Steak cross-section")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()

# Plot max temperature array
plt.imshow(T_max_arr[nx//2,:,:], cmap="hot", origin="lower", extent=[0,ny*dy,0,nz*dz])
plt.colorbar(label="Temperature (K)")
plt.title("2D Steak cross-section (Max temperature)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()
plt.imshow(T_max_arr[:,ny//2,:], cmap="hot", origin="lower", extent=[0,nz*dz,0,nx*dx])
plt.colorbar(label="Temperature (K)")
plt.title("2D Steak cross-section (Max temperature)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()
plt.imshow(T_max_arr[:,:,nz//2], cmap="hot", origin="lower", extent=[0,ny*dy,0,nx*dx])
plt.colorbar(label="Temperature (K)")
plt.title("2D Steak cross-section (Max temperature)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()
