import numpy as np
import matplotlib.pyplot as plt

# Constants
mass = 1  # kg
area = 0.005  # m^2
Cd = 1.2  # Drag coefficient (approx for irregular shape)
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
M_earth = 5.972e24  # kg
R_earth = 6371000  # m (Earth's radius)
h_initial = 100000  # m (initial height 100 km)
dt = 0.1  # Time step (s)
t_max = 500  # Max simulation time in seconds

# Atmospheric density model (simplified exponential decay)
def air_density(h):
    if h > 85000:
        return 1.0e-6  # Very low density
    elif h > 40000:
        return 4.0e-3 * np.exp(-(h - 40000) / 10000)  # Approximation for upper atmosphere
    else:
        return 1.225 * np.exp(-h / 8000)  # Below 40 km, exponential decay

# Gravity as a function of altitude
def gravity(h):
    return G * M_earth / (R_earth + h) ** 2

# Simulation variables
h = h_initial
v = 0  # Initial velocity
t = 0  # Time
times = []
velocities = []
altitudes = []

# Numerical integration (Euler method)
while h > 0 and t < t_max:
    g = gravity(h)
    rho = air_density(h)
    drag_force = 0.5 * Cd * rho * v**2 * area * np.sign(v)  # Opposes motion
    acceleration = -g - drag_force / mass  # Net acceleration
    v += acceleration * dt
    h += v * dt
    t += dt

    # Store results
    times.append(t)
    velocities.append(v)
    altitudes.append(h)

# Plot velocity vs time
plt.figure(figsize=(8, 5))
plt.plot(times, velocities, label="Velocity (m/s)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Curve of Falling Steak")
plt.legend()
plt.grid()
plt.show()
