# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 19:56:05 2025

@author: Alex Boon
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
mass = 1  # kg
area = 0.005  # m^2
Cd = 1.2  # Drag coefficient (approx for irregular shape)
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
M_earth = 5.972e24  # kg
R_earth = 6371000  # m (Earth's radius)
h_initial = 50e3  # m (initial height 50 km)
dt = 0.01  # Time step (s)
t_max = 500  # Max simulation time in seconds

# Atmospheric density model (simplified exponential decay)
def air_density(h):
    if h < 11000:
        T = 15.04 - 0.00649*h
        p = 101.29 * ((T + 273.1)/288.08)**5.256
    elif 11000 <= h <= 25000:
        T = -56.46
        p = 22.65 * np.exp(1.73 - .000157 * h)
    else:
        T = -131.21 + .00299 * h
        p = 2.488 * ((T + 273.1)/216.6)**-11.388
    return p / (0.2869 * (T + 273.1))

# Gravity as a function of altitude
def gravity(h):
    return G * M_earth / (R_earth + h) ** 2

# Function defining the system of equations
def derivatives(t, h, v):
    g = gravity(h)
    rho = air_density(h)
    drag_force = 0.5 * Cd * rho * v**2 * area * np.sign(v)  # Opposes motion
    dvdt = -g - drag_force / mass  # Net acceleration
    dhdt = v  # Velocity relation
    return dhdt, dvdt

# Simulation variables
h = h_initial
v = 0  # Initial velocity
t = 0  # Time
times = []
velocities = []
altitudes = []

# Runge-Kutta 4th Order Integration
while h > 0:
    # RK4 steps
    k1_h, k1_v = derivatives(t, h, v)
    k2_h, k2_v = derivatives(t + dt/2, h + k1_h * dt/2, v + k1_v * dt/2)
    k3_h, k3_v = derivatives(t + dt/2, h + k2_h * dt/2, v + k2_v * dt/2)
    k4_h, k4_v = derivatives(t + dt, h + k3_h * dt, v + k3_v * dt)

    h += (dt / 6) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
    v += (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
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

# Plot altitude vs time
plt.figure(figsize=(8, 5))
plt.plot(times, altitudes, label="Height (m)")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Height Curve of Falling Steak")
plt.legend()
plt.grid()
plt.show()

# Save data
data = np.column_stack((times, velocities, altitudes))
np.savetxt("Data.txt", data, delimiter=" ")
