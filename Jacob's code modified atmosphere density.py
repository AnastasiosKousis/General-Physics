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
h_initial = 100000  # m (initial height 100 km)
dt = 0.1  # Time step (s)
t_max = 500  # Max simulation time in seconds

# Atmospheric density model (simplified exponential decay)
def air_density(h):
    if h < 11000:
        #Troposhere, 0-11000m
        T = 15.04 - 0.00649*h
        p = 101.29 * ((T + 273.1)/288.08)**5.256
    elif h >= 11000 and h <= 25000:
        #Lower Stratosphere, 11,000-25,000m
        T = -56.46
        p = 22.65 * np.exp(1.73 - .000157 * h)
    else:
        #Upper stratosphere, >25000m
        T = -131.21 + .00299 * h
        p = 2.488 * ((T + 273.1)/ 216.6)**-11.388
        
    return p / (0.2869 * (T + 273.1))

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
