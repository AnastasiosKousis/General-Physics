# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 19:56:05 2025

@author: Alex Boon
"""

import numpy as np
import matplotlib.pyplot as plt

# Change this
h_initial = 200e3  # m (initial height)

# These need to be changed if the parameters of the other simulation change.
dt = 0.05  # Time step (s)
volume = 50*50*25*0.001**3 # m^3 (50x50x25 cuboid dx,dy,dz=0.001m)

# Constants
rho_steak = 1050
mass = rho_steak*volume
area = 0.0025  # m^2
Cd = 1.2  # Drag coefficient (approx for irregular shape)
Grav_const = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
M_earth = 5.972e24  # kg
R_earth = 6371000  # m (Earth's radius)
 

# Atmospheric density model
RE = 6371000  # Earth's radius in meters
M = 5.972e24  # Earth's mass (kg)
M = 5.972e24  # Earth's mass (kg)
R = 287.0  # Specific gas constant for dry air (J/kg·K)
sigma = 5.670e-8  # Stefan-Boltzmann constant (W/m²·K⁴)
gamma = 1.4  # Ratio of specific heats for air
m_air = 4.8e-26  # Approximate mass of air molecule (kg)


def air_density(h):
    """Computes air density (kg/m³) at altitude h (meters) using NASA model."""
    G = gravity(h)

    if h < 11000:
        T = 288.15 - 0.0065 * h
        P = 101325 * (T / 288.15) ** 5.256

    elif h < 20000:
        T = 216.65
        P = 22631 * np.exp(-0.000157 * (h - 11000))

    elif h < 32000:
        T = 216.65 + 0.001 * (h - 20000)
        P = 5509 * (216.65 / T) ** 34.16

    elif h < 47000:
        T = 228.65 + 0.0028 * (h - 32000)
        P = 873.58 * (228.65 / T) ** 12.2

    elif h < 51000:
        T = 270.65
        P = 111.64 * np.exp(-0.000126 * (h - 47000))

    elif h < 71000:
        T = 270.65 - 0.0028 * (h - 51000)
        P = 67.443 * (270.65 / T) ** -12.2

    elif h < 86000:
        T = 214.65 - 0.002 * (h - 71000)
        P = 3.987 * (214.65 / T) ** -17.08

    elif h < 91000:
        T = 186.8673
        P = 0.374 * np.exp(-G * (h - 86000) / (R * T))

    elif h < 110000:
        T = 263.1905 - 76.32321 * (1 - np.sqrt(1 - ((h - 91000) / 19942.9)**2))
        P = 0.374 * np.exp(-G * (h - 86000) / (R * T))

    elif h < 120000:
        T = 240 + 0.012 * (h - 110000)
        P = 0.374 * np.exp(-G * (h - 86000) / (R * T))

    else:
        T = 1000 - 640 * np.exp(-0.00001875 * (h - 120000) * ((RE + 120000) / (RE + h)))
        P = 0.374 * np.exp(-G * (h - 86000) / (R * T))

    # Compute air density using the ideal gas law
    rho = P / (R * T)
    return rho

def ambient_temp(h):
    """Computes ambient temperature (K) at altitude h (meters) using NASA model."""
    
    if h < 11000:  # Troposphere (0 - 11 km)
        T = 288.15 - 0.0065 * h

    elif h < 20000:  # Tropopause (11 - 20 km)
        T = 216.65

    elif h < 32000:  # Stratosphere Lower (20 - 32 km)
        T = 216.65 + 0.001 * (h - 20000)

    elif h < 47000:  # Stratosphere Upper (32 - 47 km)
        T = 228.65 + 0.0028 * (h - 32000)

    elif h < 51000:  # Stratopause (47 - 51 km)
        T = 270.65

    elif h < 71000:  # Mesosphere Lower (51 - 71 km)
        T = 270.65 - 0.0028 * (h - 51000)

    elif h < 86000:  # Mesosphere Upper (71 - 86 km)
        T = 214.65 - 0.002 * (h - 71000)

    elif h < 91000:  # Thermosphere (86 - 91 km)
        T = 186.8673

    elif h < 110000:  # Thermosphere (91 - 110 km)
        T = 263.1905 - 76.32321 * (1 - np.sqrt(1 - (h - 91000) / 19942.9))

    elif h < 120000:  # Thermosphere (110 - 120 km)
        T = 240 + 0.012 * (h - 110000)

    else:  # Exosphere (120 - 1000 km)
        RE = 6371000  # Earth's radius in meters
        T = 1000 - 640 * np.exp(-0.00001875 * (h - 120000) * ((RE + 120000) / (RE + h)))

    return T  # Temperature in Kelvin

# Gravity as a function of altitude
def gravity(h):
    return Grav_const * M_earth / (R_earth + h) ** 2

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
temperatures = []
densities = []

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
    temperatures.append(ambient_temp(h))
    densities.append(air_density(h))

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

# Plot ambient temp vs heights
plt.figure(figsize=(8, 5))
plt.plot(altitudes, densities, label=r"Density (kgm$^{-3}$)")
plt.xlabel("Height (m)")
plt.ylabel(r"Density (kgm$^{-3}$)")
plt.title("Air Density Curve")
plt.legend()
plt.grid()
plt.show()

# Plot ambient temp vs heights
plt.figure(figsize=(8, 5))
plt.plot(altitudes, temperatures, label="Temperature (K)")
plt.xlabel("Height (m)")
plt.ylabel("Temperature (K)")
plt.title("Ambient Temperature Curve")
plt.legend()
plt.grid()
plt.show()

# Save data
data = np.column_stack((times, velocities, altitudes))
np.savetxt("Data.txt", data, delimiter=" ")
