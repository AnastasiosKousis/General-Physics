# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 12:19:33 2025

@author: mattc
"""

import numpy as np

RE = 6371000  # Earth's radius in meters
M = 5.972e24  # Earth's mass (kg)
G = 9.80665  # Gravitational acceleration (m/s²)
M = 5.972e24  # Earth's mass (kg)
R = 287.0  # Specific gas constant for dry air (J/kg·K)
sigma = 5.670e-8  # Stefan-Boltzmann constant (W/m²·K⁴)
gamma = 1.4  # Ratio of specific heats for air
m_air = 4.8e-26  # Approximate mass of air molecule (kg)


def air_density(h):
    """Computes air density (kg/m³) at altitude h (meters) using NASA model."""

    if h < 11000:
        T = 288.15 - 0.0065 * h
        P = 101325 * (T / 288.15) ** 5.256

    elif h < 20000:
        T = 216.65
        P = 22631 * np.exp(-0.000157 * (h - 11000))

    elif h < 32000:
        T = 216.65 + 0.001 * (h - 20000)
        P = 55092 * (216.65 / T) ** 34.16

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
        T = 263.1905 - 76.32321 * (1 - np.sqrt(1 - (h - 91000) / 19942.9))
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