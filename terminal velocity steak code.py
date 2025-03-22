# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:59:00 2025

@author: Alex Boon
"""

import numpy as np
import matplotlib.pyplot as plt

# Temp in celcius 
G = 6.67e-11
M = 5.9721986e24
R_e = 6367.5e3
m = 0.5
C_d = 1.2
A = 0.0375
v = 0

velocity_list = []
height_list = []

for h in range(390000,0, -500):
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
        
    rho = p / (0.2869 * (T + 273.1))
    g = G*M/(R_e + h)**2
    v = np.sqrt(m*g/(0.5 * C_d * rho * A))
    
    velocity_list.append(v)
    height_list.append(h)
    

plt.plot(height_list, velocity_list, label='Steak velocity as it falls')
plt.xlabel('Height (m)')
plt.ylabel('Velocity (m/s)')
plt.title("Terminal Velocity as a Function of Altitude")
print(height_list,"\n" ,velocity_list)
    
        
