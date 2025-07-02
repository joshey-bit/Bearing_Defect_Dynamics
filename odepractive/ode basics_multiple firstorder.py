# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:04:23 2022

@author: Rahul Joshi

python ode solver practice - 1st order
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Initialization 
tstart = -1
tstop = 1
step = 0.1

z0 = [1,1] #inital value x0 = 1, y0 = 1
t = np.arange(tstart,tstop+1,step)

#dx/dt
def mydiff(z,t):
    dxdt = -z[1] # dx/dt = -y i.e. z[1]
    dydt = z[0]  ## dy/dt = x i.e. z[0]
    dzdt = [dxdt,dydt]
    return dzdt

#solve Ode
z = odeint(mydiff, z0, t)
print(z)

x =  z[:,0]
y = z[:,1]


#plotting x
plt.plot(t,x)
plt.plot(t,y)
plt.xlabel('t')
plt.ylabel('Z(t)')
plt.grid()
plt.axis([-1,1,-1.5,1.5])
plt.show()

