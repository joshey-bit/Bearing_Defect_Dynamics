# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:04:23 2022

@author: Rahul Joshi

python ode solver practice - higher order
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Initialization 
tstart = 0
tstop = 5
step = 0.1

y0 = np.array([1,0]) #inital value y0 = 0, y_dot0 = 0
t = np.arange(tstart,tstop,step)

#system properties and input force
m = 10 #kg
k = 1500 #N/m
c = 25 #Ns
F = 0 #N


#dx/dt
def mydiff(y,t,m,k,c,F):
    '''
    y = x1
    dy/dt = x2 or dx1/dt = x2 
    d^y/(dt)^2 = dx2/dt
    
    therefore
    my'' = F - ky - cy' or y'' = (F - ky - cy')/m 

    '''
    dx1dt = y[1]
    dx2dt = (F - k*y[0] - c*y[1])/(m)
    dxdt = [dx1dt,dx2dt]
    return dxdt

#solve Ode
y = odeint(mydiff, y0, t, args = (m,k,c,F,))
print(y)

y1 =  y[:,0]
y2 = y[:,1]

print('\n max amplitude ='+str(y1.max()))


#plotting x
plt.plot(t,y1)
plt.plot(t,y2)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend(['displacement','velocity'])
plt.grid()
plt.show()


