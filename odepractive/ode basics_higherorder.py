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

x0 = np.array([0,1]) #inital value x0 = 0, y0 = 1
t = np.arange(tstart,tstop+1,step)

#dx/dt
def mydiff(x,t):
    '''
    w = x1
    dw/dt = x2 or dx1/dt = x2 
    d^w/(dt)^2 = dx2/dt
    
    therefore
    x'' = 2 - 2tx' - 3x/(1+t^2)

    '''
    dx1dt = x[1]
    dx2dt = (2 - 2*t*x[1] - 3*x[0])/(1+t**2)
    dxdt = [dx1dt,dx2dt]
    return dxdt

#solve Ode
x = odeint(mydiff, x0, t)
print(x)

x1 =  x[:,0]
x2 = x[:,1]


#plotting x

plt.plot(t,x1)
plt.plot(t,x2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.axis([0,5,-1,2])
plt.show()


