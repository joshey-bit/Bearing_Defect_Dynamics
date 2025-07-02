# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:04:23 2022

@author: Rahul Joshi

bearing model_ vibration equation of shaft
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Initialization 
tstart = 0
tstop = 5
step = 0.1

x0 = np.array([2,0]) #inital value y0 = 0, y_dot0 = 0
y0 = np.array([1,0]) #inital value y0 = 0, y_dot0 = 0
t = np.arange(tstart,tstop,step)

#system properties and input force
m = 10 #kg
k = 1500 #N/m
c = 25 #Ns
Fx,Fy = 0,0 #N

#dx/dt
def x_diff(x,t,m,k,c,Fx):
    '''
    x = x1
    dx/dt = x2 or dx1/dt = x2 
    d^x/(dt)^2 = dx2/dt
    
    therefore
    mx'' = -Fx - ky - cy' 

    '''
    x1,x2 = x
    
    dx1dt = x2
    dx2dt = (-Fx - k*x1 - c*x2)/(m)
    dxdt = [dx1dt,dx2dt]
    
    return dxdt

#dy/dt
def y_diff(y,t,m,k,c,Fy):
    '''
    y = Y1
    dy/dt = Y2 or dY1/dt = Y2 
    d^y/(dt)^2 = dY2/dt
    
    therefore
    my'' = F - ky - cy' + m*g 

    '''
    y1, y2 = y
    
    dY1dt = y2
    dY2dt = (Fy - k*y1 - c*y2 + m*9.81)/(m)
    dydt = [dY1dt,dY2dt]
    
    return dydt



#solve Ode
x = odeint(x_diff, x0, t, args = (m,k,c,Fx,))
y = odeint(y_diff, y0, t, args = (m,k,c,Fy,))
#print(y)

#x direction result
xs1 =  x[:,0] #y displacement
xs2 = x[:,1] #y velocity

#y direction result
ys1 =  y[:,0] #y displacement
ys2 = y[:,1] #y velocity

#print('\n max amplitude ='+str(y1.max()))


#plotting x
plt.subplot(211)
plt.plot(t,xs1)
plt.plot(t,ys1)
plt.xlabel('t')
plt.ylabel('amplitude')
plt.legend(['x','y'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,xs2)
plt.plot(t,ys2)
plt.xlabel('t')
plt.ylabel('velocity')
plt.legend(['x','y'],loc = 1)
plt.grid()

plt.show()


