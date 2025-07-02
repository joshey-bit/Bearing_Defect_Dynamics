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
tstart = 0
tstop = 25
step = 1

T = 5 #constant
a = -1/T

x0 = 5 #inital value
t = np.arange(tstart,tstop+1,step)
k = []
#t = np.array([0,1,2,3])

#dx/dt
def mydiff(x,t,a):
    dxdt = a * x
    return dxdt

for i in range(len(t)):
    k.append(t[i])
    #solve Ode
    x = odeint(mydiff, x0, k, args= (a,))

print(x)        


#plotting x
plt.plot(t,x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.axis([0,25,0,6])
plt.show()


