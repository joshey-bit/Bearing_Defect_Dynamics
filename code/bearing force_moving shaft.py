# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:21:56 2021

@author: Rahul Joshi
"""
#angular postion of elements
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import bearing_vibration_moving_shaft as vib

#n = int(input("enter the no. of balls: "))
n = 9
#rpm = int(input('enter rpm: '))
rpm = 6

d = 0.008 #ball dia in m
ri = ro = 0.150  #inner and outer groove radius

#curvature ratio
const = ((ro+ri)-d)/d

#Deformation constant
Kb = (34300/(const**0.35))*(d**0.5)
e = 1.5 #constant for ball bearings
#not working for e = 3/2 which is the actual constant

'''
#initialization
tstart = 0
tstop = 0.15
step = 0.0001
t = np.arange(tstart,tstop,step)
'''
t = vib.t

#angular velocity of cage in rad/s
wc = (2 * mt.pi * rpm)/60 

#displacement inputs
x = vib.x
y = vib.y

#ball position array
fi = np.zeros(n)
#ball displacement array
delta_array = np.zeros(n)
#force array
#fx
fx_array = np.zeros(n)
#fy
fy_array = np.zeros(n)

#force time array
fx_t = np.zeros(len(t))
fy_t = np.zeros(len(t))
    
#process completion flag
#print("initial position")
#%%
#delta function for displacement
def delta_func(theta,x,y):
    #note theta is ball position
    delta = x * np.cos(theta) +  y * np.sin(theta)
    return delta
    
for k in t:
    #ball position equation
    for j in range(n):
        fi_t = ((2*mt.pi)*(j))/n + wc*(k/10000)
        fi[j] = fi_t
    
    #delta equation
    for j in range(n):
        delta_array[j] = delta_func(fi[j], x[k], y[k])
        
    #bearing force calculation
    for j in range(n):
        delta_value =  delta_array[j]
        if delta_value >= 0:
            fx_array[j] = (Kb * (delta_value**e) * np.cos(fi[j]))
            fy_array[j] = (Kb * (delta_value**e) * np.sin(fi[j]))
        
    #fx summation
    fx_t[k] = np.sum(fx_array)
    
    #fy summation
    fy_t[k] = np.sum(fy_array)
    
    '''
    print("time step: " + str(t[k]))
    print(fi)
    print('\n')
    print(delta_array)
    '''

#process completion flag

print('completed')
#%%
#plot
plt.subplot(211)
plt.plot(t,fx_t,'r-')
plt.xlabel('iterations')
plt.ylabel('Fx (N)')
plt.grid()
plt.subplot(212)
plt.plot(t,fy_t,'r-')
plt.xlabel('iterations')
plt.ylabel('Fy (N)')
plt.grid()
plt.show()




