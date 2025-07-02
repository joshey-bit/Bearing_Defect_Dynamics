# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:21:56 2021

@author: Rahul Joshi
"""
#angular postion of elements
import numpy as np
import math as mt
import matplotlib.pyplot as plt

wc = (2 * mt.pi)/10 #angular velocity
n = 10 #number of balls

tstart = 0
tstop = 10
step = 0.1

t = np.arange(tstart,tstop+step,step)
t_length = len(t)


#initialization
fi = np.zeros((t_length,n))

#initial position
for j in range(n):
    fi_new = ((2*mt.pi)*(j))/n 
    fi[0][j] = fi_new

#process completion flag
print("initial position")
#print(fi)
    
for k in range(t_length):
    for j in range(n):
        fi_t = ((2*mt.pi)*(j))/n + wc*t[k]
        fi[k][j] = fi_t

#process completion flag
print('completed')

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#correctness test
def test(fi):
    #after taking cos of the angles
    cos_values = np.cos(fi)
    cos_values = np.transpose(cos_values)

    #plotting_the obatined result
    fi_rows = np.shape(cos_values)[0]
    for i in range(fi_rows):
        plt.plot(t,cos_values[i])
        plt.xlabel("t(s)")
        plt.ylabel("cos(fi)")
        plt.grid()
    return cos_values
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#testing process
cos_val = test(fi)


