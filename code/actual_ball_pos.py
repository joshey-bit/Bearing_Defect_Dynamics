# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:21:56 2021

@author: Rahul Joshi
"""
#angular postion of elements
import numpy as np
import math as mt
#import matplotlib.pyplot as plt

wc = (2 * mt.pi)/10 #angular velocity
n = int(input("enter the no. of balls: "))
t = int(input('enter sample time: '))

#initialization
fi = np.zeros(n)

#initial position
for j in range(n):
    fi_new = ((2*mt.pi)*(j))/n 
    fi[j] = fi_new

#process completion flag
print("initial position")
print(fi)
    
for k in range(1,t+1):
    for j in range(n):
        fi_t = fi[j] + wc*k
        fi[j] = fi_t
    print("time step: " + str(k))
    print(fi)

#process completion flag
print('completed')




