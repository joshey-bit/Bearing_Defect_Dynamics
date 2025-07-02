# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:04:23 2022

@author: Rahul Joshi

bearing model_ vibration equation of pedestal and sprung_mass
"""
import numpy as np
import matplotlib.pyplot as plt
import math as mt

#Initialization/ time range 
simtime = 0.3
tstep = 0.0001
iterations = (int(simtime/tstep))+1
t = np.arange(0,iterations)

#%%
#shaft properties and input force
ms = 12 #kg
ks = 1500000 #N/m
cs = 250 #kg/s

#inner race properties
mi = 5 #kg
ki = 2000000 #N/m
ci = 350 #kg/s

#pedestel equation:
#pedestal system properties and input force
mp = 8 #kg
kp = 800000 #N/m
cp = 350 #kg/s

#inital condition values
x0 = np.array([0.002,0,-0.008,0,0.02,0])
x1_0,vx1_0,x2_0,vx2_0,x3_0,vx3_0 = x0

y0 = np.array([0.008,0,-0.005,0,0.01,0]) 
y1_0,vy1_0,y2_0,vy2_0,y3_0,vy3_0 = y0

#%%
'''
bearing force program
'''
#bearing details
n = 9 #no. of balls
d = 0.008 #ball dia in m
ri = ro = 0.150  #inner and outer groove radius
#curvature ratio
const = ((ro+ri)-d)/d

#Deformation constant
Kb = (34300/(const**0.35))*(d**0.5)
e = 1.5 #constant for ball bearings

#angular velocity of cage in rad/s
rpm = 6
wc = (2 * mt.pi * rpm)/60 

#array initailization
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



def bearing_force(x,y,wc,k,Kb,e):
    for j in range(n):
        fi_t = ((2*mt.pi)*(j))/n + wc*(k/10000)
        fi[j] = fi_t
    
    #delta equation
    for j in range(n):
        delta_array[j] = x * np.cos(fi[j]) +  y * np.sin(fi[j])
        
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
    bearin_force = [fx_t[k],fy_t[k]]
    return bearin_force


#%%
#x_direction vibration
#displacement array
x1 = np.zeros((iterations,1))
x1[0,:] = x1_0

x2 = np.zeros((iterations,1))
x2[0,:] = x2_0

x3 = np.zeros((iterations,1))
x3[0,:] = x3_0

#velocity array
vx1 = np.zeros((iterations,1))
vx1[0,:] = vx1_0

vx2 = np.zeros((iterations,1))
vx2[0,:] = vx2_0

vx3 = np.zeros((iterations,1))
vx3[0,:] = vx3_0

# y - direction vibration
#displacement array
y1 = np.zeros((iterations,1))
y1[0,:] = y1_0

y2 = np.zeros((iterations,1))
y2[0,:] = y2_0

y3 = np.zeros((iterations,1))
y3[0,:] = y3_0

#velocity array
vy1 = np.zeros((iterations,1))
vy1[0,:] = vy1_0

vy2 = np.zeros((iterations,1))
vy2[0,:] = vy2_0

vy3 = np.zeros((iterations,1))
vy3[0,:] = vy3_0

a = x2_0 - x1_0
b = y2_0 - y1_0
#bearing force function call
b_force = bearing_force(a, b, wc, 0, Kb, e)
Fx = b_force[0] #N
Fy = b_force[1] #N

#accleration array x-direction
ax1 = np.zeros((iterations,1))
ax1[0,:] = (-(kp+ki)*x1_0 - (cp+ci)*vx1_0 + ki*x2_0 + ci*vx2_0 - Fx )/(mp)

ax2 = np.zeros((iterations,1))
ax2[0,:] = (ki*x1_0 + ci*vx1_0 - (ki+ks)*x2_0 - (ci+cs)*vx2_0 + ks*x3_0 + cs*vx3_0 + Fx)/(mi)

ax3 = np.zeros((iterations,1))
ax3[0,:] = (ks*x2_0 + cs*vx2_0 - ks*x3_0 - cs*vx3_0)/(ms)

#accleration array y-direction
ay1 = np.zeros((iterations,1))
ay1[0,:] = (-(kp+ki)*y1_0 - (cp+ci)*vy1_0 + ki*y2_0 + ci*vy2_0 - Fy - mp*9.81 )/(mp)

ay2 = np.zeros((iterations,1))
ay2[0,:] = (ki*y1_0 + ci*vy1_0 - (ki+ks)*y2_0 - (ci+cs)*vy2_0 + ks*y3_0 + cs*vy3_0 + Fy - mi*9.81 )/(mi)

ay3 = np.zeros((iterations,1))
ay3[0,:] = (ks*y2_0 + cs*vy2_0 - ks*y3_0 - cs*vy3_0 - ms*9.81)/(ms)


#solve ode with euler method
for p in range(1,iterations):
    #x - displacement
    x1[p,:] = x1[p-1,:] + vx1[p-1,:] * tstep
    x2[p,:] = x2[p-1,:] + vx2[p-1,:] * tstep
    x3[p,:] = x3[p-1,:] + vx3[p-1,:] * tstep
    
    #y - displacement
    y1[p,:] = y1[p-1,:] + vy1[p-1,:] * tstep
    y2[p,:] = y2[p-1,:] + vy2[p-1,:] * tstep
    y3[p,:] = y3[p-1,:] + vy3[p-1,:] * tstep
    
    #x - velocity
    vx1[p,:] = vx1[p-1,:] + ax1[p-1,:] * tstep
    vx2[p,:] = vx2[p-1,:] + ax2[p-1,:] * tstep
    vx3[p,:] = vx3[p-1,:] + ax3[p-1,:] * tstep
    
    #velocity
    vy1[p,:] = vy1[p-1,:] + ay1[p-1,:] * tstep
    vy2[p,:] = vy2[p-1,:] + ay2[p-1,:] * tstep
    vy3[p,:] = vy3[p-1,:] + ay3[p-1,:] * tstep
    
    c = float(x2[p,:] - x1[p,:])
    d = float(y2[p,:] - y1[p,:])
    
    #bearing force function call
    b_force = bearing_force(c, d, wc, p, Kb, e)
    Fx = b_force[0] #N
    Fy = b_force[1] #N
    
    #print(b_force)
    
    #x - acceleration
    ax1[p,:] = (-(kp+ki)*x1[p,:] - (cp+ci)*vx1[p,:] + ki*x2[p,:] + ci*vx2[p,:] - Fx )/(mp)
    ax2[p,:] = (ki*x1[p,:] + ci*vx1[p,:] - (ki+ks)*x2[p,:] - (ci+cs)*vx2[p,:] + ks*x3[p,:] + cs*vx3[p,:] + Fx)/(mi)
    ax3[p,:] = (ks*x2[p,:] + cs*vx2[p,:] - ks*x3[p,:] - cs*vx3[p,:])/(ms)
    
    #acceleration
    ay1[p,:] = (-(kp+ki)*y1[p,:] - (cp+ci)*vy1[p,:] + ki*y2[p,:] + ci*vy2[p,:] - Fy - mp*9.81 )/(mp)
    ay2[p,:] = (ki*y1[p,:] + ci*vy1[p,:] - (ki+ks)*y2[p,:] - (ci+cs)*vy2[p,:] + ks*y3[p,:] + cs*vy3[p,:] + Fy - mi*9.81)/(mi)
    ay3[p,:] = (ks*y2[p,:] + cs*vy2[p,:] - ks*y3[p,:] - cs*vy3[p,:] - ms*9.81)/(ms)
    

#%%
#xi-xp
x = x2 - x1

#yi-yp
y = y2 - y1    
#%%
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#plotting
plt.figure(num=1)
plt.subplot(211)
plt.plot(t,x3)
plt.plot(t,y3)
plt.xlabel('time steps')
plt.ylabel('amplitude (m)')
plt.legend(['x_shaft','y_shaft'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,vx3)
plt.plot(t,vy3)
plt.xlabel('time steps')
plt.ylabel('velocity (m/s)')
plt.legend(['x_shaft','y_shaft'],loc = 1)
plt.grid()

plt.show()

#plotting
plt.figure(num=2)
plt.subplot(211)
plt.plot(t,x2)
plt.plot(t,y2)
plt.xlabel('time steps')
plt.ylabel('amplitude (m)')
plt.legend(['x_inner','y_inner'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,vx2)
plt.plot(t,vy2)
plt.xlabel('time steps')
plt.ylabel('velocity (m/s)')
plt.legend(['x_inner','y_inner'],loc = 1)
plt.grid()
plt.show()

#plotting
plt.figure(num=3)
plt.subplot(211)
plt.plot(t,x1)
plt.plot(t,y1)
plt.xlabel('time steps')
plt.ylabel('amplitude (m)')
plt.legend(['x_outer','y_outer'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,vx1)
plt.plot(t,vy1)
plt.xlabel('time steps')
plt.ylabel('velocity (m/s)')
plt.legend(['x_outer','y_outer'],loc = 1)
plt.grid()
plt.show()

#end result
plt.figure(num = 4)
plt.plot(t,x)
plt.plot(t,y)
plt.xlabel('time steps')
plt.ylabel('amplitude (m)')
plt.legend(['x','y'],loc = 1)
plt.grid()
plt.show()


#%%
#force plot
plt.figure(num = 5)
plt.subplot(211)
plt.plot(t,fx_t,'r-')
plt.xlabel('time steps')
plt.ylabel('Fx (N)')
plt.grid()


plt.subplot(212)
plt.plot(t,fy_t,'r-')
plt.xlabel('time steps')
plt.ylabel('Fy (N)')
plt.grid()
plt.show()

