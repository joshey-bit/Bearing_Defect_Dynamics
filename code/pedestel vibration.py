# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:04:23 2022

@author: Rahul Joshi

bearing model_ vibration equation of pedestal and sprung_mass
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Initialization/ time range 
tstart = 0
tstop = 0.5
step = 0.01

t = np.arange(tstart,tstop,step)


#sprung mass system properties
mr = 12 #kg
kr = 2000000 #N/m
cr = 80 #Ns

#pedestel equation:
#pedestal system properties and input force
mp = 5 #kg
kp = 1000000 #N/m
cp = 150 #Ns
Fx,Fy = 0,0 #N

#inital values
x0 = np.array([0.005,0]) #inital value x0 = 2, x_dot0 = 0
y0 = np.array([0.005,0,-0.005,0]) #inital value yp0 = 1, yp_dot0 = 0, yr0 = 3, yr_dot0 = 0


#dx/dt
def x_ped_diff(xp,t,mp,kp,cp,Fx):
    '''
    x = x1
    dx/dt = x2 or dx1/dt = x2 
    d^x/(dt)^2 = dx2/dt
    
    therefore
    mpx'' = Fx - kx - cx' 

    '''
    x1,x2 = xp
    dx1_dt = x2
    dx2_dt = (Fx - kp*x1 - cp*x2)/(mp)
    dx_dt = [dx1_dt,dx2_dt]
    return dx_dt

#dy/dt
def y_ped_diff(yp,t,mp,kp,cp,kr,cr,Fy):
    '''
    y = Y1
    dy/dt = Y2 or dY1/dt = Y2 
    d^y/(dt)^2 = dY2/dt
    
    therefore
    my'' = F - (kp+ky)y - (cp+cr)y' + mp*g  + kryr + cryr'
    
    yr = Y3
    dyr/dt = y4 or dy3/dt = y4
    d^yr/(dt)^2 = dy4/dt
    
    therefore
    mryr'' = kr(yp - yr) + cr(yp' - yr') - mr*g

    '''
    y1,y2,y3,y4 = yp
    
    dY1_dt = y2
    dY2_dt = (Fy - (kp + kr)*y1 - (cp+cr)*y2 - (mp*9.81) +
               (kr*y3)+(cr*y4))/(mp)
    dY3_dt = y4
    dY4_dt = kr *(y1 - y3) + cr * (y2 - y4) - (mr * 9.81)
    
    dydt = [dY1_dt,dY2_dt,dY3_dt,dY4_dt]
    return dydt

#solve Ode
xp = odeint(x_ped_diff, x0, t, args = (mp,kp,cp,Fx,))
#solve Ode
yp = odeint(y_ped_diff, y0, t, args = (mp,kp,cp,kr,cr,Fy,))
#print(y)

#x direction result
xp1 =  xp[:,0] #x displacement
xp2 = xp[:,1] #x velocity

#y direction result of pedstel
yp1 =  yp[:,0] #y displacement
yp2 = yp[:,1] #y velocity
#y direction result of sprung mass
yr1 =  yp[:,2] #y displacement
yr2 = yp[:,3] #y velocity

#print('\n max amplitude ='+str(y1.max()))


#plotting
plt.subplot(211)
plt.plot(t,xp1)
plt.plot(t,yp1)
plt.plot(t,yr1)
plt.xlabel('t')
plt.ylabel('amplitude')
plt.legend(['x_pedestal','y_pedestal','y_sprung'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,xp2)
plt.plot(t,yp2)
plt.plot(t,yr2)
plt.xlabel('t')
plt.ylabel('velocity')
plt.legend(['x_pedestal','y_pedestal','y_sprung'],loc = 1)
plt.grid()

plt.show()



