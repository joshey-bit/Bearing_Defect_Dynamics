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
tstop = 0.15
step = 0.0001

t = np.arange(tstart,tstop+step,step)

#bearing force
Fx,Fy = 0,0 #N 
#note its a common force for both pedestal and shaft

#%%
#shaft properties and input force
ms = 0.250 #kg
ks = 1500000 #N/m
cs = 250 #Ns

#pedestal vibration
#sprung mass system properties
mr = 5 #kg
kr = 2000000 #N/m
cr = 650 #Ns

#pedestel equation:
#pedestal system properties and input force
mp = 0.850 #kg
kp = 800000 #N/m
cp = 450 #Ns

#pedestal inital values
x0 = np.array([0.002,0,0.005,0]) #inital value x0 = 2, x_dot0 = 0
y0 = np.array([0.008,0,0,0,0.03,0]) 
#inital value yp0 = 1, yp_dot0 = 0, yr0 = 3, yr_dot0 = 0

#%%
#dx/dt
def x_diff(x,t,ms,mp,ks,kp,cs,cp,Fx):
    '''
    x = x1
    dx/dt = x2 or dx1/dt = x2 
    d^x/(dt)^2 = dx2/dt
    
    therefore
    mx'' = -Fx - ky - cy' 

    '''
    x1,x2,x3,x4 = x
    
    dx1dt = x2
    dx2dt = (- (ks+kp)*x1 - (cs+cp)*x2 + kp*x3 + cp*x4 -Fx)/(ms)
    dx3dt = x4
    dx4dt = ((kp*x1) + (cp*x2) - (kp*x3) - (cp*x4) + Fx)/(mp)
    dxdt = [dx1dt,dx2dt,dx3dt,dx4dt]
    
    return dxdt

#dy/dt
def y_diff(y,t,ms,mp,mr,ks,cs,kp,cp,kr,cr,Fy):
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
    y1,y2,y3,y4,y5,y6 = y
    
    dY1_dt = y2
    
    dY2_dt = (-((ks+kp)*y1)-((cs+cp)*y2)+(kp*y3)+(cp*y4) - Fy + (ms*9.81))/(ms)
    dY3_dt = y4
    dY4_dt = (kp*y1 + cp*y2 - ((kp+kr)*y3) - ((cp+cr)*y4) + kr*y5 + cr*y6 + Fy + (mp*9.81))/(mp)
    dY5_dt = y6
    dY6_dt = (kr*y3 + cr*y4 - kr*y5 - cr*y6 + (mr*9.81))/(mr)
    
    dydt = [dY1_dt,dY2_dt,dY3_dt,dY4_dt,dY5_dt,dY6_dt]
    
    return dydt

#solve pedestal equation
x = odeint(x_diff, x0, t, args = (ms,mp,ks,kp,cs,cp,Fx,))
y = odeint(y_diff, y0, t, args = (ms,mp,mr,ks,cs,kp,cp,kr,cr,Fy,))

#x direction result
xs1 =  x[:,0] #x displacement
xs2 = x[:,1] #x velocity
#xs2 = (0.16)*xs2 #frequency

#x direction result
xp1 =  x[:,2] #x displacement
xp2 = x[:,3] #x velocity
#xp2 = (0.16)*xp2 #frequency 


#y direction result
ys1 =  y[:,0] #y displacement
ys2 = y[:,1] #y velocity
#ys2 = (0.16)*ys2 #frequency
#shaft result - end
#y direction result of pedstel
yp1 =  y[:,2] #y displacement
yp2 = y[:,3] #y velocity
#yp2 = (0.16)*yp2 #frequency
#y direction result of sprung mass
yr1 =  y[:,4] #y displacement
yr2 = y[:,5] #y velocity
#yr2 = (0.16)*yr2 #frequency
#pedestal resul -end


#xs-xp
x = xs1 - xp1

#ys-yp
y = ys1 - yp1

#%%
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#plotting
plt.figure(num=1)
plt.subplot(211)
plt.plot(t,xs1)
plt.plot(t,ys1)
plt.xlabel('t (s)')
plt.ylabel('amplitude (m)')
plt.legend(['x_shaft','y_shaft'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,xs2)
plt.plot(t,ys2)
plt.xlabel('t (s)')
plt.ylabel('velocity (m/s)')
plt.legend(['x_shaft','y_shaft'],loc = 1)
plt.grid()

plt.show()

#plotting
plt.figure(num=2)
plt.subplot(211)
plt.plot(t,xp1)
plt.plot(t,yp1)
plt.xlabel('t (s)')
plt.ylabel('amplitude (m)')
plt.legend(['x_pedestal','y_pedestal'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,xp2)
plt.plot(t,yp2)
plt.xlabel('t (s)')
plt.ylabel('velocity (m/s)')
plt.legend(['x_pedestal','y_pedestal'],loc = 1)
plt.grid()
plt.show()

#plotting
plt.figure(num=3)
plt.subplot(211)
plt.plot(t,yr1)
plt.xlabel('t (s)')
plt.ylabel('amplitude (m)')
plt.legend(['y_sprung'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,yr2)
plt.xlabel('t (s)')
plt.ylabel('velocity (m/s)')
plt.legend(['y_sprung'],loc = 1)
plt.grid()
plt.show()

#end result
plt.figure(num = 4)
plt.plot(t,x)
plt.plot(t,y)
plt.xlabel('t (s)')
plt.ylabel('amplitude (m)')
plt.legend(['x','y'],loc = 1)
plt.grid()
plt.show()




