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


#system properties and input force
ms = 4 #kg
ks = 1500000 #N/m
cs = 250 #Ns

#shaft vibration initial value
xs0 = np.array([0.002,0]) #inital value x0 = 2, x_dot0 = 0
ys0 = np.array([0.008,0]) #inital value y0 = 1, y_dot0 = 0

#dx/dt
def x_diff(xs,t,ms,ks,cs,Fx):
    '''
    x = x1
    dx/dt = x2 or dx1/dt = x2 
    d^x/(dt)^2 = dx2/dt
    
    therefore
    mx'' = -Fx - ky - cy' 

    '''
    x1,x2 = xs
    
    dx1dt = x2
    dx2dt = (-Fx - ks*x1 - cs*x2)/(ms)
    dxdt = [dx1dt,dx2dt]
    
    return dxdt

#dy/dt
def y_diff(ys,t,ms,ks,cs,Fy):
    '''
    y = Y1
    dy/dt = Y2 or dY1/dt = Y2 
    d^y/(dt)^2 = dY2/dt
    
    therefore
    my'' = F - ky - cy' + m*g 

    '''
    y1, y2 = ys
    
    dY1dt = y2
    dY2dt = (Fy - ks*y1 - cs*y2 + ms*9.81)/(ms)
    dydt = [dY1dt,dY2dt]
    
    return dydt


#pedestal vibration
#sprung mass system properties
mr = 10 #kg
kr = 2000000 #N/m
cr = 650 #Ns

#pedestel equation:
#pedestal system properties and input force
mp = 5 #kg
kp = 800000 #N/m
cp = 450 #Ns

#pedestal inital values
xp0 = np.array([0.005,0]) #inital value x0 = 2, x_dot0 = 0
yp0 = np.array([0.0,0,-0.03,0]) 
#inital value yp0 = 1, yp_dot0 = 0, yr0 = 3, yr_dot0 = 0



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

#solve shaft equation
xs = odeint(x_diff, xs0, t, args = (ms,ks,cs,Fx,))
ys = odeint(y_diff, ys0, t, args = (ms,ks,cs,Fy,))


#solve pedestal equation
xp = odeint(x_ped_diff, xp0, t, args = (mp,kp,cp,Fx,))
yp = odeint(y_ped_diff, yp0, t, args = (mp,kp,cp,kr,cr,Fy,))


#shaft result- start

#x direction result
xs1 =  xs[:,0] #x displacement
xs2 = xs[:,1] #x velocity
#xs2 = (0.16)*xs2 #frequency 

#y direction result
ys1 =  ys[:,0] #y displacement
ys2 = ys[:,1] #y velocity
#ys2 = (0.16)*ys2 #frequency
#shaft result - end

#pedestal resul -start

#x direction result
xp1 =  xp[:,0] #x displacement
xp2 = xp[:,1] #x velocity
#xp2 = (0.16)*xp2 #frequency

#y direction result of pedstel
yp1 =  yp[:,0] #y displacement
yp2 = yp[:,1] #y velocity
#yp2 = (0.16)*yp2 #frequency
#y direction result of sprung mass
yr1 =  yp[:,2] #y displacement
yr2 = yp[:,3] #y velocity
#yr2 = (0.16)*yr2 #frequency
#pedestal resul -end


#xs-xp
x = xs1 - xp1

#ys-yp
y = ys1 - yp1


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




