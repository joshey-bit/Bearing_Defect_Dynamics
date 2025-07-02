# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:04:23 2022

@author: Rahul Joshi

bearing model_ vibration equation of pedestal and sprung_mass
"""
import numpy as np
import matplotlib.pyplot as plt
import bearing_vibration_final_model as vibrn

#Initialization/ time range 
simtime = 0.3
tstep = 0.0001
iterations = (int(simtime/tstep))+1
t = np.arange(0,iterations)

#bearing force
Fx,Fy = 0,0 #N 
#note its a common force for both pedestal and shaft

#%%
#shaft properties and input force
ms = 12 #kg
ks = 1500000 #N/m
cs = 250 #Ns

#inner race properties
mi = 5 #kg
ki = 2000000 #N/m
ci = 150 #Ns

#pedestel equation:
#pedestal system properties and input force
mp = 8 #kg
kp = 800000 #N/m
cp = 550 #Ns

#inital condition values
x0 = np.array([0.002,0,-0.008,0,0.02,0])
x1_0,vx1_0,x2_0,vx2_0,x3_0,vx3_0 = x0

y0 = np.array([0.008,0,-0.015,0,0.03,0]) 
y1_0,vy1_0,y2_0,vy2_0,y3_0,vy3_0 = y0

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

#accleration array x-direction
ax1 = np.zeros((iterations,1))
ax1[0,:] = (-(kp+ki)*x1_0 - (cp+ci)*vx1_0 + ki*x2_0 + ci*vx2_0 - Fx )/(mp)

ax2 = np.zeros((iterations,1))
ax2[0,:] = (ki*x1_0 + ci*vx1_0 - (ki+ks)*x2_0 - (ci+cs)*vx2_0 + ks*x3_0 + cs*vx3_0 + Fx)/(mi)

ax3 = np.zeros((iterations,1))
ax3[0,:] = (ks*x2_0 + cs*vx2_0 - ks*x3_0 - cs*vx3_0)/(ms)

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

#accleration array
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
    
    #x - acceleration
    ax1[p,:] = (-(kp+ki)*x1[p,:] - (cp+ci)*vx1[p,:] + ki*x2[p,:] + ci*vx2[p,:] - Fx )/(mp)
    ax2[p,:] = (ki*x1[p,:] + ci*vx1[p,:] - (ki+ks)*x2[p,:] - (ci+cs)*vx2[p,:] + ks*x3[p,:] + cs*vx3[p,:] + Fx)/(mi)
    ax3[p,:] = (ks*x2[p,:] + cs*vx2[p,:] - ks*x3[p,:] - cs*vx3[p,:])/(ms)
    
    #acceleration
    ay1[p,:] = (-(kp+ki)*y1[p,:] - (cp+ci)*vy1[p,:] + ki*y2[p,:] + ci*vy2[p,:] - Fy - mp*9.81 )/(mp)
    ay2[p,:] = (ki*y1[p,:] + ci*vy1[p,:] - (ki+ks)*y2[p,:] - (ci+cs)*vy2[p,:] + ks*y3[p,:] + cs*vy3[p,:] + Fy - mi*9.81)/(mi)
    ay3[p,:] = (ks*y2[p,:] + cs*vy2[p,:] - ks*y3[p,:] - cs*vy3[p,:] - ms*9.81)/(ms)
    
#%%
#xs-xp
x = x2 - x1
xn = vibrn.x

#ys-yp
y = y2 - y1
yn = vibrn.y

#%%
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#plotting
plt.figure(num=1)
plt.subplot(211)
plt.plot(t,x3)
plt.plot(t,y3)
plt.xlabel('iterations')
plt.ylabel('amplitude (m)')
plt.legend(['x_shaft','y_shaft'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,vx3)
plt.plot(t,vy3)
plt.xlabel('iterations')
plt.ylabel('velocity (m/s)')
plt.legend(['x_shaft','y_shaft'],loc = 1)
plt.grid()

plt.show()

#plotting
plt.figure(num=2)
plt.subplot(211)
plt.plot(t,x2)
plt.plot(t,y2)
plt.xlabel('iterations')
plt.ylabel('amplitude (m)')
plt.legend(['x_inner','y_inner'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,vx2)
plt.plot(t,vy2)
plt.xlabel('iterations')
plt.ylabel('velocity (m/s)')
plt.legend(['x_inner','y_inner'],loc = 1)
plt.grid()
plt.show()

#plotting
plt.figure(num=3)
plt.subplot(211)
plt.plot(t,x1)
plt.plot(t,y1)
plt.xlabel('iterations')
plt.ylabel('amplitude (m)')
plt.legend(['x_pedestal','y_pedestal'],loc = 1)
plt.grid()

plt.subplot(212)
plt.plot(t,vx1)
plt.plot(t,vy1)
plt.xlabel('iterations')
plt.ylabel('velocity (m/s)')
plt.legend(['x_pedestal','y_pedestal'],loc = 1)
plt.grid()
plt.show()

#end result
plt.figure(num = 4)
plt.subplot(211)
plt.plot(t,x)
plt.plot(t,xn,'r*')
plt.xlabel('iterations')
plt.ylabel('amplitude (m)')
plt.legend(['x_free','x_forced'],loc = 1)
plt.grid()
plt.subplot(212)
plt.plot(t,y)
plt.plot(t,yn,'r*')
plt.xlabel('iterations')
plt.ylabel('amplitude (m)')
plt.legend(['y_free','y_forced'],loc = 1)
plt.grid()
plt.show()


