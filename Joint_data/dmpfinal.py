#!/usr/bin/env python3

import sys
import time
import rospy
import geometry_msgs.msg
from math import pi
from std_srvs.srv import Empty
import numpy as np
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
from math import pi, degrees
from geometry_msgs.msg import PoseStamped
import rospkg
import math as m
 


T = 0.02
d = np.loadtxt('/home/rahul/catkin_workspace/Joint_State_data.csv', delimiter=',', skiprows=1)
t1 = 0.025
data = d[:, 1:]

for joint in range (0,19,3):

    yd = data[:, joint]  # demonstrated position
    yd_dot = data[:, joint+1]  # demonstrated velocity
    yd_ddot = np.diff(yd_dot) / t1  # demonstrated acceleration

    # Do not change
    yd = yd[1:]
    yd_dot = yd_dot[1:]
    length_data = len(yd)
    y0 = yd[0]
    g = yd[-1]
    x = 1

    # Basis function parameters
    Nc=int(input("Enter the Nc value: "))
    ax_new=float(input("Enter the ax_new value: "))
    
    j = np.arange(1, Nc + 1)
                
    c = np.exp(-ax_new * (j - 1) / (Nc - 1))
    h = np.zeros(Nc)
    for i in range(Nc - 1):
        h[i] = 1 / ((c[i + 1] - c[i]) ** 2)
    h[Nc - 1] = h[Nc - 2]
    sigma = 1 / np.sqrt(2 * h)

    # DMP parameters

    ax=float(input("Enter the ax value: "))
    beta=float(input("Enter the beta value: "))

    tau = 1

    alpha = float(input("Enter alpha value:"))

    # Learn the forcing function (f) i.e. the weights
    f_target = np.zeros(length_data)
    phi = np.zeros((length_data, Nc))
    zeta = np.zeros(length_data)

    for k in range(length_data):
        f_target[k] = tau ** 2 * yd_ddot[k] - alpha * (beta * (g - yd[k]) - tau * yd_dot[k])
        for i in range(Nc):
            phi[k, i] = np.exp(-((x - c[i]) ** 2) / (2 * sigma[i] ** 2))
        zeta[k] = x * (g - y0)
        x = x + (T / tau) * (-ax * x)

    w = np.zeros(Nc)
    for i in range(Nc):
        Phi = np.diag(phi[:, i])
        w[i] = np.dot(zeta, np.dot(Phi, f_target)) / np.dot(zeta, np.dot(Phi, zeta))


    goal=float(input(f'Enter the goal position for joint') or yd[-1])
    y=float(input(f'Enter the initial position for joint') or yd[0])
    y1=y
    z = 0
    x = 1


    f = np.zeros(length_data)
    acc = np.zeros(length_data)
    y = np.zeros(length_data)
    z = np.zeros(length_data)
    x = np.zeros(len(yd))
    phi_n = np.zeros(Nc)

    x[0]=1
    y[0]=y1
    z[0]=0
    data_list=[]
    print(y[0])

    for k in range(length_data - 1):
        for i in range(Nc):
            phi_n[i] = np.exp(-((x[k] - c[i])**2) / (2 * sigma[i]**2))
        
        f[k] = 1 * (np.dot(w, phi_n) * x[k] * (goal - y0) / np.sum(phi_n))
        
        
        # z[k+1] = z[k] + (T / tau) * (alpha * (beta * (goal - y[k]) - z[k]) + f[k])
        # y[k+1] = y[k] + (T / tau) * z[k]
        # x[k+1] = x[k] + (T / tau) * (-ax * x[k])
        # acc[k+1] = (1 / tau) * (alpha * (beta * (goal - y[k]) - z[k]) + f[k])
        
        z[k+1] = z[k] + (T / tau) * (alpha * (beta * m.tanh(goal - y[k]) - m.tanh(z[k])) + f[k]) 
        y[k+1] = y[k] + (T / tau) * z[k]     #output position 
        x[k+1] = x[k] + (T / tau) * (-ax * x[k])
        acc[k+1] = (1 / tau) * (alpha * (beta * m.tanh(goal - y[k]) - m.tanh(z[k])) + f[k])

        
    # Plots
    plt.figure()

    # Subplot 1
    plt.subplot(3, 2, 1)
    plt.plot(yd, 'r', linewidth=2)
    plt.plot(y, '--b', linewidth=2)
    plt.plot(length_data - 1, g, 'r*')
    plt.title('Position vs time')
    plt.legend(['demonstrated position', 'y'])

    # Subplot 2
    plt.subplot(3, 2, 3)
    plt.plot(yd_dot, 'r', linewidth=2)
    plt.plot(z, '--b', linewidth=2)
    plt.title('Velocity vs time')
    plt.legend(['demonstrated velocity', 'v'])

    # Subplot 3
    plt.subplot(3, 2, 5)
    plt.plot(yd_ddot, 'r', linewidth=2)
    plt.plot(acc, '--b', linewidth=2)
    plt.title('Acceleration vs time')
    plt.legend(['demonstrated acceleration', 'a'])

    # Subplot 4
    plt.subplot(3, 2, 2)
    plt.plot(x, 'r', linewidth=2)
    plt.title('x vs time')

    # Subplot 5
    plt.subplot(3, 2, 4)
    plt.plot(f_target, 'r', linewidth=2)
    plt.plot(f, '--b', linewidth=2)
    plt.title('f_{target}, f vs time')
    plt.legend(['f_{target}', 'f_{actual}'])

    # Subplot 6
    plt.subplot(3, 2, 6)
    for k in range(Nc):
        plt.plot(phi[:, k], linewidth=2)
    plt.title('phi vs time')

    # Adjust layout to prevent subplot overlap
    plt.tight_layout()

    # Display the plots
    plt.show()
        


    
