#!/usr/bin/env python3
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import numpy as np 
import random
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from termcolor import colored
import pandas as pd


    global g, y0,yd,yd_dot,yd_ddot,d,t1,length_data,T,path
    
    T = 0.02
    path = '/home/rahul/catkin_workspace/merged_file.csv'
    
    d = np.loadtxt(path, delimiter=',', skiprows=1)
    t1 = 0.024
    data = d[:, 1:]
    # for joint in range (0,19,3):
    
    t1 = 0.025
    yd = data[:, joint]  # demonstrated position
    yd_dot = np.diff(yd) / t1 # demonstrated velocity
    yd_ddot = np.diff(yd_dot) / t1 # demonstrated acceleration

    yd = yd[1:]
    yd = yd[1:]
    yd_dot = yd_dot[1:]
    g = yd[-1]
    y0 = yd[0]


    # Do not change
    # yd = yd[1:]
    # yd_dot = yd_dot[1:]
    length_data = len(yd)
    # y0 = yd[0]
    # g = yd[-1]
    x = 1



    ax=float(input("Enter the ax value: "))
    ax_new=float(input("Enter the ax_new value: "))
    beta=float(input("Enter the beta value: "))
    alpha = float(input("Enter aplha value: "))
    Nc=int(input("Enter the Nc value: "))
    

    j = np.arange(1, Nc + 1)

    c = np.exp(-ax_new * (j - 1) / (Nc - 1))
    h = np.zeros(Nc)
    for i in range(Nc - 1):
        h[i] = 1 / ((c[i + 1] - c[i]) ** 2)
    h[Nc - 1] = h[Nc - 2]
    sigma = 1 / np.sqrt(2 * h)




    tau = 1

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

    #goal = g
    #y = y0
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


    for k in range(length_data - 1):
        for i in range(Nc):
            phi_n[i] = np.exp(-((x[k] - c[i])**2) / (2 * sigma[i]**2))
        
        f[k] = 1 * (np.dot(w, phi_n) * x[k] * (goal - y0) / np.sum(phi_n))
        
        
        z[k+1] = z[k] + (T / tau) * (alpha * (beta * (goal - y[k]) - z[k]) + f[k])
        y[k+1] = y[k] + (T / tau) * z[k]
        x[k+1] = x[k] + (T / tau) * (-ax * x[k])
        acc[k+1] = (1 / tau) * (alpha * (beta * (goal - y[k]) - z[k]) + f[k])
        data_list.append([y[k+1]])

    path = 'xyz{joint}.csv'.format(joint=joint)
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_list)





if __name__ == "__main__":

    for joint in range(3):
        main()
    

    import csv
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Predicted Trajectory Visualization
    with open('/home/rahul/Lab/DMP-main/Cartesian/xyz0.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        x = [float(row[0]) for row in reader]

    with open('/home/rahul/Lab/DMP-main/Cartesian/xyz1.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        y = [float(row[0]) for row in reader]

    with open('/home/rahul/Lab/DMP-main/Cartesian/xyz2.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        z = [float(row[0]) for row in reader]

    # # Desired Trajectory Visualization
    csv_file_path = path
    d = pd.read_csv(csv_file_path, skiprows=1)

    # Extract the first three columns by position instead of name
    xd = d.iloc[:, 1]
    yd = d.iloc[:, 2]
    zd = d.iloc[:, 3]

    # Check if xd, yd, zd are not empty before accessing their last elements
    if len(xd) > 0 and xd.iloc[-1] != -1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', marker='o', label='Predicted')  # Predicted Trajectory
        ax.scatter(xd, yd, zd, c='r', marker='o', label='Desired')  # Desired Trajectory

        # Plot blue star at initial point with larger size
        ax.scatter(x[0], y[0], z[0], c='b', marker='*', s=200, label='Initial Point')

        # Plot black star at end point with larger size
        ax.scatter(x[-1], y[-1], z[-1], c='k', marker='*', s=200, label='End Point')

        # Add lines connecting initial and final points
        ax.plot([x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]], color='g')
        ax.plot([xd.iloc[0], xd.iloc[-1]], [yd.iloc[0], yd.iloc[-1]], [zd.iloc[0], zd.iloc[-1]], color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.title('3D Plot of X, Y, Z Coordinates')
        plt.legend()
        plt.savefig("3d.png")
        plt.show()
        
    else:
        print("Desired trajectory data is empty or contains invalid values.")
