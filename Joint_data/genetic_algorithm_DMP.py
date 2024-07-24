#!/usr/bin/env python3

import sys
import time
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_srvs.srv import Empty
import numpy as np
import matplotlib.pyplot as plt
import numpy as np 
from sensor_msgs.msg import JointState
import os
import pandas as pd
from math import pi, degrees
from geometry_msgs.msg import PoseStamped
import rospkg
import itertools
import random
import math


T = 0.02
d = np.loadtxt('/home/archit/Desktop/my/DMP Related/DMP trajectories/final_pick_pour.csv', delimiter=',', skiprows=1)
t1 = 0.025
data = d[:, 1:]

def dmp_loop(ax,ax_new,beta,alpha):
    
    
    # for joint in range (0,19,3):

    yd = data[:, 18]  # demonstrated position
    yd_dot = data[:, 19]  # demonstrated velocity
    yd_ddot = np.diff(yd_dot) / t1  # demonstrated acceleration

    # Do not change
    yd = yd[1:]
    yd_dot = yd_dot[1:]
    length_data = len(yd)
    y0 = yd[0]
    g = yd[-1]
    x = 1

    # Basis function parameters
    Nc = 100
    j = np.arange(1, Nc + 1)
    # ax_new = 6
    c = np.exp(-ax_new * (j - 1) / (Nc - 1))
    h = np.zeros(Nc)
    for i in range(Nc - 1):
        h[i] = 1 / ((c[i + 1] - c[i]) ** 2)
    h[Nc - 1] = h[Nc - 2]
    sigma = 1 / np.sqrt(2 * h)



    # ax = 0.4
    tau = 1
    # beta = 12
    # alpha = 4 * beta

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

    goal = yd[-1]
    y = yd [0]
    # goal=float(input(f'Enter the goal position for joint') or yd[-1])
    # y=float(input(f'Enter the initial position for joint') or yd[0])
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

    n = len(y)
    mse = np.mean((yd - y) ** 2)
    print(ax,ax_new)
    print(mse)
    return mse
    
    
   

def create_individual():
    # Generate a random parameter combination
    ax = random.choice(ax1_values)
    ax_new = random.choice(ax_new1_values)
    beta = random.choice(beta_values)
    alpha = random.choice(alpha_values)
    return (ax, ax_new, beta, alpha)
    
# def create_individual():
#     while True:
#         # Generate a random parameter combination
#         ax = random.choice(ax1_values)
#         ax_new = random.choice(ax_new1_values)
#         mse = dmp_loop(ax, ax_new)
#         if not math.isnan(mse):
#             return (ax, ax_new)

def crossover(parent1, parent2):
    # Crossover formula: c of f = c a - η(c a - c b)
    neta = random.uniform(0, 1)
    ax_crossover = parent1[0] - neta * (parent1[0] - parent2[0])
    ax_new_crossover = parent1[1] - neta * (parent1[1] - parent2[1])
    beta_crossover = parent1[2] - neta * (parent1[2] - parent2[2])
    alpha_crossover = parent1[3] - neta * (parent1[3] - parent2[3])
    return (ax_crossover, ax_new_crossover, beta_crossover, alpha_crossover)

def mutate(individual):
    # Mutation formula: X m (1 - X r) N p N v
    mutation_rate = random.uniform(0, 1)
    ax_mutated = individual[0] * (1 - mutation_rate)
    ax_new_mutated = individual[1] * (1 - mutation_rate)
    beta_mutated = individual[2] * (1 - mutation_rate)
    alpha_mutated = individual[3] * (1 - mutation_rate)

    return (ax_mutated, ax_new_mutated, beta_mutated, alpha_mutated)

def genetic_algorithm(population_size, generations):
    population = [create_individual() for _ in range(population_size)]

    for generation in range(generations):
        print(f"Generation {generation + 1} / {generations}")

        # Filter out NaN values from fitness_scores
        fitness_scores = [dmp_loop(ax, ax_new, beta, alpha) for ax, ax_new, beta, alpha in population]
        

        valid_indices = [i for i, score in enumerate(fitness_scores) if not math.isnan(score)]
        valid_population = [population[i] for i in valid_indices]
        fitness_scores = [score for score in fitness_scores if not math.isnan(score)]
        print(fitness_scores)

        num_top_individuals = int(0.2 * population_size)
        top_individuals = [population[i] for i in sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:num_top_individuals]]

        new_population = top_individuals[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(top_individuals, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        best_individual = valid_population[fitness_scores.index(min(fitness_scores))]
        best_mse = min(fitness_scores)
        if best_mse <= 0.00009:
            return best_individual, best_mse
    return best_individual, best_mse


def main():
    best_individual, best_mse = genetic_algorithm(population_size=100, generations=10)
    
    print(f"Best MSE: {best_mse}")
    print(f"Best parameters (ax, ax_new, beta, alpha): {best_individual}")
    T = 0.02
    d = np.loadtxt('/home/archit/Desktop/my/DMP Related/DMP trajectories/final_pick_pour.csv', delimiter=',', skiprows=1)
    t1 = 0.025
    data = d[:, 1:]
  
    yd = data[:, 18]  # demonstrated position
    yd_dot = data[:, 19]  # demonstrated velocity
    yd_ddot = np.diff(yd_dot) / t1  # demonstrated acceleration

    # Do not change
    yd = yd[1:]
    yd_dot = yd_dot[1:]
    length_data = len(yd)
    y0 = yd[0]
    g = yd[-1]
    x = 1

    # Basis function parameters
    Nc = 100
    j = np.arange(1, Nc + 1)
    ax_new = best_individual[1]
    c = np.exp(-ax_new * (j - 1) / (Nc - 1))
    h = np.zeros(Nc)
    for i in range(Nc - 1):
        h[i] = 1 / ((c[i + 1] - c[i]) ** 2)
    h[Nc - 1] = h[Nc - 2]
    sigma = 1 / np.sqrt(2 * h)



    ax = best_individual[0]
    tau = 1
    beta = best_individual[2]
    alpha = best_individual[3]

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

    goal = yd[-1]
    y = yd [0]
    # goal=float(input(f'Enter the goal position for joint') or yd[-1])
    # y=float(input(f'Enter the initial position for joint') or yd[0])
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
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ax1_values = np.arange(0.1,20,0.1)
    ax_new1_values = np.arange(0,15,0.2)
    beta_values = np.arange(0,20,1)
    alpha_values = np.arange(0,40,1)
    main()




















# T = 0.02
#     d = np.loadtxt('/home/archit/Desktop/my/DMP Related/DMP trajectories/final_pick_pour.csv', delimiter=',', skiprows=1)
#     t1 = 0.025
#     data = d[:, 1:]

    
        
        
        
#     yd = data[:, 3]  # demonstrated position
#     yd_dot = data[:, 4]  # demonstrated velocity
#     yd_ddot = np.diff(yd_dot) / t1  # demonstrated acceleration

#     # Do not change
#     yd = yd[1:]
#     yd_dot = yd_dot[1:]
#     length_data = len(yd)
#     y0 = yd[0]
#     g = yd[-1]
#     x = 1

#     # Basis function parameters
#     Nc = 50
#     j = np.arange(1, Nc + 1)
#     ax_new = best_ax_new
#     c = np.exp(-ax_new * (j - 1) / (Nc - 1))
#     h = np.zeros(Nc)
#     for i in range(Nc - 1):
#         h[i] = 1 / ((c[i + 1] - c[i]) ** 2)
#     h[Nc - 1] = h[Nc - 2]
#     sigma = 1 / np.sqrt(2 * h)



#     ax = best_ax
#     tau = 1
#     beta = 12
#     alpha = 4 * beta

#     # Learn the forcing function (f) i.e. the weights
#     f_target = np.zeros(length_data)
#     phi = np.zeros((length_data, Nc))
#     zeta = np.zeros(length_data)

#     for k in range(length_data):
#         f_target[k] = tau ** 2 * yd_ddot[k] - alpha * (beta * (g - yd[k]) - tau * yd_dot[k])
#         for i in range(Nc):
#             phi[k, i] = np.exp(-((x - c[i]) ** 2) / (2 * sigma[i] ** 2))
#         zeta[k] = x * (g - y0)
#         x = x + (T / tau) * (-ax * x)

#     w = np.zeros(Nc)
#     for i in range(Nc):
#         Phi = np.diag(phi[:, i])
#         w[i] = np.dot(zeta, np.dot(Phi, f_target)) / np.dot(zeta, np.dot(Phi, zeta))

#     goal = yd[-1]
#     y = yd [0]
#     # goal=float(input(f'Enter the goal position for joint') or yd[-1])
#     # y=float(input(f'Enter the initial position for joint') or yd[0])
#     y1=y
#     z = 0
#     x = 1


#     f = np.zeros(length_data)
#     acc = np.zeros(length_data)
#     y = np.zeros(length_data)
#     z = np.zeros(length_data)
#     x = np.zeros(len(yd))
#     phi_n = np.zeros(Nc)

#     x[0]=1
#     y[0]=y1
#     z[0]=0
#     data_list=[]


#     for k in range(length_data - 1):
#         for i in range(Nc):
#             phi_n[i] = np.exp(-((x[k] - c[i])**2) / (2 * sigma[i]**2))
        
#         f[k] = 1 * (np.dot(w, phi_n) * x[k] * (goal - y0) / np.sum(phi_n))
        
        
#         z[k+1] = z[k] + (T / tau) * (alpha * (beta * (goal - y[k]) - z[k]) + f[k])
#         y[k+1] = y[k] + (T / tau) * z[k]
#         x[k+1] = x[k] + (T / tau) * (-ax * x[k])
#         acc[k+1] = (1 / tau) * (alpha * (beta * (goal - y[k]) - z[k]) + f[k])




    # plt.figure()

    # # Subplot 1
    # plt.subplot(3, 2, 1)
    # plt.plot(yd, 'r', linewidth=2)
    # plt.plot(y, '--b', linewidth=2)
    # plt.plot(length_data - 1, g, 'r*')
    # plt.title('Position vs time')
    # plt.legend(['demonstrated position', 'y'])

    # # Subplot 2
    # plt.subplot(3, 2, 3)
    # plt.plot(yd_dot, 'r', linewidth=2)
    # plt.plot(z, '--b', linewidth=2)
    # plt.title('Velocity vs time')
    # plt.legend(['demonstrated velocity', 'v'])

    # # Subplot 3
    # plt.subplot(3, 2, 5)
    # plt.plot(yd_ddot, 'r', linewidth=2)
    # plt.plot(acc, '--b', linewidth=2)
    # plt.title('Acceleration vs time')
    # plt.legend(['demonstrated acceleration', 'a'])

    # # Subplot 4
    # plt.subplot(3, 2, 2)
    # plt.plot(x, 'r', linewidth=2)
    # plt.title('x vs time')

    # # Subplot 5
    # plt.subplot(3, 2, 4)
    # plt.plot(f_target, 'r', linewidth=2)
    # plt.plot(f, '--b', linewidth=2)
    # plt.title('f_{target}, f vs time')
    # plt.legend(['f_{target}', 'f_{actual}'])

    # # Subplot 6
    # plt.subplot(3, 2, 6)
    # for k in range(Nc):
    #     plt.plot(phi[:, k], linewidth=2)
    # plt.title('phi vs time')

    # # Adjust layout to prevent subplot overlap
    # plt.tight_layout()

    # # Display the plots
    # plt.show()