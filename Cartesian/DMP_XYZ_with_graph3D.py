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

ax1_values = np.arange(0.1,10,0.2)
ax_new1_values = np.arange(0,15,0.2)
beta_values = np.arange(0,15,0.5)
alpha_values = np.arange(0,30,1)
Nc_values = np.arange(50,200,10)



the_best_individuals = []
def dmp_loop(ax,ax_new,beta,alpha, Nc):
    global g, y0,yd,yd_dot,yd_ddot,d,t1,length_data,T,path
    
    T = 0.02
    path = '/home/rahul/catkin_workspace/DMP_cartesian_data3.csv'
    
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

    # print(len(yd), len(yd_dot), len(yd_ddot))
    # g=float(input(f'Enter the goal position for joint') or yd[-1])
    # y0=float(input(f'Enter the initial position for joint') or yd[0])
    g = yd[-1]
    y0 = yd[0]


    # Do not change
    # yd = yd[1:]
    # yd_dot = yd_dot[1:]
    length_data = len(yd)
    # y0 = yd[0]
    # g = yd[-1]
    x = 1

    # Basis function parameters
    Nc = int(Nc)
    # print("Number of basis functions: ", Nc)
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

    goal = g
    y = y0
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
    # print(ax,ax_new)
    # print(mse)
    return mse
    
    


def create_individual():
    # Generate a random parameter combination
    ax = random.choice(ax1_values)
    ax_new = random.choice(ax_new1_values)
    beta = random.choice(beta_values)
    alpha = random.choice(alpha_values)
    Nc = int(random.choice(Nc_values))
    return (ax, ax_new, beta, alpha, Nc)
    
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
    Nc_crossover = int(parent1[4] - neta * (parent1[4] - parent2[4]))
    # print("Crossover occured",Nc_crossover)
    return (ax_crossover, ax_new_crossover, beta_crossover, alpha_crossover, Nc_crossover)

def mutate(individual):
    # Mutation formula: X m (1 - X r) N p N v
    mutation_rate = random.uniform(0, -0.1)
    ax_mutated = individual[0] * (1 - mutation_rate)
    ax_new_mutated = individual[1] * (1 - mutation_rate)
    beta_mutated = individual[2] * (1 - mutation_rate)
    alpha_mutated = individual[3] * (1 - mutation_rate)
    Nc_mutated = int(individual[4] * (1 - mutation_rate))
    # print("Mutation occured",Nc_mutated)
    return (ax_mutated, ax_new_mutated, beta_mutated, alpha_mutated, Nc_mutated)

def genetic_algorithm(population_size, generations):
    population = [create_individual() for _ in range(population_size)]

    for generation in range(generations):
        print(f"Generation {generation + 1} / {generations}")

        # Calculate fitness scores and handle NaN values
        fitness_scores = [dmp_loop(ax, ax_new, beta, alpha, Nc) for ax, ax_new, beta, alpha, Nc in population]
        
        valid_indices = [i for i, score in enumerate(fitness_scores) if not math.isnan(score)]
        valid_population = [population[i] for i in valid_indices]
        fitness_scores = [score for score in fitness_scores if not math.isnan(score)]
        # print(fitness_scores)

        # Selection of the top individuals
        num_top_individuals = int(0.4 * population_size)

        top_individuals = [population[i] for i in sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:num_top_individuals]]
        print("Top individuals:", len(top_individuals))

        new_population = top_individuals[:]
        parent_pair = set()
        print("length of new_population",len(new_population))
        while len(new_population) < population_size:
            print("------------------------------------------------------------")
            parent1 = random.choice(top_individuals)
            available_parents = [p for p in top_individuals if (tuple(parent1),tuple(p)) not in parent_pair and p != parent1]
            parent2 = random.choice(available_parents)
            parent_pair.add((tuple(parent1), tuple(parent2)))
            # print("Randomly selected parents", parent1, parent2)
            
            child = crossover(parent1, parent2)
            # print("crossover occured between",child)
            # print("Crossed over child", child)
            child = mutate(individual=child)
            # print("Mutation occured",child)
            print("------------------------------------------------------------")
            # print("Mutated child", child)
            new_population.append(child)
            # print("lenghth of new_populaton", len(new_population))


        population = new_population
        print("length of new population",len(population))
        # print(population)
        # print("length of new population",len(population))
        
        
        best_individual = valid_population[fitness_scores.index(min(fitness_scores))]
        best_mse = min(fitness_scores)
        
        the_best_individuals.append((best_individual, best_mse))
        
        # print(f"Best individual: {valid_population[fitness_scores.index(best_mse)]}")
        print(f"Best MSE: {best_mse:.6f}")
        best_ind = the_best_individuals
        # print(f"Best individual all time: {best_ind}")
        if best_mse <= 0.0005:
            return best_individual, best_mse
        
    return best_individual, best_mse



def main():


    best_individual, best_mse = genetic_algorithm(population_size=50, generations=5)
    
    print(f"Best MSE: {best_mse}")



    print(f"Best parameters (ax, ax_new, beta, alpha): {best_individual}")

    
    T = 0.02
    # d = np.loadtxt('/home/rahul/testing_new_initial.csv', delimiter=',', skiprows=1)
    t1 = 0.025
    data = d[:, 1:]



    length_data = len(yd)
    # y0 = yd[0]
    # g = yd[-1]
    x = 1


    Nc = int(best_individual[4])
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

    goal = g
    y = y0
    #goal=float(input(f'Enter the goal position for joint') or yd[-1])
    #y=float(input(f'Enter the initial position for joint') or yd[0])
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


def Merge_file():
    import pandas as pd
    import os

# Load the three CSV files
    file1 = pd.read_csv('/home/rahul/Lab/DMP-main/Cartesian/xyz0.csv', header=None)
    file2 = pd.read_csv('/home/rahul/Lab/DMP-main/Cartesian/xyz1.csv', header=None)
    file3 = pd.read_csv('/home/rahul/Lab/DMP-main/Cartesian/xyz2.csv', header=None)

# Add a serial number column
    max_length = max(len(file1), len(file2), len(file3))
    serial_numbers = pd.Series(range(1, max_length + 1))

# Create a DataFrame with the serial numbers and the data from the files
    merged = pd.DataFrame({
        'Serial No': serial_numbers,
        'p_x': file1[0].reindex(range(max_length)).reset_index(drop=True),
        'p_y': file2[0].reindex(range(max_length)).reset_index(drop=True),
        'p_z': file3[0].reindex(range(max_length)).reset_index(drop=True),
    })

# Save the merged dataframe to a new CSV file

    merged.to_csv("/home/rahul/catkin_workspace/merged_file.csv", index=False)

    print("Files have been merged successfully into 'merged_file.csv'")


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
        
        #merge file xyz0 ,xyz1, xyz2
Merge_file()
        
    
