#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3D

from computeCost import compute_cost
from gradientDescent import gradient_descent


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_data(filename):
    data_load = np.loadtxt(filename, delimiter=",")
    return data_load


def warm_up_exercise():
    print(np.eye(5))


if __name__ == '__main__':
    # ==================== Part 1: Basic Function ====================
    print('Running warmUpExercise ... \n')
    print('5x5 Identity Matrix: \n')
    warm_up_exercise()
    # pause_func()

    # ======================= Part 2: Plotting =======================
    data = load_data("ex1data1.txt")
    X = data[:, 0]
    y = data[:, 1]
    m = len(y)
    # reshape that will convert vector into matrix
    X = X.reshape(m, 1)
    y = y.reshape(m, 1)

    plt.ion()
    plt.figure()
    plt.plot(X, y, 'rx', label="Training data")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    # pause_func()

    # =================== Part 3: Cost and Gradient descent ===================
    X = np.append(np.ones((m, 1)), X, axis=1)  # Add a column of ones to x
    theta = np.zeros((2, 1))  # initialize fitting parameters
    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01
    print('\nTesting the cost function ...\n')
    # compute and display initial cost
    J = compute_cost(X, y, theta)
    print('With theta = [0 ; 0]\nCost computed = %f\n' % J[0])
    print('Expected cost value (approx) 32.07\n')
    J = compute_cost(X, y, np.array(([-1], [2])))
    print('\nWith theta = [-1 ; 2]\nCost computed = %f\n' % J[0])
    print('Expected cost value (approx) 54.24\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    print('\nRunning Gradient Descent ...\n')
    # run gradient descent
    theta = gradient_descent(X, y, theta, alpha, iterations)
    # print theta to screen
    print('Theta found by gradient descent:\n')
    print(theta)
    print('Expected theta values (approx)\n')
    print(' -3.6303\n  1.1664\n\n')

    plt.plot(X[:, 1], np.dot(X, theta), '-', label="Linear regression")
    plt.legend()
    plt.pause(0.5)
    plt.close()

    predict1 = np.dot(np.array(([1, 3.5])), theta)
    print('For population = 35,000, we predict a profit of ', predict1 * 10000, '\n')
    predict2 = np.dot(np.array(([1, 7])), theta)
    print('For population = 70,000, we predict a profit of ', predict2 * 10000, '\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...\n')
    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # Fill out J_vals
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array(([theta0_vals[i]], [theta1_vals[j]]))
            J_vals[i, j] = compute_cost(X, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T

    # Surface plot
    surface_figure = plt.figure()
    ax = Axes3D(surface_figure)
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap="jet")
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.pause(0.5)
    plt.close()

    plt.figure()
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.plot(theta[0], theta[1], "rx", markersize=20)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$', rotation=0)
    plt.pause(0.5)
    plt.close()
