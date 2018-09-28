#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from featureNormalize import feature_normalize
from gradientDescentMulti import gradient_descent_multi
from normalEqn import normal_eqn


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_data(filename):
    data_load = np.loadtxt(filename, delimiter=",")
    return data_load


if __name__ == '__main__':
    # ================ Part 1: Feature Normalization ================
    print('Loading data ...\n')
    # Load Data
    data = load_data('ex1data2.txt')
    data = np.split(data, [2], axis=1)
    X = data[0]
    y = data[1]
    m = len(y)
    # Print out some data points
    print('First 10 examples from the dataset: \n')
    for i in range(10):
        print(' x = [%.0f %.0f], y = %.0f \n' % (X[i][0], X[i][1], y[i]))
    # pause_func()

    # Scale features and set them to zero mean
    print('Normalizing Features ...\n')
    X, mu, sigma = feature_normalize(X)
    # Add intercept term to X
    X = np.append(np.ones((m, 1)), X, axis=1)

    # ================ Part 2: Gradient Descent ================
    print('Running gradient descent ...\n')
    # Number of iterations (loops)
    num_iters = 400
    # Try some other values of alpha
    alpha = 1
    theta = np.zeros((3, 1))
    theta, J_history_0 = gradient_descent_multi(X, y, theta, alpha, num_iters)
    print('theta is \n', theta, '\n')

    alpha = 0.3
    theta = np.zeros((3, 1))
    theta, J_history_1 = gradient_descent_multi(X, y, theta, alpha, num_iters)
    print('theta is \n', theta, '\n')

    alpha = 0.01
    theta = np.zeros((3, 1))
    theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)
    print('theta is \n', theta, '\n')

    plt.ion()
    plt.figure()
    j_history_plt_x = np.linspace(1, num_iters, num_iters).reshape(400, 1)
    plt.plot(j_history_plt_x, J_history_0, "-r")
    plt.plot(j_history_plt_x, J_history_1, "-g")
    plt.plot(j_history_plt_x, J_history, "-b")
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.pause(0.5)
    plt.close()
    # Display gradient descent's result
    print('Theta computed from gradient descent: \n', theta, '\n')

    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.dot(
        np.array(([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]])),
        theta)
    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n' % price)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================ Part 3: Normal Equations ================
    print('Solving with normal equations...\n')
    data = load_data('ex1data2.txt')
    data = np.split(data, [2], axis=1)
    X = data[0]
    y = data[1]
    X = np.append(np.ones((m, 1)), X, axis=1)
    # Calculate the parameters from the normal equation
    theta = normal_eqn(X, y)
    # Display normal equation's result
    print('Theta computed from the normal equations: \n  ', theta, '\n\n')
    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.dot(
        np.array(([1, 1650, 3])),
        theta)
    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n' % price)
