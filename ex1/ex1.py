#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from computeCost import compute_cost
from gradientDescent import gradient_descent

def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    return data


def warm_up_exercise():
    print(np.eye(5))


if __name__ == '__main__':
    # ==================== Part 1: Basic Function ====================
    print('Running warmUpExercise ... \n')
    print('5x5 Identity Matrix: \n')
    warm_up_exercise()
    # pause_func()
    # ======================= Part 2: Plotting =======================
    data1 = load_data("ex1data1.txt")
    X = data1[:, 0]
    y = data1[:, 1]
    m = len(y)
    X = X.reshape(m, 1)
    y = y.reshape(m, 1)
    plt.plot(X, y, 'rx')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()
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