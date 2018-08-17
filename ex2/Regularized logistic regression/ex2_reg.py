#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mapFeature import map_feature
from costFunctionReg import cost_function_reg
from fminunc_reg import my_fminunc_reg
import sys

sys.path.append("../")
from plotData import plot_data
from plotDecisionBoundary import plot_decision_boundary
from predict import predict


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_data(filename):
    data_load = np.loadtxt(filename, delimiter=",")
    return data_load


if __name__ == '__main__':
    data = load_data('ex2data2.txt')
    data = np.split(data, [2], axis=1)
    X = data[0]
    y = data[1]
    plot_data(X, y)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(["y = 1", "y = 0"])
    plt.pause(1.5)
    plt.close()

    # =========== Part 1: Regularized Logistic Regression ============
    X = map_feature(X[:, 0], X[:, 1])
    # Initialize fitting parameters
    initial_theta = np.zeros((X.shape[1], 1))
    # Set regularization parameter lambda to 1
    reg_lambda = 1
    # Compute and display initial cost and gradient for regularized logistic regression
    cost, grad = cost_function_reg(initial_theta, X, y, reg_lambda)
    print('Cost at initial theta (zeros): ', cost, '\nExpected cost (approx): 0.693\n')
    np.set_printoptions(suppress=True)
    print('Gradient at initial theta (zeros) - first five values only:\n', grad[0: 5])
    print('\nExpected gradients (approx) - first five values only:\n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')
    print('\nProgram paused. Press enter to continue.\n')
    # pause_func()

    # Compute and display cost and gradient with all-ones theta and lambda = 10
    test_theta = np.ones((X.shape[1], 1))
    cost, grad = cost_function_reg(test_theta, X, y, 10)
    print('Cost at test theta (with lambda = 10): ', cost, '\nExpected cost (approx): 3.16\n')
    np.set_printoptions(suppress=True)
    print('Gradient at test theta - first five values only:\n', grad[0: 5])
    print('\nExpected gradients (approx) - first five values only:\n 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')
    print('\nProgram paused. Press enter to continue.\n')
    # pause_func()

    # ============= Part 2: Regularization and Accuracies =============
    reg_lambda = 1
    result = my_fminunc_reg(X, y, initial_theta, reg_lambda)
    theta = result["x"]
    # Plot Boundary
    plot_decision_boundary(theta, X, y)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.title('lambda = %g' % reg_lambda)
    plt.pause(2)
    plt.close()
    # Compute accuracy on our training set
    p = predict(theta, X).reshape(118, 1)
    print('Train Accuracy: ', np.mean((p == y)) * 100)
    print('\nExpected accuracy (approx): 83.1\n')
