#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from plotData import plot_data
from costFunction import cost_function
from fminunc import my_fminunc
from plotDecisionBoundary import plot_decision_boundary
from sigmoid import sigmoid
from predict import predict


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_data(filename):
    data_load = np.loadtxt(filename, delimiter=",")
    return data_load


if __name__ == '__main__':
    data = load_data('ex2data1.txt')
    data = np.split(data, [2], axis=1)
    X = data[0]
    y = data[1]

    # ==================== Part 1: Plotting ====================
    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
    plot_data(X, y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(["Admitted", "Not admitted"])
    plt.pause(3)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ============ Part 2: Compute Cost and Gradient ============
    #  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape
    # Add intercept term to x and X_test
    X = np.append(np.ones((m, 1)), X, axis=1)
    # Initialize fitting parameters
    initial_theta = np.zeros((n + 1, 1))
    # Compute and display initial cost and gradient
    cost, grad = cost_function(initial_theta, X, y)
    print('Cost at initial theta (zeros): \n', cost, '\nExpected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros): \n', grad,
          '\nExpected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array(([-24], [0.2], [0.2]))
    cost, grad = cost_function(test_theta, X, y)
    print('\nCost at test theta: \n', cost, '\nExpected cost (approx): 0.218\n')
    print('Gradient at test theta: \n', grad, '\nExpected gradients (approx):\n 0.043\n 2.566\n 2.647\n')
    print('\nProgram paused. Press enter to continue.\n')
    # pause_func()

    # ============= Part 3: Optimizing using fminunc  =============
    result = my_fminunc(X, y, initial_theta)
    theta = result["x"]
    # Print theta to screen
    print('Cost at theta found by fminunc: \n', result["fun"], '\nExpected cost (approx): 0.203\n')
    print('theta: \n', theta, '\nExpected theta (approx):\n -25.161\n 0.206\n 0.201\n')
    # Plot Boundary
    plot_decision_boundary(theta, X, y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    # Legend, specific for the exercise
    plt.legend(loc='upper right')
    plt.pause(2)
    plt.close()
    print('\nProgram paused. Press enter to continue.\n')
    # pause_func()

    # ============== Part 4: Predict and Accuracies ==============
    prob = sigmoid(np.dot(np.array([1, 45, 85]), theta))
    print('For a student with scores 45 and 85, we predict an admission probability of \n', prob)
    print('\nExpected value: 0.775 +/- 0.002\n\n')
    # Compute accuracy on our training set
    p = predict(theta, X).reshape(100, 1)
    print('Train Accuracy: ', np.mean((p == y)) * 100)
    print('\nExpected accuracy (approx): 89.0\n')
