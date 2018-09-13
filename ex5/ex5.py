#!/usr/bin/env python
# -*- coding=utf-8 -*-

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from linearRegCostFunction import linear_reg_cost_function


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_data(filename):
    return loadmat(filename)


# Exercise 5 | Regularized Linear Regression and Bias-Variance
if __name__ == '__main__':
    # =========== Part 1: Loading and Visualizing Data =============
    print('Loading and Visualizing Data ...\n')
    # Load from ex5data1:
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = load_data('ex5data1.mat')
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']
    # m = Number of examples
    m = X.shape[0]
    # Plot training data
    plt.ion()
    plt.figure()
    plt.plot(X, y, 'rx', markersize=10)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    #plt.pause(3)
    plt.close()
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =========== Part 2: Regularized Linear Regression Cost =============
    theta = np.array([[1], [1]])
    J = linear_reg_cost_function(np.append(np.ones((m, 1)), X, axis=1)
                                 , y, theta, 1)

    print('Cost at theta = [1 ; 1]: %f \n(this value should be about 303.993192)\n', J)

    print('Program paused. Press enter to continue.\n')
    # pause_func()
    a = 1
