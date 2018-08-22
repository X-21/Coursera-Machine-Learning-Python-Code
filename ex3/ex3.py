#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from scipy.io import loadmat
from displayData import display_data
from lrCostFunction import lr_cost_function
from oneVsAll import one_vs_all
from predictOneVsAll import predict_one_vs_all


def pause_func():
    print('Program paused. Press enter to continue.\n')
    while input() != '':
        pass


def load_mat_data(filename):
    return loadmat(filename)


if __name__ == '__main__':
    # Setup the parameters you will use for this part of the exercise
    # 20x20 Input Imc c.reshape(2,8,order='F')= a.ravel(order='F')ages of Digits
    input_layer_size = 400
    # 10 labels, from 1 to 10
    num_labels = 10
    # =========== Part 1: Loading and Visualizing Data =============
    data = load_mat_data("ex3data1.mat")
    X = data['X']
    y = data['y']
    m = len(y)
    # Load Training Data
    print('Loading and Visualizing Data ...\n')
    # Randomly select 100 data points to display
    shuffle_100_X = np.arange(0, m, 1, dtype=int)
    np.random.shuffle(shuffle_100_X)
    sel = X[shuffle_100_X[0:100], :]
    display_data(sel)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ============ Part 2a: Vectorize Logistic Regression ============
    # Test case for lrCostFunction
    print('\nTesting lrCostFunction() with regularization')
    theta_t = np.array([[-2], [-1], [1], [2]])
    X_t = np.append(np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F') / 10, axis=1)
    y_t = np.array([[1], [0], [1], [0], [1]])
    lambda_t = 3
    J, grad = lr_cost_function(theta_t, X_t, y_t, lambda_t)
    print('\nCost: \n', J, '\nExpected cost: 2.534819\n')
    print('Gradients:\n', grad, '\nExpected gradients:\n', ' 0.146561\n -0.548558\n  0.724722\n  1.398003\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()
    # ============ Part 2b: One-vs-All Training ============
    print('\nTraining One-vs-All Logistic Regression...\n')
    ova_lambda = 0.1
    all_theta = one_vs_all(X, y, num_labels, ova_lambda)
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================ Part 3: Predict for One-Vs-All ================
    pred = predict_one_vs_all(all_theta, X) + 1
    print('\nTraining Set Accuracy: \n', np.mean((pred == y).astype(np.float64) * 100))
