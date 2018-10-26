# !/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
from computeNumericalGradient import compute_numerical_gradient
from cofiCostFunc import cofi_cost_func


def disp(disp_list):
    size_no_0 = disp_list[0].size
    disp_num = len(disp_list)
    for i in range(disp_num):
        if disp_list[0].shape != disp_list[i].shape:
            return False
    for i in range(size_no_0):
        print("\n", end='')
        for j in range(disp_num):
            print("{: >17.11f}".format(disp_list[j][i]), end=' ')
    print("\n")


def check_gradients(check_nn_lambda=0):
    # Create small problem
    x_t = np.random.rand(4, 3)
    theta_t = np.random.rand(5, 3)

    # Zap out most entries
    y = np.dot(x_t, theta_t.T)
    rand_x_axis, rand_y_axis = np.where(np.random.rand(y.shape[0], y.shape[1]) > 0.5)
    y[rand_x_axis, rand_y_axis] = 0
    y_not_0_x_axis, y_not_0_y_axis = np.where(y == 0)

    r = np.ones(y.shape)
    r[y_not_0_x_axis, y_not_0_y_axis] = 0

    y[rand_x_axis, rand_y_axis] = 0

    # Run Gradient Checking
    x = np.random.randn(x_t.shape[0], x_t.shape[1])
    theta = np.random.randn(theta_t.shape[0], theta_t.shape[1])
    num_users = y.shape[1]
    num_movies = y.shape[0]
    num_features = theta_t.shape[1]

    numgrad = compute_numerical_gradient(
        cofi_cost_func,
        np.hstack((np.ravel(x, order='F'), np.ravel(theta, order='F'))),
        y, r,
        num_users, num_movies, num_features, check_nn_lambda
    )
    cost, grad = cofi_cost_func(
        np.hstack((np.ravel(x, order='F'), np.ravel(theta, order='F'))),
        y, r, num_users,
        num_movies, num_features, check_nn_lambda
    )
    disp([numgrad, grad])
    print(
        'The above two columns you get should be very similar.\n'
        '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(
        'If your cost function implementation is correct, then \n'
        'the relative difference will be small (less than 1e-9). \n\n'
        'Relative Difference: %s\n' % diff)
