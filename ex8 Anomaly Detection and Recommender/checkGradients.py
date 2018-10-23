# !/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
from computeNumericalGradient import compute_numerical_gradient
from cofiCostFunc import cofi_cost_func


def check_nn_gradients(check_nn_lambda=0):
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
    x = np.random.randn(x_t.shape)
    theta = np.random.randn(theta_t.shape)
    num_users = y.shape[1]
    num_movies = y.shape[0]
    num_features = theta_t.shape[1]

    compute_numerical_gradient(
        cofi_cost_func
        , np.hstack((np.ravel(x, order='F'), np.ravel(theta, order='F')))
        , y, r
        , num_users, num_movies, num_features, 0
    )


