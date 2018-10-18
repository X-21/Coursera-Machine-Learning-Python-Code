# !/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def cofi_cost_func(params, y, r, num_users, num_movies, num_features, lambda_co_fi):
    x = np.reshape(params[0:num_movies * num_features], (num_movies, num_features), order='F')
    theta = np.reshape(params[num_movies * num_features:], (num_users, num_features), order='F')

    # You need to return the following values correctly
    # j = 0
    # x_grad = np.zeros(x.shape)
    # theta_grad = np.zeros(theta.shape)

    diff = np.dot(x, theta.T) - y

    # unregularized
    j = (1 / 2) * np.sum((diff * r) ** 2)
    # regularized term for Theta
    j += (lambda_co_fi / 2) * np.sum(theta ** 2)
    # regularized term for X
    j += (lambda_co_fi / 2) * np.sum(x ** 2)

    # unregularized
    x_grad = np.dot(diff * r, theta)

    # unregularized
    theta_grad = np.dot((diff * r).T, x)

    # regularized
    x_grad += lambda_co_fi * x

    # regularized
    theta_grad += lambda_co_fi * theta

    grad = np.hstack((np.ravel(x_grad, order='F'), np.ravel(theta_grad, order='F')))

    return j, grad
