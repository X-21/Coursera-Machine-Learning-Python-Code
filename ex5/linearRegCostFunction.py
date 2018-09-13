#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def linear_reg_cost_function(x, y, theta, linear_lambda):
    theta = np.reshape(theta, (theta.shape[0], 1))
    m = x.shape[0]
    h = np.dot(x, theta)
    j_without_regularization = (1 / (2 * m)) * np.sum((h - y) ** 2)
    j = j_without_regularization + (linear_lambda / (2 * m)) * np.sum(theta[1:] ** 2)
    grad_without_regularization = (np.dot(x.T, h - y) / m)
    grad = grad_without_regularization + (linear_lambda / m) * theta
    grad[0] = grad_without_regularization[0]
    return j, grad
