#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from trainLinearReg import train_linear_reg
from linearRegCostFunction import linear_reg_cost_function


def validation_curve(x, y, xval, yval):
    # Selected values of lambda (you should not change this)
    lambda_vec = np.array(([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]))
    len_of_vec = len(lambda_vec)
    # You need to return these variables correctly.
    error_train = np.zeros((len_of_vec, 1))
    error_val = np.zeros((len_of_vec, 1))

    for i in range(len_of_vec):
        lambda_temp = lambda_vec[i]

        # compute parameter theta (learning)
        result = train_linear_reg(x, y, lambda_temp)
        theta = result['x']

        # compute training error
        j, grad = linear_reg_cost_function(x, y, theta, 0)
        error_train[i] = j

        # compute cross validation error
        j, grad = linear_reg_cost_function(xval, yval, theta, 0)
        error_val[i] = j
    return lambda_vec, error_train, error_val
