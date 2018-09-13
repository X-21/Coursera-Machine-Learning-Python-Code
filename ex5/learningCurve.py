#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from linearRegCostFunction import linear_reg_cost_function
from trainLinearReg import train_linear_reg


def learning_curve(x, y, xval, yval, curve_lambda):
    m = x.shape[0]
    m_xval = xval.shape[0]

    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    x = np.append(np.ones((m, 1)), x, axis=1)
    xval = np.append(np.ones((m_xval, 1)), xval, axis=1)
    for i in range(m):
        # compute parameter theta
        result = train_linear_reg(x[0:i + 1], y[0:i + 1], curve_lambda)
        theta = result['x']

        # compute training error
        error_train[i] = linear_reg_cost_function(x[0:i + 1], y[0:i + 1], theta, 0)[0]

        # compute cross validation error
        error_val[i] = linear_reg_cost_function(xval, yval, theta, 0)[0]
    return error_train, error_val
