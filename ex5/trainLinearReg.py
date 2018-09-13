#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from linearRegCostFunction import linear_reg_cost_function
import scipy.optimize as sciopt

grad = np.arange(0)


def train_linear_reg(x, y, train_lambda):
    initial_theta = np.zeros((x.shape[1], 1))
    global grad
    grad = np.zeros((x.shape[1], 1))
    return sciopt.minimize(fun=cost_function, x0=initial_theta, args=(x, y, train_lambda), method="TNC", jac=gradient)


def cost_function(theta, x, y, cf_lambad):
    j, new_grad = linear_reg_cost_function(x, y, theta, cf_lambad)
    global grad
    grad = new_grad
    return j


def gradient(*args):
    global grad
    return grad.flatten()
