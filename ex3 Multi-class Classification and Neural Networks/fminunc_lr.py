#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import scipy.optimize as sciopt
from sigmoid import sigmoid


def my_fminunc_lr(x, y, theta, lr_lambda):
    return sciopt.minimize(fun=cost_function, x0=theta, args=(x, y, lr_lambda), method="L-BFGS-B", jac=gradient)


def gradient(theta, x, y, lr_lambda):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    grad = (x.T.dot(sigmoid(np.dot(x, theta)) - y)) / m
    grad = grad + (lr_lambda / m) * theta
    grad[0] = grad[0] - (lr_lambda / m) * theta[0]

    return grad.flatten()


def cost_function(theta, x, y, lr_lambda):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    s1 = np.log(sigmoid(np.dot(x, theta)))
    s2 = np.log(1 - sigmoid(np.dot(x, theta)))

    s1 = s1.reshape((m, 1))
    s2 = s2.reshape((m, 1))

    s = y * s1 + (1 - y) * s2
    j = -(np.sum(s)) / m
    j = j + (lr_lambda / (2 * m)) * np.sum((theta[1:]) ** 2)

    return j
