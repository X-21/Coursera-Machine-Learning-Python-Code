#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import scipy.optimize as sciopt
from sigmoid import sigmoid


def my_fminunc(x, y, theta):
    return sciopt.minimize(fun=cost_function, x0=theta, args=(x, y), method="TNC", jac=gradient)


def gradient(theta, x, y):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    grad = (x.T.dot(sigmoid(np.dot(x, theta)) - y)) / m

    return grad.flatten()


def cost_function(theta, x, y):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))

    s1 = np.log(sigmoid(np.dot(x, theta)))
    s2 = np.log(1 - sigmoid(np.dot(x, theta)))

    s1 = s1.reshape((m, 1))
    s2 = s2.reshape((m, 1))

    s = y * s1 + (1 - y) * s2
    j = -(np.sum(s)) / m

    return j
