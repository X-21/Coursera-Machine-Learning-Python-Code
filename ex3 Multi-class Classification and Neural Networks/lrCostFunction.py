#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from sigmoid import sigmoid


def lr_cost_function(theta, x, y, lr_lambda):
    m = len(y)
    sub1 = np.dot((-1 * y).T, np.log(sigmoid(np.dot(x, theta))))
    sub2 = np.dot((1 - y.T), np.log(1 - sigmoid(np.dot(x, theta))))
    j = (1 / m) * np.sum(sub1 - sub2)
    j = j + (lr_lambda / (2 * m)) * np.sum((theta[1:]) ** 2)
    grad = (1 / m) * np.dot(x.T, (sigmoid(np.dot(x, theta)) - y))
    grad = grad + (lr_lambda / m) * theta
    grad[0] = grad[0] - (lr_lambda / m) * theta[0]
    return j, grad
