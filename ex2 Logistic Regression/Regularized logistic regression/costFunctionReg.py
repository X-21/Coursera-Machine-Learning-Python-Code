#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import sys

sys.path.append("../")
from sigmoid import sigmoid


def cost_function_reg(theta, x, y, reg_lambda):
    m = len(y)
    sub1 = np.dot((-1 * y).T, np.log(sigmoid(np.dot(x, theta))))
    sub2 = np.dot((1 - y.T), np.log(1 - sigmoid(np.dot(x, theta))))
    j = (1 / m) * np.sum(sub1 - sub2)
    j = j + (reg_lambda / (2 * m)) * np.sum((theta[1:]) ** 2)
    grad = (1 / m) * np.dot(x.T, (sigmoid(np.dot(x, theta)) - y))
    grad = grad + (reg_lambda / m) * theta
    grad[0] = grad[0] - (reg_lambda / m) * theta[0]
    return j, grad
