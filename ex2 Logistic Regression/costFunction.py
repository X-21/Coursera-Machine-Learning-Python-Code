#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from sigmoid import sigmoid


def cost_function(theta, x, y):
    m = len(y)
    sub1 = np.dot((-1 * y).T, np.log(sigmoid(np.dot(x, theta))))
    sub2 = np.dot((1 - y.T), np.log(1 - sigmoid(np.dot(x, theta))))
    j = (1 / m) * np.sum(sub1 - sub2)
    grad = (1 / m) * np.dot(x.T, (sigmoid(np.dot(x, theta)) - y))
    return j, grad
