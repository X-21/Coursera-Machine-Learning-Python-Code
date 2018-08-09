#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from computeCostMulti import compute_cost_multi


def gradient_descent_multi(x, y, theta, alpha, num_iters):
    m = len(y)
    j_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        hypothesis = np.dot(x, theta)
        sub = hypothesis - y
        theta = theta - (alpha / m) * (np.dot(x.T, sub))
        j_history[i] = compute_cost_multi(x, y, theta)
    return theta, j_history
