#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def compute_cost(x, y, theta):
    m = len(y)
    h_theta = np.dot(x, theta)
    err = h_theta - y
    err_sum = sum(err ** 2)
    j = err_sum / (2 * m)
    return j
