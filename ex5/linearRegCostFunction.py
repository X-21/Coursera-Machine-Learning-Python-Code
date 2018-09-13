#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def linear_reg_cost_function(x, y, theta, linear_lambda):
    m = x.shape[0]
    h = np.dot(x, theta)
    j_without_regularization = (1 / (2 * m)) * np.sum((h - y) ** 2)
    j = j_without_regularization + (linear_lambda / (2 * m)) * np.sum(theta[1:] ** 2)
    pass
