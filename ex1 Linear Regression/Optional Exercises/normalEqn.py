#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def normal_eqn(x, y):
    temp_x = np.dot(x.T, x)
    matrix_x_inverse = np.mat(temp_x).I.getA()
    temp_x = np.dot(matrix_x_inverse, x.T)
    return np.dot(temp_x, y)
