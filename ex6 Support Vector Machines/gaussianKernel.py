#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import numpy.linalg as linalg


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-((linalg.norm(x1 - x2)) ** 2) / (2 * (sigma ** 2)))
