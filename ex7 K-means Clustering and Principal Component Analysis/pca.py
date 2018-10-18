#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def pca(x):
    m = x.shape[0]
    sigma = (1 / m) * (np.dot(x.T, x))
    u, s, v = np.linalg.svd(sigma)
    return u, s
