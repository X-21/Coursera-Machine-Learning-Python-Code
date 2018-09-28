# !/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def feature_normalize(x):
    x_norm = x.copy()
    # np.shape(x)[1] is the column of x
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    for i in range(np.shape(x)[0]):
        x_norm[i] = (x[i] - mu) / sigma
    return x_norm, mu, sigma
