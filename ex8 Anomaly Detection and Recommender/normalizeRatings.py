# !/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def normalize_ratings(y, r):
    m, n = y.shape
    y_mean = np.zeros((m, 1))
    y_norm = np.zeros((m, n))
    for i in range(m):
        idx = np.where(r[i] == 1)[0]
        y_mean[i] = np.mean(y[i, idx])
        y_norm[i, idx] = y[i, idx] - y_mean[i]
    return y_norm, y_mean
