#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def feature_normalize(x):
    x_norm = np.zeros(x.shape)
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    for i in range(np.shape(x)[0]):
        x_norm[i] = (x[i] - mu) / sigma
    return x_norm, mu, sigma
