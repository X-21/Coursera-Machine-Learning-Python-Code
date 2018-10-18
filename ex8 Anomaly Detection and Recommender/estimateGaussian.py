# !/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def estimate_gaussian(x):
    m = x.shape[0]
    mu = np.mean(x, axis=0)
    err = x - mu
    sigma2 = (np.sum(err ** 2, axis=0)) / m
    return mu, sigma2
