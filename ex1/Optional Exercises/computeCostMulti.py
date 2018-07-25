#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def compute_cost_multi(x, y, theta):
    m = len(y)
    hypothesis = np.dot(x, theta)
    err = (hypothesis - y) ** 2
    return (1 / (2 * m)) * (np.sum(err))
