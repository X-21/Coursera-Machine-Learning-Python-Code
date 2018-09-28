#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np


def predict_one_vs_all(all_theta, x):
    m = x.shape[0]
    p = np.zeros((m, 1))
    x = np.append(np.ones((m, 1)), x, axis=1)
    prob = np.dot(x, all_theta.T)
    for i in range(m):
        p[i] = np.argmax(prob[i, :])
    return p
