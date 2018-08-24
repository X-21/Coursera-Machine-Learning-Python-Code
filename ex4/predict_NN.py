#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from sigmoid import sigmoid


def predict(theta1, theta2, x):
    m = x.shape[0]
    p = np.zeros((m, 1))
    x = np.append(np.ones((m, 1)), x, axis=1)
    h1 = sigmoid(np.dot(x, theta1.T))
    h1 = np.append(np.ones((m, 1)), h1, axis=1)
    h2 = sigmoid(np.dot(h1, theta2.T))
    for i in range(m):
        p[i] = np.argmax(h2[i, :])
    return p + 1
